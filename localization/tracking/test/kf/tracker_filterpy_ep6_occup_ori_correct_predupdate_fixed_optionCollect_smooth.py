'''
This is the final version that fixes:
1) matching between actual prediction and detection
2) deleting tracks
3) adding new tracks
4) making state collection as option
'''

import numpy as np 
from scipy.optimize import linear_sum_assignment
from collections import deque
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import copy

class Tracks(object):
	"""docstring for Tracks"""
	def __init__(self, detection, trackId):
		super(Tracks, self).__init__()

		self.w = 10  # orientation from movement
		self.dt = 1
		self.KF = KalmanFilter(dim_x=8, dim_z=4) # x, y, u, v, dx, dy, du, dv
		self.KF.F = np.array([[1, 0, 0, 0, self.dt, 0, 0, 0],
													[0, 1, 0, 0, 0, self.dt, 0, 0],
													[0, 0, 1, 0, self.w, 0, self.dt, 0],
													[0, 0, 0, 1, 0, self.w, 0, self.dt],
													[0, 0, 0, 0, 1, 0, 0, 0],
													[0, 0, 0, 0, 0, 1, 0, 0],
													[0, 0, 0, 0, 0, 0, 1, 0],
													[0, 0, 0, 0, 0, 0, 0, 1]])
		self.KF.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
													[0, 1, 0, 0, 0, 0, 0, 0],
													[0, 0, 1, 0, 0, 0, 0, 0],
													[0, 0, 0, 1, 0, 0, 0, 0]])
		# Measurement Noise Covariance
		self.KF.R *= 1 
		# Error covariance
		self.KF.P *= 1
		# Process Noise Covariance
		self.KF.Q = Q_discrete_white_noise(dim=2, dt=self.dt, var=1,
																			 block_size=4, order_by_dim=False)
		# print (self.KF.R)
		# print (self.KF.P)
		# print (self.KF.Q)
		# assert False

		# print (detection.shape)
		# print (self.KF.x.shape)
		self.KF.x[:4] = detection.reshape(4,1)

		self.pos_state = detection.reshape(1,4)

		# collect state information
		self.trace = [] # collect position state (= x[:dim_z]) either from prediction (when no detection is availalbe) or update (when detection is available)

		self.means = [] # collect either x (state) either from prediction (when no detection is availalbe) or update (when detection is available)
		self.covariance = [] # collect P (posterior/state covariance either from prediction (when no detection is availalbe) or update (when detection is available)

		self.trackId = trackId
		self.skipped_frames = 0

		self.ts = [] # timestep of each sample in datetime object

	def predict(self):
		self.KF.predict()
		self.pos_state = np.array(self.KF.x[:4]).reshape(1,4)
	
	def update(self, detection):
		self.KF.update(np.matrix(detection).reshape(4,1))
		self.pos_state = np.array(self.KF.x[:4]).reshape(1,4)

	def smooth(self):
		assert len(self.means) > 0, 'state mean had to be collected'
		assert len(self.covariance) > 0, 'state covariacne had to be collected'

		# https://filterpy.readthedocs.io/en/latest/_modules/filterpy/kalman/kalman_filter.html#KalmanFilter.batch_filter
		# https://filterpy.readthedocs.io/en/latest/_modules/filterpy/kalman/kalman_filter.html#KalmanFilter.rts_smoother

		mu = np.array(self.means)
		cov = np.array(self.covariance)

		Xs, Ps, Ks, Pp = self.KF.rts_smoother(mu, cov)
		self.means_smoothed = Xs
		self.covariance_smoothe = Ps
		print (f'smoothed and killed ... {self.trackId}')

class Tracker(object):
	"""docstring for Tracker"""
	def __init__(self, dist_threshold, max_frame_skipped,
							 collect_states=False, max_trace_length=None):
		super(Tracker, self).__init__()
		self.dist_threshold = dist_threshold
		self.max_frame_skipped = max_frame_skipped
		self.collect_states = collect_states
		self.max_trace_length = max_trace_length
		self.trackId = 0
		self.tracks = {} # Live tracking container
		self.tracks_collected = {} # this is to collect final "smoothed" tracks  before deleting for live tracking

	def update(self, detections=None, ts=None):
		
		if detections is not None and len(self.tracks) == 0:
			for i in range(detections.shape[0]):
				# print (detections[i])
				# assert False
				track = Tracks(detections[i], self.trackId)
				self.tracks[self.trackId] = track

				if self.collect_states and self.max_trace_length is not None:
					self.tracks[self.trackId].ts = deque(maxlen=self.max_trace_length)
					self.tracks[self.trackId].trace = deque(maxlen=self.max_trace_length)
					self.tracks[self.trackId].means = deque(maxlen=self.max_trace_length)
					self.tracks[self.trackId].covariance = deque(maxlen=self.max_trace_length)

				self.trackId +=1
		
		if detections is None:
			# just predict next state and exit
			for trackId in self.tracks:
				self.tracks[trackId].predict()
				self.tracks[trackId].skipped_frames += 1

				if self.collect_states:
					self.tracks[trackId].ts.append(ts)
					self.tracks[trackId].trace.append(self.tracks[trackId].pos_state)
					self.tracks[trackId].means.append(self.tracks[trackId].KF.x)
					self.tracks[trackId].covariance.append(self.tracks[trackId].KF.P)

			return 0

		N = len(self.tracks)
		M = len(detections)
		index_trackId_assign = {}
		cost = []
		for i, trackId in enumerate(self.tracks):
			index_trackId_assign[i] = trackId

			self.tracks[trackId].predict()
			loc_pred = self.tracks[trackId].pos_state[:,:2]
			# print (loc_pred)
			loc_det = detections[:,:2].reshape(-1,2)
			# print (loc_det)
			
			loc_err = np.linalg.norm(loc_pred - loc_det, axis=1)
			# print (self.tracks[trackId].pos_state.shape)
			# print (detections.shape)
			# print ((loc_pred - loc_det).shape)
			# print (loc_err)
			# print (loc_err.shape)
			# assert False

			diff = loc_err

			if 0:
				ori_pred = self.tracks[trackId].pos_state[:,2:]
				# print (ori_pred)
				ori_pred_norm = np.sqrt(np.sum(ori_pred**2, axis=1)).reshape((-1, 1))
				# print (ori_pred_norm)			
				ori_pred_norm[ori_pred_norm==0] = 1.
				ori_pred /= ori_pred_norm
				# print (ori_pred)
				# print ('--')

				ori_det = detections[:,2:].reshape(-1,2)
				# print (ori_det)
				ori_det_norm = np.sqrt(np.sum(ori_det**2, axis=1)).reshape((-1, 1))
				# print (ori_det_norm)
				ori_det_norm[ori_det_norm==0] = 1.
				ori_det /= ori_det_norm
				# print (ori_det)
				# print ('--')

				neg_cos_sim = -np.sum(ori_pred*ori_det, axis=1)
				# print (ori_pred*ori_det)
				# print (np.sum(ori_pred*ori_det, axis=1))
				# print (neg_cos_sim)
				# print (neg_cos_sim.shape)

				diff += 5*neg_cos_sim
				# print (diff)
				# assert False


			cost.append(diff)

		cost = np.array(cost)
		row, col = linear_sum_assignment(cost)

		# Collect information for prediction and detection
		det_trackId_assign = {}
		for i in range(M):
			det_trackId_assign[i] = -1
		
		for r, c in zip(row, col):
			trackId = index_trackId_assign[r]
			det_trackId_assign[c] = trackId

		trackId_det_assign = {}
		for i in range(N):
			trackId = index_trackId_assign[i]
			trackId_det_assign[trackId] = -1
		
		for r, c in zip(row, col):
			trackId = index_trackId_assign[r]
			trackId_det_assign[trackId] = {'row': r,
																		'col': c,
																		'cost': cost[r, c]}
		# check skipped frames
		for trackId in trackId_det_assign:
			assignment = trackId_det_assign[trackId]
			if assignment == -1:
				''' Unassigned '''
				self.tracks[trackId].skipped_frames += 1
			elif assignment['cost'] > self.dist_threshold:
				''' Assigned but too fat to be true '''
				trackId_det_assign[trackId] = -1
				det_trackId_assign[assignment['col']] = -1
				self.tracks[trackId].skipped_frames += 1
		
		# delete tracks no longer needed
		del_trackIds = []
		for trackId in self.tracks:
			if self.tracks[trackId].skipped_frames > self.max_frame_skipped:
				del_trackIds.append(trackId)
		
		if len(del_trackIds) > 0:
			for trackId in del_trackIds:
				assert trackId not in self.tracks_collected
				trackId_det_assign[trackId] = -1
				# smooth and add to collection before deleting the live track
				self.tracks[trackId].smooth()
				self.tracks_collected[trackId] = copy.deepcopy(self.tracks[trackId])
				del self.tracks[trackId]
		
		# Add new tracks for unassigned detections:
		for i in det_trackId_assign:
			if det_trackId_assign[i] == -1:
				track = Tracks(detections[i], self.trackId)
				self.tracks[self.trackId] = track

				if self.collect_states and self.max_trace_length is not None:
					self.tracks[self.trackId].trace = deque(maxlen=self.max_trace_length)
					self.tracks[self.trackId].means = deque(maxlen=self.max_trace_length)
					self.tracks[self.trackId].covariance = deque(maxlen=self.max_trace_length)
				
				self.trackId +=1
		
		# Update and record trace
		for trackId in self.tracks:
			if trackId in trackId_det_assign \
			and trackId_det_assign[trackId] != -1:
				c = trackId_det_assign[trackId]['col']
				self.tracks[trackId].skipped_frames = 0
				self.tracks[trackId].update(detections[c])

			if self.collect_states:
				self.tracks[trackId].ts.append(ts)
				self.tracks[trackId].trace.append(self.tracks[trackId].pos_state)
				self.tracks[trackId].means.append(self.tracks[trackId].KF.x)
				self.tracks[trackId].covariance.append(self.tracks[trackId].KF.P)

	def close(self):
		'''
		Close Multi-object tracking with smoothing end deleting live tracks
		'''
		del_trackIds = list(self.tracks.keys())
		for trackId in del_trackIds:
			assert trackId not in self.tracks_collected
			self.tracks[trackId].smooth()
			self.tracks_collected[trackId] = copy.deepcopy(self.tracks[trackId])
			del self.tracks[trackId]
