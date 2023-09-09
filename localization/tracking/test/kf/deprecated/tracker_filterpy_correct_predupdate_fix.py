'''
This is the final version that fixes:
1) matching between actual prediction and detection
2) deleting tracks
3) adding new tracks
'''

import numpy as np 
from scipy.optimize import linear_sum_assignment
from collections import deque
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


class Tracks(object):
	"""docstring for Tracks"""
	def __init__(self, detection, trackId):
		super(Tracks, self).__init__()

		self.dt = 1
		self.KF = KalmanFilter(dim_x=4, dim_z=2)
		self.KF.F = np.array([[1, 0, self.dt, 0],
													[0, 1, 0, self.dt],
													[0, 0, 1, 0],
													[0, 0, 0, 1]])
		self.KF.H = np.array([[1, 0, 0, 0],
													[0, 1, 0, 0]])
		# Measurement Noise Covariance
		self.KF.R *= 1 
		# Error covariance
		self.KF.P *= 1
		# Process Noise Covariance
		self.KF.Q = Q_discrete_white_noise(dim=2, dt=self.dt, var=1,
																			 block_size=2, order_by_dim=False)
		if 0:
			# from NickNair (works well.)
			self.KF.Q = np.matrix([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],[0, (self.dt**4)/4, 0, (self.dt**3)/2],
															[(self.dt**3)/2, 0, self.dt**2, 0],[0, (self.dt**3)/2, 0, self.dt**2]])
		# print (self.KF.R)
		# print (self.KF.P)
		# print (self.KF.Q)
		# assert False

		# print (detection.shape)
		# print (self.KF.x.shape)
		self.KF.x[:2] = detection.reshape(2,1)

		self.prediction = detection.reshape(1,2)

		# self.trace = deque(maxlen=20)
		self.trace = []
		
		self.trackId = trackId
		self.skipped_frames = 0

	def predict(self):
		self.KF.predict()
		self.prediction = np.array(self.KF.x[:2]).reshape(1,2)
	
	def update(self, detection):
		self.KF.update(np.matrix(detection).reshape(2,1))
		self.prediction = np.array(self.KF.x[:2]).reshape(1,2)

class Tracker(object):
	"""docstring for Tracker"""
	def __init__(self, dist_threshold, max_frame_skipped, max_trace_length=None):
		super(Tracker, self).__init__()
		self.dist_threshold = dist_threshold
		self.max_frame_skipped = max_frame_skipped
		self.max_trace_length = max_trace_length
		self.trackId = 0
		self.tracks = {}

	def update(self, detections):
		if len(self.tracks) == 0:
			for i in range(detections.shape[0]):
				# print (detections[i])
				# assert False
				track = Tracks(detections[i], self.trackId)
				self.tracks[self.trackId] = track
				self.trackId +=1

		N = len(self.tracks)
		M = len(detections)
		index_trackId_assign = {}
		cost = []
		for i, trackId in enumerate(self.tracks):
			index_trackId_assign[i] = trackId

			self.tracks[trackId].predict()
			diff = np.linalg.norm(self.tracks[i].prediction - detections.reshape(-1,2), axis=1)
			# print (diff)
			cost.append(diff)
		# assert False

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
				trackId_det_assign[trackId] = -1
				del self.tracks[trackId]
		
		# Add new tracks for unassigned detections:
		for i in det_trackId_assign:
			if det_trackId_assign[i] == -1:
				track = Tracks(detections[i], self.trackId)
				self.tracks[self.trackId] = track
				self.trackId +=1
		
		# Update and record trace
		for trackId in self.tracks:
			if trackId in trackId_det_assign \
			and trackId_det_assign[trackId] != -1:
				c = trackId_det_assign[trackId]['col']
				self.tracks[trackId].skipped_frames = 0
				self.tracks[trackId].update(detections[c])
			self.tracks[trackId].trace.append(self.tracks[trackId].prediction)

