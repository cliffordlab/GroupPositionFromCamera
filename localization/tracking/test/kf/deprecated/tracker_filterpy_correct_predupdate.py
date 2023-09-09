'''
This is the version that fixes:
1) matching between actual prediction and detection
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
	def __init__(self, dist_threshold, max_frame_skipped, max_trace_length):
		super(Tracker, self).__init__()
		self.dist_threshold = dist_threshold
		self.max_frame_skipped = max_frame_skipped
		self.max_trace_length = max_trace_length
		self.trackId = 0
		self.tracks = []

	def update(self, detections):
		if len(self.tracks) == 0:
			for i in range(detections.shape[0]):
				# print (detections[i])
				# assert False
				track = Tracks(detections[i], self.trackId)
				self.trackId +=1
				self.tracks.append(track)

		N = len(self.tracks)
		M = len(detections)
		cost = []
		for i in range(N):
			self.tracks[i].predict()
			diff = np.linalg.norm(self.tracks[i].prediction - detections.reshape(-1,2), axis=1)
			# print (diff)
			cost.append(diff)
		# assert False

		cost = np.array(cost)*0.1
		row, col = linear_sum_assignment(cost)
		assignment = [-1]*N
		for i in range(len(row)):
			assignment[row[i]] = col[i]

		un_assigned_tracks = []

		for i in range(len(assignment)):
			if assignment[i] != -1:
				if (cost[i][assignment[i]] > self.dist_threshold):
					assignment[i] = -1
					un_assigned_tracks.append(i)
				else:
					self.tracks[i].skipped_frames +=1

		del_tracks = []
		for i in range(len(self.tracks)):
			if self.tracks[i].skipped_frames > self.max_frame_skipped :
				del_tracks.append(i)

		if len(del_tracks) > 0:
			for i in range(len(del_tracks)):
				del self.tracks[i]
				del assignment[i]

		for i in range(len(detections)):
			if i not in assignment:
				track = Tracks(detections[i], self.trackId)
				self.trackId +=1
				self.tracks.append(track)


		for i in range(len(assignment)):
			if(assignment[i] != -1):
				self.tracks[i].skipped_frames = 0
				self.tracks[i].update(detections[assignment[i]])
			self.tracks[i].trace.append(self.tracks[i].prediction)
