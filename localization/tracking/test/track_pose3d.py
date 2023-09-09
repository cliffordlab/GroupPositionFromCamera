'''
- tracking poses and finding new person showing up in the camera
- eventually track across cameras
'''

import os
import matplotlib.pyplot as plt
import numpy as np
from shutil import copyfile
import scipy.io
from pprint import pprint
from matplotlib import cm
from utils import test
import pickle as cp
import torch

if 0:
	pi_ip = '155'

	year = '2021'
	month = '10'
	day = '06'
	hour = '14'

	minute = 34
	minute = 43
	# second = 37
else:
	pi_ip = '108'

	year = '2021'
	month = '11'
	day = '15'
	hour = '12'
	minute = 24
	second = 00

is_filter_noisy_pose = True
track_method = 'hungarian'
distance = 'kps' # kps / bbox

dir_save = '/tmp/hhyeokk'
dir_save_head = '/home/hhyeokk/Research/CEP/exp'

dir_save += '/track_pose3d'
dir_save_head += '/track_pose3d'

if is_filter_noisy_pose:
	dir_save += '_filtered'
	dir_save_head += '_filtered'
if track_method == 'hungarian':
	dir_save += f'_{track_method}'
	dir_save_head += f'_{track_method}'
if distance == 'kps':
	dir_save += f'_{distance}'
	dir_save_head += f'_{distance}'

camera_time = f'{year}{month}{day}_{hour}{minute}_pi{pi_ip}'
dir_save += f'/{camera_time}'
dir_save_head += f'/{camera_time}'
os.makedirs(dir_save, exist_ok=True)
os.makedirs(dir_save_head, exist_ok=True)

dir_ep6  = '/labs/cliffordlab/data/EP6'
dir_op = dir_ep6 + '/openpose'
dir_pi = dir_op +f'/{year}/{month}/{day}/hour_{hour}/pi{pi_ip}.pi.bmi.emory.edu'

dir_kp = dir_pi +'/keypoints'
dir_vid = dir_pi +'/videos'

# frame_name = f'cam87_{year}{month}{day}_{hour}{minute}{second}'
# file_frame = dir_vid +f'/{frame_name}.jpg'
# frame = plt.imread(file_frame)
# print ('load ...', file_frame)

# file_save_head = dir_save_head + f'/{frame_name}.jpg'
# copyfile(file_frame, file_save_head)
# print ('save in ...', file_save_head)

# frame = np.flip(frame, axis=1) # flip
# h, w, c = frame.shape
# center_line = float(frame.shape[1])/2

kp_name = f'pi87_{year}{month}{day}_{hour}{minute}'
file_kp = dir_kp +f'/{kp_name}.mat'
if not os.path.exists(file_kp):
	kp_name = f'pi{pi_ip}.pi.bmi.emory.edu{year}{month}{day}_{hour}{minute}'
	file_kp = dir_kp +f'/{kp_name}.mat'
keypoints = scipy.io.loadmat(file_kp)
print ('load ...', file_kp)

file_save_head = dir_save_head + f'/{kp_name}.mat'
copyfile(file_kp, file_save_head)
print ('save in ...', file_save_head)

# flip all keypoint sequence
list_timestep = list(keypoints.keys())
# pprint (list_timestep)
list_timestep = [item for item in list_timestep if 'pi87' in item]
# pprint (list_timestep)
list_timestep.sort()
# pprint (list_timestep)

#TODO: get the consecutive time block.


# create frames.npy for fast load
file_save = dir_save_head + '/frames.npy'
if os.path.exists(file_save):
	list_frames = np.load(file_save)
	print ('load from ...', file_save)
else:
	list_frames = []
	for ts in list_timestep:
		# load frame
		frame_name = f'cam87_{ts[5:]}'
		file_frame = dir_vid +f'/{frame_name}.jpg'
		frame = plt.imread(file_frame)
		print ('load ...', file_frame)

		# flip frame
		frame = np.flip(frame, axis=1) # flip
		list_frames.append(frame)

	list_frames = np.array(list_frames)
	np.save(file_save, list_frames)
	print ('save in ...', file_save)

list_kps = []
list_dets = []

fid_names = {}
fid = 0
for ts in list_timestep:
	fid_names[fid] = ts
	frame = list_frames[fid]
	fid += 1

	h, w, c = frame.shape
	center_line = float(frame.shape[1])/2

	# load keypoints
	kps = keypoints[ts]
	# print (kps.shape)

	kp_sum = np.sum(kps, axis=(1,2))
	kps = kps[kp_sum > 0,...]
	# print (kps.shape)

	# flip keypoints
	center_line = float(frame.shape[1])/2
	kps[:,:,1] -= center_line
	kps[:,:,1] = -kps[:,:,1]
	kps[:,:,1] += center_line
	# print (kps.shape)

	list_kps.append(kps)

	# get bounding box
	x1_y1 = np.amin(kps, axis=1)
	x2_y2 = np.amax(kps, axis=1)
	# print (x1_y1)
	# print (x2_y2)
	
	conf = np.ones((x1_y1.shape[0], 1))

	dets = np.concatenate((x1_y1, x2_y2, conf), axis=1)
	# print (dets)
	# assert False
	list_dets.append(dets)

	# print (ts)
	# print (kps.shape)

if is_filter_noisy_pose:
	# remove identical poses for all frame
	# possibly due to the pattern on the floor.

	_list_kps = []
	_list_dets = []
	for t in range(len(list_kps)):
		kps_curr = list_kps[t]
		if t == len(list_kps)-1:
			kps_next = list_kps[t-1]
		else:
			kps_next = list_kps[t+1]
		# print (kps_next.shape)

		val_kps = np.ones((kps_curr.shape[0],), dtype=bool)
		for i in range(len(kps_curr)):
			kps = kps_curr[i].reshape((1, 17, 2))
			# print (kps)

			is_noise = np.sum(np.abs(kps_next - kps), axis=(1,2))
			if np.any(is_noise == 0):
				val_kps[i] = False
		# print (t)
		# print (val_kps, np.sum(val_kps))

		kps_curr = kps_curr[val_kps]
		# print (kps_curr.shape)
		_list_kps.append(kps_curr)

		dets = list_dets[t]
		dets = dets[val_kps]
		_list_dets.append(dets)
	
	list_kps = _list_kps
	list_dets = _list_dets

	file_save = dir_save + '/list_kps.p'
	cp.dump(list_kps, open(file_save, 'wb'))

	file_save_head = dir_save_head + '/list_kps.p'
	copyfile(file_save, file_save_head)
	print ('save in ...', file_save_head)
	
	

# print (len(list_kps))
# print (len(list_dets))
# for kps, dets in zip(list_kps, list_dets):
# 	print (kps.shape, dets.shape)
# assert False

# sort tracking 
if track_method == 'hungarian':

	from utils import hungarian 
	mot_tracker = hungarian(distance=distance)

	track_kps = {}
	fid = 0
	for dets in list_dets:
		trackers = mot_tracker.update(dets)
		for id, pid in enumerate(trackers):
			if pid not in track_kps:
				track_kps[pid] = {'frame': [], 'id': []}
			
			track_kps[pid]['frame'].append(fid)
			track_kps[pid]['id'].append(id)
		fid += 1
	# assert False

else:

	import sys
	sys.path.append('/home/hhyeokk/Research/CEP/repos/sort')
	# https://github.com/abewley/sort/issues/12

	from sort import Sort
	mot_tracker = Sort()

	track_kps = {}
	fid = 0
	for dets in list_dets:
		trackers = mot_tracker.update(dets)
		# print (dets)
		# print (trackers)

		dets = dets[:,:4]

		for d in trackers:
			pid = d[4]
			if pid not in track_kps:
				track_kps[pid] = {'frame': [], 'id': []}

			# find idx in dets
			bbs = d[:4]
			_dets = np.sum(np.abs(dets - bbs.reshape((1,4))), axis=1)
			# print (_dets)
			i_dets = np.argmin(_dets)

			# try:
			# 	i_dets = np.where(_dets < 1e-10)[0][0] 
			# 	# print (i_dets)
			# except:
			# 	print (fid, '|', pid)
			# 	print (dets)
			# 	print (bbs)
			# 	print (_dets)
			# 	assert False

			track_kps[pid]['frame'].append(fid)
			track_kps[pid]['id'].append(i_dets)
			# assert False

		fid += 1
		# pprint(track_kps)
		# assert False

if 0:
	for pid in track_kps:
		print (pid, '|', len(track_kps[pid]['frame']))
		print ('frame:', track_kps[pid]['frame'])
		print ('id:', track_kps[pid]['id'])
		print ('-----------')
# assert False

file_save = dir_save + '/track_kps.p'
cp.dump(track_kps, open(file_save, 'wb'))

file_save_head = dir_save_head + '/track_kps.p'
copyfile(file_save, file_save_head)
print ('save in ...', file_save_head)


# frame based
frame_kps = {}
for pid in track_kps:

	list_fid = track_kps[pid]['frame']
	list_id = track_kps[pid]['id']

	for fid, id in zip(list_fid, list_id):
		if fid not in frame_kps:
			frame_kps[fid] = {
				'pid': [],
				'id': []}
		
		frame_kps[fid]['pid'].append(pid)
		frame_kps[fid]['id'].append(id)

if 0:
	for fid in frame_kps:
		print (fid, '|', len(frame_kps[fid]['pid']))
		print ('pid:', frame_kps[fid]['pid'])
		print ('id:', frame_kps[fid]['id'])
		print ('-----------')
# assert False

cmap = cm.get_cmap('rainbow')
tester = test()

if 0:
	# plot input
	dir_save_kps = dir_save + f'/input'
	dir_save_head_kps = dir_save_head + f'/input'
	os.makedirs(dir_save_kps, exist_ok=True)
	os.makedirs(dir_save_head_kps, exist_ok=True)

	for fid in frame_kps:

		frame = list_frames[fid]
		ts = fid_names[fid]
		frame_name = f'cam87_{ts[5:]}'

		fig, ax = plt.subplots()
		ax.imshow(frame)

		kps = list_kps[fid]
		c_list = cmap(np.linspace(0, 1, len(kps)))

		for i in range(len(kps)):
			kp = kps[i]
			ax = tester.plot_kp(kp, ax, c_list[i])

		fig_name = f'{frame_name}.png'
		file_save = dir_save_kps + f'/{fig_name}'
		plt.savefig(file_save)

		plt.close()
		
		# copy to head node
		file_save_head = dir_save_head_kps + f'/{fig_name}'
		copyfile(file_save, file_save_head)

		print ('save in ...', file_save_head)

# assert False

# plot
# print (len(track_kps))
# print(track_kps.keys())
# print(np.amax(list(track_kps.keys())))
# assert False
max_pid = int(np.amax(list(track_kps.keys())))
c_list = cmap(np.linspace(0, 1, max_pid))

# per frame
if 0:
	dir_save_frame = dir_save + f'/frame'
	dir_save_head_frame = dir_save_head + f'/frame'
	os.makedirs(dir_save_frame, exist_ok=True)
	os.makedirs(dir_save_head_frame, exist_ok=True)
	for fid in frame_kps:
		list_pid = frame_kps[fid]['pid']
		list_id = frame_kps[fid]['id']

		frame = list_frames[fid]
		ts = fid_names[fid]
		frame_name = f'cam87_{ts[5:]}'


		fig, ax = plt.subplots()
		ax.imshow(frame)

		for pid, id in zip(list_pid, list_id):

			kp = list_kps[fid][id]
			ax = tester.plot_kp(kp, ax, c_list[int(pid-1)])

		fig_name = f'{frame_name}.png'
		file_save = dir_save_frame + f'/{fig_name}'
		plt.savefig(file_save)

		plt.close()
		
		# copy to head node
		file_save_head = dir_save_head_frame + f'/{fig_name}'
		copyfile(file_save, file_save_head)

		print ('save in ...', file_save_head)

# per pid
if 0:
	for pid in track_kps:

		dir_save_pid = dir_save + f'/pid_{int(pid):02d}'
		dir_save_head_pid = dir_save_head + f'/pid_{int(pid):02d}'
		os.makedirs(dir_save_pid, exist_ok=True)
		os.makedirs(dir_save_head_pid, exist_ok=True)

		list_fid = track_kps[pid]['frame']
		list_id = track_kps[pid]['id']
		for fid, id in zip(list_fid, list_id):
			ts = fid_names[fid]
			frame_name = f'cam87_{ts[5:]}'

			frame = list_frames[fid]
			kp = list_kps[fid][id]
			# print (kp.shape)
			# assert False

			fig, ax = plt.subplots()
			ax.imshow(frame)
			ax = tester.plot_kp(kp, ax, c_list[int(pid-1)])

			fig_name = f'{frame_name}.png'
			file_save = dir_save_pid + f'/{fig_name}'
			plt.savefig(file_save)

			plt.close()
			
			# copy to head node
			file_save_head = dir_save_head_pid + f'/{fig_name}'
			copyfile(file_save, file_save_head)

			print ('save in ...', file_save_head)

# collect pid kps
kps_pid = {}
for pid in track_kps:

	list_fid = track_kps[pid]['frame']
	list_id = track_kps[pid]['id']

	kps_pid[pid] = []
	for fid, id in zip(list_fid, list_id):
		kp = list_kps[fid][id]
		kps_pid[pid].append(kp)
	
	kps_pid[pid] = np.array(kps_pid[pid])


#------------------------------
# lift to 3D

import sys
sys.path.append('/home/hhyeokk/Research/CEP/repos/VideoPose3D')

coco_metadata = {
		'layout_name': 'coco',
		'num_joints': 17,
		'keypoints_symmetry': [
				# 0: nose
				[1, # left eye
				3, # left ear
				5, # left shoulder
				7,  # left elbow
				9,  # left wrist
				11, # left hip
				13, # left knee
				15], # left ankle
				[2, # right eye
				4, # right ear
				6, # right shoulder
				8, # right elbow
				10, # right wrist
				12, # right hip
				14, # right knee
				16], # right ankle
		]
}

metadata = coco_metadata
metadata['video_metadata'] = {'temp_name': {'h': h, 'w': w}}
keypoints_symmetry = metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])

from skeleton.h36m import Human36mDataset
h36m = Human36mDataset()
joints_left, joints_right = list(h36m.skeleton().joints_left()), list(h36m.skeleton().joints_right())

from common.model import TemporalModel
architecture = '3,3,3,3,3'
filter_widths = [int(x) for x in architecture.split(',')]
causal = False
dropout = 0.25
channels = 1024
dense = False
model_pos = TemporalModel(
	metadata['num_joints'], 
	2, 
	metadata['num_joints'],
	filter_widths=filter_widths, 
	causal=causal, 
	dropout=dropout, 
	channels=channels,
	dense=dense)

receptive_field = model_pos.receptive_field()
# print('INFO: Receptive field: {} frames'.format(receptive_field))
pad = (receptive_field - 1) // 2 # Padding on each side
if causal:
		# print('INFO: Using causal convolutions')
		causal_shift = pad
else:
		causal_shift = 0

if torch.cuda.is_available():
	device = torch.device('cuda')
	model_pos = model_pos.to(device)

chk_filename = '/home/hhyeokk/Research/CEP/repos/VideoPose3D/checkpoint/pretrained_h36m_detectron_coco.bin'
checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
epoch = checkpoint['epoch']
print(f'This model was trained for {epoch} epochs')
model_pos.load_state_dict(checkpoint['model_pos'])

from common.camera import normalize_screen_coordinates

for pid in kps_pid:
	kps = kps_pid[pid]
	# print (kps.shape)
	# assert False

	kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=w, h=h)

	with torch.no_grad():
		model_pos.eval()

		batch_2d = np.expand_dims(np.pad(kps[..., :2],
								((pad + causal_shift, pad - causal_shift), 
								(0, 0), (0, 0)), 'edge'), axis=0)

		batch_2d = np.concatenate((batch_2d, batch_2d), axis=0)
		batch_2d[1, :, :, 0] *= -1
		batch_2d[1, :, kps_left + kps_right] = batch_2d[1, :, kps_right + kps_left]

		inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
		if torch.cuda.is_available():
				inputs_2d = inputs_2d.to(device)
		inputs_2d = inputs_2d.contiguous()
		# print (inputs_2d.shape)
		# assert False
		predicted_3d_pos = model_pos(inputs_2d)
		# print (predicted_3d_pos.shape)

		# Undo flipping and take average with non-flipped version
		predicted_3d_pos[1, :, :, 0] *= -1
		predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
		predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)
		# print ('predicted_3d_pos:', predicted_3d_pos.size())
		prediction = predicted_3d_pos.squeeze(0).cpu().numpy()

	# print (prediction.shape)
	# assert False

	# save result
	name_save = f'pose_3d_pid_{pid}.npy'
	file_save = dir_save +f'/{name_save}'
	np.save(file_save, prediction)

	file_save_head = dir_save_head + f'/{name_save}'
	copyfile(file_save, file_save_head)

	print ('save in ...', file_save_head)

