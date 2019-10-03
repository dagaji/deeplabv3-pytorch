import torch
from deeplabv3.metrics import RunningScore
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
import pdb

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from deeplabv3.dataset import get_dataset
from deeplabv3.model import get_model
from deeplabv3.optimizer import get_optimizer
from deeplabv3.scheduler import get_scheduler
from deeplabv3.loss import get_loss
import deeplabv3.utils as utils
from deeplabv3.metrics import RunningScore, AverageMeter
import pdb
import time
from tqdm import tqdm
from skimage.io import imsave, imread
import os.path
import os
from argparse import ArgumentParser
from torchsummary import summary
from deeplabv3.save import ResultsSaverFactory, CheckpointSaver, VideoSaver
from pathlib import Path
import time
import collections
import cv2
from deeplabv3.lines import points2line_eq, create_grid_intersect, find_intesect_borders, normal_form, create_grid, general_form
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift


class MultiFrameMerge():

	def __init__(self,):
		self.buffer = collections.deque(maxlen=3)
		self.sift = cv2.xfeatures2d.SIFT_create()
		self.bf = cv2.BFMatcher()

	def homography(self, sift_src, sift_dst):

		kp_a, des_a = sift_src
		kp_b, des_b = sift_dst

		matches = self.bf.knnMatch(des_a, trainDescriptors=des_b, k=2)

		# Lowes Ratio
		good_matches = []
		for m, n in matches:
			if m.distance < .75 * n.distance:
				good_matches.append(m)

		src_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
		dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

		if len(src_pts) > 4:
			#M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5)
			M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)
		else:
			M = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

		return M

	def transform_point(self, point, M):

		p = np.array(point + (1,))
		p_prime = np.dot(M, p)

		y_t = p_prime[1] / p_prime[2]
		x_t = p_prime[0] / p_prime[2]

		return x_t, y_t

	def transform_line(self, line_eq, M):

		p = np.array(line_eq)
		p_prime = np.dot(M, p)
		line_eq_t = tuple(p_prime.tolist())
		return line_eq_t

	def angle_diff(self, anchor_angle, angles):
		unit_vertor_anchor = np.array([np.cos(anchor_angle), np.sin(anchor_angle)]).reshape(-1,2)
		unit_vertors = np.hstack([np.cos(angles).reshape(-1,1), np.sin(angles).reshape(-1,1)])
		dot_product = np.dot(unit_vertor_anchor, unit_vertors.T)
		dot_product = np.clip(dot_product, a_max=1.0, a_min=-1.0)
		diff_angles = np.abs(np.arccos(dot_product)).squeeze()
		diff_angles = np.minimum(diff_angles, np.pi - diff_angles)
		return diff_angles

	def angle_mean(self, angles):
		return np.arctan2(np.sum(np.sin(angles)), np.sum(np.cos(angles)))

	def cluster_lines(self, _lines):

		if np.size(_lines) == 0:
			return []

		ms = MeanShift(bandwidth=50.0)
		pred = ms.fit_predict(_lines[:,0].reshape(-1,1))
		cluster_labels = np.unique(pred)

		cluster_lines = []
		for _label in cluster_labels.tolist():
			res = _lines[pred == _label]
			res = (res[:,0].mean(), self.angle_mean(res[:,1]))
			cluster_lines.append(res)

		return cluster_lines


	def filter_lines(self, lines1, lines2):
			

		if len(lines2) == 0:
			print("lines2 empty")
			return lines1
		else:

			lines2 = np.array(lines2)
			ori_lines2 = lines2[:,1]
			anchor_ori = ori_lines2[0]
			angle_diff = np.rad2deg(self.angle_diff(anchor_ori, ori_lines2))
			ori1 =  self.angle_mean(ori_lines2[angle_diff > 20])
			ori2 =  self.angle_mean(ori_lines2[angle_diff < 20])

			lines1 = np.array(lines1)
			angles = lines1[:,1]
			angles_diff_ori1 =  np.rad2deg(self.angle_diff(ori1, angles))
			angles_diff_ori2 =  np.rad2deg(self.angle_diff(ori2, angles))
			is_not_outlier = np.logical_or(angles_diff_ori1 < 3.0, angles_diff_ori2 < 3.0)
			lines1 = lines1[is_not_outlier]

			for _line in lines2:
				rhos = lines1[:,0]
				angles = lines1[:,1]
				rhos_diff = np.abs(_line[0] - rhos)
				angles_diff = np.rad2deg(self.angle_diff(_line[1], angles))
				keep = np.logical_and(rhos_diff < 75.0, angles_diff < 5.0)
				lines1 = lines1[np.invert(keep)]

			lines1 = lines1.tolist()

			if len(lines1) == 0:
				print("lines1 empty")
				return lines2

			lines1 = np.array(lines1)
			ori_lines1 = lines1[:,1]

			angle_diff_ori1 = np.rad2deg(self.angle_diff(ori1, ori_lines1))
			angle_diff_ori1 = np.atleast_1d(angle_diff_ori1)
			lines1_1 = lines1[angle_diff_ori1 < 5]

			angle_diff_ori2 = np.rad2deg(self.angle_diff(ori2, ori_lines1))
			angle_diff_ori2 = np.atleast_1d(angle_diff_ori2)
			lines1_2 = lines1[angle_diff_ori2 < 5]

			cluster_lines1_1 = self.cluster_lines(lines1_1)
			cluster_lines1_2 = self.cluster_lines(lines1_2)

			return cluster_lines1_1 + cluster_lines1_2 + lines2.tolist()


	def merge_lines(self,):

		M_01 = self.homography(self.buffer[0][2], self.buffer[1][2])
		M_21 = self.homography(self.buffer[2][2], self.buffer[1][2])

		sz = self.buffer[0][0].shape

		lines1 = []
		for frame_lines, M in zip([self.buffer[0][1], self.buffer[2][1]], [M_01, M_21]):
			for _frame_line in frame_lines:

				line_points = find_intesect_borders(_frame_line, sz)
				line_point_0 = self.transform_point(line_points[0], M)
				line_point_1 = self.transform_point(line_points[1], M)
				line_eq = points2line_eq(line_point_0, line_point_1)

				if find_intesect_borders(line_eq, sz) is not None:
					lines1.append(normal_form(*line_eq))

		lines2  = [normal_form(*line_eq) for line_eq in self.buffer[1][1]]
		return create_grid(sz, self.filter_lines(lines1, lines2))


	def __call__(self, frame, pred_S, detected_lines):

		frame_bgr = frame.copy()
		frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)
		sift_out = self.sift.detectAndCompute(frame, None)
		self.buffer.append((pred_S, detected_lines, sift_out, frame_bgr))

		if len(self.buffer) == 3:
			grid = self.merge_lines()
			out = self.buffer[1][0].copy()
			mask = self.buffer[1][0] == 0
			out[mask] = grid[mask]
			return out, self.buffer[1][-1]
		else:
			return None



def get_last_checkpoint(checkpoint_dir):

	if not os.path.exists(checkpoint_dir):
		return None
	else:
		checkpoints_globs = list(Path(checkpoint_dir).glob('*.pth'))
		if len(checkpoints_globs) == 0:
			return None
		key = lambda x: int(os.path.basename(str(x)).split('.')[0].split('_')[-1])
		last_checkpoint_path = str(sorted(checkpoints_globs, key=key, )[-1])
		return last_checkpoint_path


def get_dataloader(dataset_cfg):
	dataset_params = dict(dataset_cfg)
	dataset = get_dataset('video_dataset')(**dataset_params)
	return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


def parse_args():
	parser = ArgumentParser()
	parser.add_argument('--config', type=str, required=True)
	parser.add_argument('--videos', nargs='+', type=str)
	parser.add_argument('--expers', nargs='+', type=str)
	parser.add_argument('-cpu', dest='use_cpu', action='store_true')
	parser.set_defaults(use_cpu=False)
	return parser.parse_args()


def validate(val_model, val_loader, num_classes, device, saver=None):

	val_model.eval()   # Set model to evaluate mode
	merge_frame = MultiFrameMerge()

	with torch.set_grad_enabled(False):

		# Iterate over data.
		for _iter, (image, frame) in tqdm(enumerate(val_loader), total=len(val_loader), dynamic_ncols=True):

			inputs = image.to(device)
			#preds = val_model(inputs)
			pred_S, detected_lines = val_model(inputs)
			results = merge_frame(frame.numpy().squeeze(), pred_S, detected_lines)
			if results is not None:
				out, frame = results
				saver.save_frame(frame.astype(np.uint8), out)
				

	# if saver is not None:
	# 	saver.save_video()


if __name__ == "__main__":

	args = parse_args()

	exper_name = os.path.basename(args.config).split(".")[0]

	num_classes, video_cfg = utils.get_cfgs_video(args.config)
	device = torch.device("cuda:0" if torch.cuda.is_available() and not args.use_cpu else "cpu")

	_save_dir = os.path.join('videos', exper_name, '{}')
	
	for video_info in video_cfg["videos-info"]:

		if args.videos is not None:
			if video_info["name"] not in args.videos:
				continue

		val_dataloader = get_dataloader(video_info['params'])

		checkpoint_dir = os.path.join('checkpoint', video_info['checkpoint'], exper_name)
		last_checkpoint_path = get_last_checkpoint(checkpoint_dir)
		if last_checkpoint_path is not None:
			print("CHECKPOINT")
			last_checkpoint = torch.load(last_checkpoint_path)

		for video_exper in video_cfg["expers"]:

			if args.expers is not None:
				if video_exper["name"] not in args.expers:
					continue

			save_dir = _save_dir.format(video_exper["name"])
			if not os.path.exists(save_dir):
				os.makedirs(save_dir)
			save_path = os.path.join(save_dir, video_info["name"] + '.MOV')

			model_val = get_model(num_classes, video_exper["model"]).to(device)
			model_val.load_state_dict(last_checkpoint["model_state_dict"], strict=False)

			results_saver = VideoSaver(num_classes, save_path)

			validate(model_val, 
				val_dataloader, 
				num_classes, 
				device, saver=results_saver)


