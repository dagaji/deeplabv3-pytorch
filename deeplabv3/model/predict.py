import torch
from torch.nn import functional as F
import cv2
import numpy as np
import pdb
from skimage.measure import label
import deeplabv3.lines as lines
import pdb
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from skimage.transform import hough_line
from scipy.signal import find_peaks
from deeplabv3.lines import general_form, find_intesect_borders, normal_form

def getLargestCC(labels, segmentation):
    return np.argmax(np.bincount(labels.flat, weights=segmentation.flat))

def get_sortedCC(labels, segmentation):
	cc_count = np.bincount(labels.flat, weights=segmentation.flat)
	return np.argsort(cc_count)[::-1], cc_count

def argmax_predict(x):
	_, pred = torch.max(x, 1)
	return pred

# def line_detect_cnn(inputs, data):

# 	def norm_coords(coords, sz):
#         coords[:, 0] = 2 * coords[:,0] / float(sz[1] - 1) - 1
#         coords[:, 1] = 2 * coords[:,1] / float(sz[0] - 1) - 1
#         coords = coords.astype(np.float32)
#         return coords

#     def line_coords(intersect_points, orientation_rad, n_points, plot=False):

#         if plot:
#             fig, ax = plt.subplots(1)
#             ax.imshow(np.zeros(self.bbox_size[::-1] + (3,), dtype=np.uint8))

#         step_len = distance.euclidean(intersect_points[0], intersect_points[1]) / n_points
#         unit_vector = np.array((np.sin(orientation_rad), np.cos(orientation_rad)))
#         line_points = []
#         for i in np.arange(1, n_points):
#             line_point = np.array(intersect_points[0]) + i * step_len * unit_vector
#             line_points.append(line_point)
#             if plot:
#                 circle = plt.Circle(tuple(line_point.tolist()), 2, color='b')
#                 ax.add_patch(circle)

#         if plot:
#             plt.show()

#         return np.array(line_points)

#     pred = inputs['pred']
#     features = inputs['features']
#     device = pred.device

#     prob = torch.softmax(pred, dim=1).transpose(0,1)[1].unsqueeze(1)

# 	sz = pred.shape[-2:]
# 	angle_step = 1.0
# 	rho_step = 10.0
# 	max_distance = 2 * np.sqrt(sz[0] ** 2 + sz[1] ** 2)
# 	rhos =  np.arange(-max_distance / 2.0, max_distance / 2.0 + rho_step, rho_step)

# 	angle_ranges = data['angle_ranges']

# 	detected_lines = []
# 	for _features, _prob, _angle_range in zip(features, prob, angle_ranges):
# 		_angle_range = np.rad2deg(_angle_range.cpu().numpy())
# 		thetas = np.deg2rad(np.arange(_angle_range[0], _angle_range[1] + angle_step, angle_step))
# 		for _theta in thetas:
# 			for _rho in rhos:
# 				line_coeffs = general_form(_rho, _theta)
# 				intersect_points = find_intesect_borders(line_coeffs)
# 				if intesect_borders is not None:
# 					line_points = line_coords(intersect_points, _theta, 100)
# 					line_points = norm_coords(line_points, sz)
# 					line_points = line_points.reshape(1, line_points.shape[0],1,2)
# 					grid = torch.Tensor(line_points)
# 					sampled_features = torch.squeeze(F.grid_sample(_features, grid), dim=3)
# 					sampled_prob = torch.squeeze(F.grid_sample(_prob, grid), dim=3)
# 					sampled_features = sampled_features * sampled_prob / sampled_prob.sum()
					

def draw_lines(line_probs, data, line_coeffs):
	# line_coeffs = data['line_coeffs'].cpu().numpy().squeeze()
	sz = data['image'].shape[-2:]
	line_probs = line_probs.cpu().detach().numpy()
	selected_lines = line_coeffs[line_probs > 0.5]
	return lines.create_grid(sz, selected_lines.tolist())
	# plt.figure()
	# plt.imshow(mask)
	# plt.show()

def draw_lines2(line_intersect, scores, offset, data):

	probs = torch.sigmoid(scores).cpu().numpy().squeeze()
	offset = offset.cpu().numpy().squeeze()
	valid_line = (probs > 0.9)
	sz = data['image'].shape[-2:]
	line_intersect = line_intersect[valid_line] + offset[valid_line] * 417
	line_intersect1 = line_intersect[:,:2]
	line_intersect2 = line_intersect[:,2:]
	line_intersect = [[tuple(pt1), tuple(pt2)] for pt1, pt2 in zip(line_intersect1, line_intersect2)]
	return lines.create_grid_intersect(line_intersect, sz)


def _line_detection(x, kernel=np.ones((3,3), np.uint8), iterations=5):

	pred = np.squeeze(argmax_predict(x).cpu().numpy()).astype(np.uint8)
	pred_1 = (pred == 1).astype(np.uint8)
	APR_pred = np.logical_or(pred == 0, pred == 1)
	pred_S = pred.copy()
	pred_S[APR_pred] = 0

	if np.any(APR_pred):

		APR_pred_cc = label(APR_pred.astype(np.uint8))
		APR_pred = (APR_pred_cc == getLargestCC(APR_pred_cc, APR_pred.astype(np.uint8)))

		if np.sum(APR_pred) > 0.3 * np.prod(APR_pred.shape):

			pred_1 *= APR_pred
			pred_1_cc = label(pred_1)
			CC_indices, cc_count = get_sortedCC(pred_1_cc, pred_1)
			cc1_size = cc_count[CC_indices[0]]
			cc2_size = cc_count[CC_indices[1]]
			if (float(cc2_size) / cc1_size) > 0.1:
				pred_1 = np.logical_or(pred_1_cc == CC_indices[0], pred_1_cc == CC_indices[1]).astype(np.uint8)
			else:
				pred_1 = (pred_1_cc == CC_indices[0]).astype(np.uint8)
	
			pred_1 = cv2.erode(pred_1, kernel, iterations=iterations)

			angles_coarse = np.linspace(np.deg2rad(-45), np.deg2rad(135), 180)
			hspace, angles, distances = hough_line(pred_1, angles_coarse)
			tresh = np.max(hspace) * 0.1
			hspace[hspace < tresh] = 0

			bin_length = 5
			resol = 1.0
			hist, bins_edges = lines.compute_hist(hspace, angles, bin_length=bin_length)
			hist_peaks, prop = find_peaks(hist, height=0.1 * max(hist))
			indices = np.argsort(prop['peak_heights'])[::-1]
			hist_peaks = hist_peaks[indices] * bin_length * resol - 45.0

			main_peaks = None
			for peak in hist_peaks.tolist():
				dist = np.abs(hist_peaks - peak)
				pair_peaks_loc = np.logical_and((90 - bin_length * resol) <= dist,
				                                 dist <= (90 + bin_length * resol))
				if np.any(pair_peaks_loc):
					pair_peak = hist_peaks[pair_peaks_loc][0]
					main_peaks = (peak, pair_peak)
					break
			
			detected_lines = []
			if main_peaks is not None:

				for peak in main_peaks:
					center_angle = (2 * peak + bin_length * resol) / 2
					angles_range = (center_angle - 2 * bin_length * resol, center_angle + 2 * bin_length * resol)
					_, line_angles, line_dist = lines.search_lines(pred_1, angles_range, npoints=1000, min_distance=100, min_angle=300, threshold=None)
					detected_lines += lines.get_lines(line_dist, line_angles)

				# detected_lines = [general_form(*line_coeff) for line_coeff in detected_lines]

	return pred_S, detected_lines

def line_detection(x):
	pred_S, detected_lines = _line_detection(x)
	grid = lines.create_grid(pred_S.shape, detected_lines, width=10)
	pred = pred_S.copy()
	mask = (pred_S == 0)
	pred[mask] = grid[mask]
	return pred























