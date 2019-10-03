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

def line_detection(x, kernel=np.ones((3,3), np.uint8), iterations=5):

	APR_prob = F.softmax(x, dim=1).cpu().numpy().squeeze()[:2].sum(0)
	pred = np.squeeze(argmax_predict(x).cpu().numpy())
	pred_1 = (pred == 1).astype(np.uint8)
	APR_pred = np.logical_or(pred == 0, pred == 1)

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

			if main_peaks is not None:

				detected_lines = []
				for peak in main_peaks:
					center_angle = (2 * peak + bin_length * resol) / 2
					angles_range = (center_angle - 2 * bin_length * resol, center_angle + 2 * bin_length * resol)
					_, line_angles, line_dist = lines.search_lines(pred_1, angles_range, npoints=1000, min_distance=100, min_angle=300, threshold=None)
					detected_lines += lines.get_lines(line_dist, line_angles)

				grid = lines.create_grid(pred.shape, detected_lines, width=16) * APR_pred
				pred[APR_pred] = grid[APR_pred]

	pred = pred[np.newaxis,...]
	return torch.cuda.LongTensor(pred)


# def line_detection_multi(x, kernel=np.ones((3,3), np.uint8), iterations=5):

# 	pred = np.squeeze(argmax_predict(x).cpu().numpy()).astype(np.uint8)
# 	pred_1 = (pred == 1).astype(np.uint8)
# 	APR_pred = np.logical_or(pred == 0, pred == 1)
# 	pred_S = pred.copy()
# 	pred_S[np.logical_or(pred == 0, pred == 1)] = 0

# 	if np.any(APR_pred):

# 		APR_pred_cc = label(APR_pred.astype(np.uint8))
# 		APR_pred = (APR_pred_cc == getLargestCC(APR_pred_cc, APR_pred.astype(np.uint8)))

# 		if np.sum(APR_pred) > 0.3 * np.prod(APR_pred.shape):

# 			pred_1 *= APR_pred
# 			pred_1_cc = label(pred_1)
# 			CC_indices, cc_count = get_sortedCC(pred_1_cc, pred_1)
# 			cc1_size = cc_count[CC_indices[0]]
# 			cc2_size = cc_count[CC_indices[1]]
# 			if (float(cc2_size) / cc1_size) > 0.1:
# 				pred_1 = np.logical_or(pred_1_cc == CC_indices[0], pred_1_cc == CC_indices[1]).astype(np.uint8)
# 			else:
# 				pred_1 = (pred_1_cc == CC_indices[0]).astype(np.uint8)
	
# 			pred_1 = cv2.erode(pred_1, kernel, iterations=iterations)

# 			angles_coarse = np.linspace(np.deg2rad(-45), np.deg2rad(135), 180)
# 			hspace, angles, distances = hough_line(pred_1, angles_coarse)
# 			tresh = np.max(hspace) * 0.1
# 			hspace[hspace < tresh] = 0

# 			bin_length = 5
# 			resol = 1.0
# 			hist, bins_edges = lines.compute_hist(hspace, angles, bin_length=bin_length)
# 			hist_peaks, prop = find_peaks(hist, height=0.1 * max(hist))
# 			indices = np.argsort(prop['peak_heights'])[::-1]
# 			hist_peaks = hist_peaks[indices] * bin_length * resol - 45.0

# 			main_peaks = None
# 			for peak in hist_peaks.tolist():
# 				dist = np.abs(hist_peaks - peak)
# 				pair_peaks_loc = np.logical_and((90 - bin_length * resol) <= dist,
# 				                                 dist <= (90 + bin_length * resol))
# 				if np.any(pair_peaks_loc):
# 					pair_peak = hist_peaks[pair_peaks_loc][0]
# 					main_peaks = (peak, pair_peak)
# 					break
			
# 			detected_lines = []
# 			if main_peaks is not None:

# 				for peak in main_peaks:
# 					center_angle = (2 * peak + bin_length * resol) / 2
# 					angles_range = (center_angle - 2 * bin_length * resol, center_angle + 2 * bin_length * resol)
# 					_, line_angles, line_dist = lines.search_lines(pred_1, angles_range, npoints=1000, min_distance=100, min_angle=300, threshold=None)
# 					detected_lines += lines.get_lines(line_dist, line_angles)

# 				detected_lines = [find_intesect_borders(general_form(*line_coeff), pred.shape) for line_coeff in detected_lines]

# 	return pred_S, detected_lines


def line_detection_multi(x, kernel=np.ones((3,3), np.uint8), iterations=5):

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

				# print(detected_lines[-1])
				# aux = general_form(*detected_lines[-1])
				# print(aux)
				# normal_form(*aux)
				detected_lines = [general_form(*line_coeff) for line_coeff in detected_lines]

	return pred_S, detected_lines



















