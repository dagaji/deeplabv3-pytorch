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

def getLargestCC(labels, segmentation):
    return np.argmax(np.bincount(labels.flat, weights=segmentation.flat))

def get_sortedCC(labels, segmentation):
	cc_count = np.bincount(labels.flat, weights=segmentation.flat)
	return np.argsort(cc_count)[::-1], cc_count


def argmax_predict(x):
	_, pred = torch.max(x, 1)
	return pred

def erode_predict(x, kernel=np.ones((3,3), np.uint8), iterations=8):
	pred = argmax_predict(x)
	pred = np.squeeze(pred.cpu().numpy())
	pred_1 = (pred == 1).astype(np.uint8)
	pred_1 = cv2.erode(pred_1, kernel, iterations=iterations)
	pred[pred == 1] = 0
	pred[pred_1 > 0] = 1
	pred = pred[np.newaxis,...]
	return torch.cuda.LongTensor(pred)

def multilabel_predict(x):
	pass

def argmax_cc_predict(x):
	pred = np.squeeze(argmax_predict(x).cpu().numpy())
	pred_1 = (pred == 1).astype(np.uint8)
	if np.any(pred_1):
		pred_1_cc = label(pred_1)
		largets_cc = (pred_1_cc == getLargestCC(pred_1_cc, pred_1))
		pred[pred == 1] = 0
		pred[largets_cc] = 1
	pred = pred[np.newaxis,...]
	return torch.cuda.LongTensor(pred)

# def multiltask_predict(x):

# 	probs_4class = F.softmax(x, dim=1)
# 	probs_4class = probs_4class.transpose(0,1)
# 	probs_3class = torch.stack([(probs_4class[0] + probs_4class[1]), 
# 							 probs_4class[2], 
# 							 probs_4class[3]]).transpose(0,1)
# 	_, pred_3c = torch.max(probs_3class, 1)
# 	pred_3c = np.squeeze(pred_3c.cpu().numpy())

# 	probs_2class = F.softmax(x.transpose(0,1)[:2].transpose(0,1), dim=1)
# 	_, pred_2c = torch.max(probs_3class, 1)
# 	pred_2c = np.squeeze(pred_2c.cpu().numpy())

# 	pred_4c = pred_3c.copy()
# 	pred_4c[pred_3c == 0] = pred_2c[pred_3c == 0]

def line_detection(x, kernel=np.ones((3,3), np.uint8), iterations=5):

	angle_range_coarse = (45.0, 135.0)
	APR_prob = F.softmax(x, dim=1).cpu().numpy().squeeze()[:2].sum(0)
	plt.figure()
	plt.imshow(APR_prob)
	pred = np.squeeze(argmax_predict(x).cpu().numpy())
	pred_1 = (pred == 1).astype(np.uint8)
	# plt.figure()
	# plt.imshow(pred_1)
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
				# print("cc1_size:{}, cc2_size:{}".format(cc1_size, cc2_size))
				# pdb.set_trace()
				pred_1 = np.logical_or(pred_1_cc == CC_indices[0], pred_1_cc == CC_indices[1]).astype(np.uint8)
			else:
				pred_1 = (pred_1_cc == CC_indices[0]).astype(np.uint8)
			# pred_1 = (pred_1_cc == getLargestCC(pred_1_cc, pred_1)).astype(np.uint8)
			pred_1 = cv2.erode(pred_1, kernel, iterations=iterations)

			angles_coarse = np.linspace(np.deg2rad(45), np.deg2rad(135), 1000)
			hspace, angles, distances = hough_line(pred_1, angles_coarse)
			tresh = np.max(hspace) * 0.1
			hspace[hspace < tresh] = 0

			bin_length = 50
			resol = 90 / 1000.0
			hist, bins_edges = lines.compute_hist(hspace, angles, bin_length=bin_length)
			max_angle = (2 * bins_edges[np.argmax(hist)] + bin_length * resol) / 2
			angles_fine_h = (max_angle - 2.5 * bin_length * resol, max_angle + 2.5 * bin_length * resol)
			angles_fine_v = (angles_fine_h[0] + 90, angles_fine_h[1] + 90)
			# plt.figure()
			# plt.imshow(pred_1)
			# plt.figure()
			# plt.plot(hist)
			# plt.show()
			# print("angles_fine_h {}".format(angles_fine_h))

			# angles_fine_h = (85.54, 103.54)
			# angles_fine_v = (angles_fine_h[0] + 90, angles_fine_h[1] + 90)


			_, angles_h, dist_h = lines.search_lines(pred_1, angles_fine_h, npoints=1000, min_distance=100, min_angle=300, threshold=None)
			_, angles_v, dist_v = lines.search_lines(pred_1, angles_fine_v, npoints=1000, min_distance=100, min_angle=300, threshold=None)

			lines_h = lines.get_lines(dist_h, angles_h)
			lines_v = lines.get_lines(dist_v, angles_v)

			grid = lines.create_grid(pred.shape, lines_h + lines_v, width=16) * APR_pred
			pred[APR_pred] = grid[APR_pred]

	pred = pred[np.newaxis,...]
	return torch.cuda.LongTensor(pred)



















