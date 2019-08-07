import torch
from torch.nn import functional as F
import cv2
import numpy as np
import pdb
from skimage.measure import label

def getLargestCC(labels, segmentation):
    return np.argmax(np.bincount(labels.flat, weights=segmentation.flat))

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

def multiltask_predict(x):

	probs_4class = F.softmax(x, dim=1)
	probs_4class = probs_4class.transpose(0,1)
	probs_3class = torch.stack([(probs_4class[0] + probs_4class[1]), 
							 probs_4class[2], 
							 probs_4class[3]]).transpose(0,1)
	_, pred_3c = torch.max(probs_3class, 1)
	pred_3c = np.squeeze(pred_3c.cpu().numpy())

	probs_2class = F.softmax(x.transpose(0,1)[:2].transpose(0,1), dim=1)
	_, pred_2c = torch.max(probs_3class, 1)
	pred_2c = np.squeeze(pred_2c.cpu().numpy())

	pred_4c = pred_3c.copy()
	pred_4c[pred_3c == 0] = pred_2c[pred_3c == 0]



