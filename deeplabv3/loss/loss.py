from torch.nn import functional as F
import torch
import pdb
from .register import register
import numpy as np
from deeplabv3.model.gabor import GaborConv2d


@register.attach('multitask')
def multitask(inputs, data):

	targets_3classes = data['label_3c'].to(inputs.device)
	targets_2classes = data['label_2c'].to(inputs.device)

	probs_4class = F.softmax(inputs, dim=1)
	probs_4class = probs_4class.transpose(0,1)
	probs_3class = torch.stack([(probs_4class[0] + probs_4class[1]), 
							 probs_4class[2], 
							 probs_4class[3]]).transpose(0,1)
	probs_3class = torch.log(probs_3class)
	loss_3c = F.nll_loss(probs_3class, targets_3classes)

	probs_2class = F.log_softmax(inputs.transpose(0,1)[:2].transpose(0,1), dim=1)
	if 'weights' in data:
		batch_size, _, H, W = inputs.shape
		weights = data['weights'].to(inputs.device)
		loss_2c = torch.gather(probs_2class, 1, targets_2classes.view(batch_size, 1, H, W)).squeeze()
		loss_2c = (loss_2c * weights).view(batch_size, -1).sum(1) / (weights.view(batch_size, -1).sum(1) + 1.0)
		loss_2c = -1.0 * loss_2c.mean()
	else:
		loss_2c = F.nll_loss(probs_2class, targets_2classes)

	return loss_3c + loss_2c

@register.attach('mosaic_loss')
def mosaic_loss(inputs, data):

	def _offset_loss(tol=0.3):
		offset = inputs['offset']
		batch_size, _, H, W = offset.shape
		offset = offset.view(batch_size, 2, H * W).transpose(0,1)
		delta = offset[0] ** 2 +  offset[1] ** 2
		return torch.clamp(delta - tol ** 2, min=0, max=tol ** 2).mean()

	def _cross_entropy():
		_inputs = inputs['seg']
		targets = data['frame_label'].to(_inputs.device)
		return F.cross_entropy(_inputs, targets, ignore_index=255)

	# return _cross_entropy() + 0.1 * _offset_loss()
	return _cross_entropy()


@register.attach('hist_loss')
class HistLoss:

	def __init__(self, angle_step, max_angle=45, min_angle=-45, hist_weight=0.4):

		angles = np.deg2rad(np.arange(min_angle, max_angle, angle_step))
		self.num_angles = len(angles)
		self.filter_bank1 = []
		self.filter_bank2 = []
		for angle in angles:
			self.filter_bank1.append(GaborConv2d(angles))
			self.filter_bank2.append(GaborConv2d(angles + np.pi/2))

		self.hist_weight = hist_weight

	def __call__(self, inputs, data):

		pred = inputs['out']
		bs = inputs.shape[0]
		device = pred.device
		seg_label = data['seg_label'].to(device)
		hist_label1 = data['hist_label1'].to(device)
		hist_label2 = data['hist_label2'].to(device)
		angle_indices = data['angle_idx']

		prob = torch.softmax(pred, dim=1).transpose(0,1)[1]

		res1 = []
		res2 = []
		for gabor_filter1, gabor_filter2 in zip(self.filter_bank1, self.filter_bank2):

			res1.append(gabor_filter1(prob) * hist_label1)
			res2.append(gabor_filter2(prob) * hist_label2)

		res1 = torch.cat(res1, dim=1).view(bs, self.num_angles, -1).sum(2)
		res2 = torch.cat(res1, dim=1).view(bs, self.num_angles, -1).sum(2)
		hist = res1 + res2
		hist /= (hist.sum(1).repeat([1, self.num_angles]) + 1.0)

		hist_loss = []
		for i in np.arange(bs):
			idx = angle_indices[i]
			if idx >= 0:
				hist_loss.append(hist[i, idx])

		return F.cross_entropy(pred, seg_label, ignore_index=255) + self.hist_weight * sum(hist_loss)




















       

