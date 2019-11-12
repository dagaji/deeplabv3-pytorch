from torch.nn import functional as F
import torch
import pdb
from .register import register
import numpy as np
from deeplabv3.model.gabor import GaborConv2d
import matplotlib.pyplot as plt


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
		return torch.clamp(delta - tol ** 2, min=0, max=3 * (tol ** 2)).mean()

	def _cross_entropy():
		_inputs = inputs['seg']
		targets = data['frame_label'].to(_inputs.device)
		return F.cross_entropy(_inputs, targets, ignore_index=255)

	return _cross_entropy() + 0.4 * _offset_loss()


@register.attach('hist_loss_v2')
def hist_loss(inputs, data):

	pred = inputs['out']
	hist = inputs['hist']
	bs = pred.shape[0]
	device = pred.device
	seg_label = data['seg_label'].to(device)
	angle_gt = data['angle_gt'].to(device)

	hist_loss = []
	for _angle_gt, _hist in zip(angle_gt, hist):
		if bool(_angle_gt.sum() > 0):
			_hist_loss = (_hist * _angle_gt).sum()
			hist_loss.append(-1.0 * torch.log(_hist_loss))

	hist_loss = sum(hist_loss) / max(len(hist_loss), 1)
	seg_loss = F.cross_entropy(pred, seg_label, ignore_index=255)

	print("hist_loss: {}".format(hist_loss))
	print("seg_loss: {}".format(seg_loss))

	return seg_loss + 0.1 * hist_loss


@register.attach('dice_loss')
def dice_loss(inputs, data):

	pred = inputs['out']
	pred_coop = inputs['out_coop']
	device = pred.device
	label = data['label'].to(device)

	label_s = label.clone()
	label_s[label != 255] = torch.clamp(label_s[label != 255] - 1, min=0)

	label_dice = (label == 1).float()
	mask = (label == 0).float() + (label == 1).float()

	def _dice_loss(pred, target, mask):
		numerator = 2 * torch.sum(pred * target * mask)
		denominator = torch.sum((pred + target) * mask)
		return 1.0 - (numerator + 1.0) / (denominator + 1.0)

	def _cross_entropy(pred, target):

		probs_4class = F.softmax(pred, dim=1).transpose(0,1)
		probs_3class = torch.stack([(probs_4class[0] + probs_4class[1]), 
							 probs_4class[2], 
							 probs_4class[3]]).transpose(0,1)
		logprobs_3class = torch.log(probs_3class)
		return F.nll_loss(logprobs_3class, targets_3classes)

	return _dice_loss(pred_coop, label_dice, mask) + _cross_entropy(pred, label_s)


@register.attach('line_detect')
def line_detect_loss(inputs, data):

	score = inputs["score"]
	offset = inputs["offset"]
	x_seg = inputs["seg"]
	device = x_seg.device

	seg_targets = data['seg_label'].to(device)
	score_targets = data['lines_gt'].to(device)
	offset_targets = data['reg_gt'].to(device)

	seg_loss = F.cross_entropy(x_seg, seg_targets, ignore_index=255)
	score_loss = F.binary_cross_entropy_with_logits(score, score_targets)
	l1_loss = F.smooth_l1_loss(offset, offset_targets, reduction='none').sum(2) * score_targets / score_targets.sum(1).unsqueeze(1)
	l1_loss = l1_loss.mean()
	
	return seg_loss + score_loss + l1_loss


@register.attach('hist_loss')
class HistLoss:

	def __init__(self, angle_step, max_angle=45, min_angle=-45, hist_weight=0.4, plot_filters=True):

		angles = np.deg2rad(np.arange(min_angle, max_angle, angle_step))
		self.num_angles = len(angles)
		self.filter_bank = []
		for angle in angles:
			self.filter_bank.append(GaborConv2d(angle))
			if plot_filters:
				self.filter_bank[-1].plot_filter()
		if plot_filters:
			plt.show()

		self.hist_weight = hist_weight

	def __call__(self, inputs, data):

		pred = inputs['out']
		bs = pred.shape[0]
		device = pred.device
		seg_label = data['seg_label'].to(device)
		hist_mask = data['hist_mask'].to(device).unsqueeze(1)
		pred_1 = pred.transpose(0,1)[1].unsqueeze(1) * hist_mask
		angle_gt = data['angle_gt'].to(device)

		res = []
		for gabor_filter in self.filter_bank:
			res.append(gabor_filter(pred_1))

		res = F.relu(torch.cat(res, dim=1))
		hist = res.view(bs, self.num_angles, -1).sum(2)
		hist /= (hist.sum(1).unsqueeze(1).repeat([1, self.num_angles]) + 1.0)

		hist_loss = []
		for _angle_gt, _hist, _res in zip(angle_gt, hist, res):
			if bool(_angle_gt.sum() > 0):
				# print(np.where(_angle_gt.cpu().numpy()))
				# print((_hist * _angle_gt).sum())
				# for i in np.arange(len(_angle_gt)):
				# 	plt.figure()
				# 	plt.imshow(_res[i].cpu().detach().numpy().squeeze())
				# plt.show()
				_hist_loss = (_hist * _angle_gt).sum()
				hist_loss.append(-1.0 * torch.log(_hist_loss))

		hist_loss = sum(hist_loss) / max(len(hist_loss), 1)
		seg_loss = F.cross_entropy(pred, seg_label, ignore_index=255)

		print("hist_loss: {}".format(hist_loss))
		print("seg_loss: {}".format(seg_loss))

		if hist_loss > 10.0:
			pdb.set_trace()

		# print("hist_loss: {}".format(hist_loss.item()))

		return seg_loss + self.hist_weight * hist_loss




















       

