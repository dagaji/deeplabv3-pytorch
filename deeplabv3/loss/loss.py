from torch.nn import functional as F
import torch
import pdb
from .register import register
import numpy as np
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

@register.attach('cross_entropy')
def cross_entropy(inputs, data):
	device = inputs.device
	targets = data['label'].to(device)
	return F.cross_entropy(inputs, targets, ignore_index=255)

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
	iou = inputs["iou"]
	device = score.device

	iou_targets = data['iou_gt'].to(device)
	score_targets = data['lines_gt'].to(device)
	offset_targets = data['reg_gt'].to(device)
	norm_den = score_targets.sum(1)

	score_loss = F.binary_cross_entropy_with_logits(score, score_targets)

	l1_loss = F.smooth_l1_loss(offset, offset_targets, reduction='none')
	l1_loss = (l1_loss * score_targets).sum(1) / norm_den
	l1_loss = l1_loss.mean()

	iou_loss = F.binary_cross_entropy_with_logits(iou, iou_targets, reduction='none')
	iou_loss = (iou_loss * score_targets).sum(1) / norm_den
	iou_loss = iou_loss.mean()

	# print("score_loss: {}".format(score_loss.item()))
	# print("l1_loss: {}".format(l1_loss.item()))
	# print("iou_loss: {}".format(iou_loss.item()))

	return score_loss + l1_loss + iou_loss


@register.attach('line_detect_v2')
def line_detect_loss_v2(inputs, data):

	score = inputs["score"]
	device = score.device
	log_prob = F.log_softmax(score, dim=2)

	line_gt = data['lines_gt'].to(device)
	entropy_gt = data['entropy_gt'].to(device)

	max_score, _ = score.max(2)
	score_loss = F.binary_cross_entropy_with_logits(max_score, line_gt)
	entropy_loss = -1.0 * ((log_prob * entropy_gt).sum(2) * line_gt).sum(1) / line_gt.sum(1)

	return score_loss + entropy_loss

@register.attach('angle_range_loss')
def angle_range_loss(inputs, data):
	margin = 0.1
	angle_ranges = inputs["angle_ranges"]
	n_ranges = angle_ranges.shape[1]
	device = angle_ranges.device
	angle_ranges_prob = angle_ranges / angle_ranges.sum(1).unsqueeze()
	angle_range_label = data['angle_range_label'].to(device)
	margin_loss = []
	for prob, idx in zip(angle_ranges_prob, angle_range_label):
		dist = prob[idx] - prob
		dist = torch.clamp(margin - dist, min=0.0, max=1.0)
		one_hot = F.one_hot(idx, n_ranges).float()
		margin_loss.append((dist * (1 - one_hot)).sum())
	margin_loss = sum(margin_loss) / len(margin_loss)

	# cross_entropy_loss = F.nll_loss(torch.log(angle_ranges), angle_range_label)
	# # print(">> cross_entropy_loss: {}".format(0.01 * cross_entropy_loss))

	# total_loss = margin_loss + 0.0001 * cross_entropy_loss

	return margin_loss

@register.attach('angle_range_loss_v2')
def angle_range_loss_v2(inputs, data):
	margin = 0.1
	angle_ranges = inputs["angle_ranges"]
	n_ranges = angle_ranges.shape[1]
	device = angle_ranges.device
	angle_ranges_prob = angle_ranges / angle_ranges.sum(1).unsqueeze()
	angle_range_label = data['angle_range_label'].to(device)
	margin_loss = []
	for prob, idx in zip(angle_ranges_prob, angle_range_label):
		dist = (prob[idx] - prob) ** 2
		dist = torch.clamp(margin ** 2 - dist, min=0.0, max=0.1)
		one_hot = F.one_hot(idx, n_ranges).float()
		margin_loss.append((dist * (1 - one_hot)).sum())
	margin_loss = sum(margin_loss) / len(margin_loss)

	# cross_entropy_loss = F.nll_loss(torch.log(angle_ranges), angle_range_label)
	# # print(">> cross_entropy_loss: {}".format(0.01 * cross_entropy_loss))

	# total_loss = margin_loss + 0.0001 * cross_entropy_loss

	return margin_loss

@register.attach('angle_range_loss_v3')
def angle_range_loss_v3(inputs, data):
	angle_ranges = inputs["angle_ranges"]
	device = angle_ranges.device
	angle_range_label = data['angle_range_label'].to(device)
	return F.cross_entropy(angle_ranges, angle_range_label, ignore_index=255)
