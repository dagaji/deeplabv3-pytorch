from torch.nn import functional as F
import torch
import pdb
from .register import register
import numpy as np


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











       

