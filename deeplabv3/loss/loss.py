from torch.nn import functional as F
import torch
import pdb
from .register import register
import numpy as np


@register.attach('hist_loss')
def hist_loss(inputs, data):

	seg_inputs = inputs['seg']
	seg_loss = multitask(seg_inputs, data)

	hist_inputs = inputs['hist']
	bs = hist_inputs.shape[0]
	hist_gt = data['hist_gt'].to(hist_inputs.device)
	hist_loss = []
	for _hist, idx in zip(hist_inputs, hist_gt):
		hist_loss.append(torch.clamp(0.5 - _hist[idx], min=0.0))
	hist_loss =  sum(hist_loss) / hist_inputs.shape[0]

	print("hist score: {}".format(hist_inputs.cpu().detach().numpy()[0][data['hist_gt'].numpy()[0]]))
	print("seg_loss: {}".format(seg_loss.item()))
	return seg_loss + 0.4 * hist_loss


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

	# probs_2class = F.softmax(inputs.transpose(0,1)[:2].transpose(0,1), dim=1)
	# probs_2class = torch.log(probs_2class)
	probs_2class = F.log_softmax(inputs.transpose(0,1)[:2].transpose(0,1), dim=1)
	if 'weights' in data:
		batch_size, _, H, W = inputs.shape
		weights = data['weights'].to(inputs.device)
		loss_2c = torch.gather(probs_2class, 1, targets_2classes.view(batch_size, 1, H, W)).squeeze()
		loss_2c = (loss_2c * weights).view(batch_size, -1).sum(1) / (weights.view(batch_size, -1).sum(1) + 1.0)
		loss_2c = -1.0 * loss_2c.mean()
	else:
		loss_2c = F.nll_loss(probs_2class, targets_2classes)

	#loss = F.nll_loss(probs_3class, targets_3classes) + F.nll_loss(probs_2class, targets_2classes)
	return loss_3c + loss_2c








       

