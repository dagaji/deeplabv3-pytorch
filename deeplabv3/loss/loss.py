from torch.nn import functional as F
import torch
import pdb
from .register import register

@register.attach('relax')
def relax_loss(inputs, data):
	targets = data['label_1hot'].to(inputs.device)
	weights = data['mask'].to(inputs.device)
	batch_size = inputs.shape[0]
	nll = (-F.log_softmax(inputs, dim=1) * targets).sum(1).view(batch_size, -1).sum(1) / weights.view(batch_size, -1).sum(1)
	return nll.mean()

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


@register.attach('multilabel')
def multilabel_loss(inputs, data):

	binary_targets = data['label_1hot'].to(inputs.device)
	weights = data['mask'].to(inputs.device)

	inputs = inputs.transpose(0,1)
	binary_targets = binary_targets.transpose(0,1)

	binary_losses = []
	for _inputs, _targets in zip(inputs, binary_targets):
		batch_size = _inputs.shape[0]
		_binary_loss = F.binary_cross_entropy_with_logits(_inputs, _targets, reduction='none').view(batch_size, -1)
		_weights = weights.view(batch_size, -1)
		_binary_loss = (_binary_loss * _weights).sum(1) / _weights.sum(1)
		binary_losses.append(_binary_loss.mean())

	return sum(binary_losses) / 4

@register.attach('multilabel+refine')
def refine_multilabel_loss(inputs, data):
	total_multilabel_loss = multilabel_loss(inputs, data)
	refine_loss = refine_loss(inputs, data)
	return total_multilabel_loss + 0.4 * refine_loss

def refine_loss(inputs, data):
	refine_targets = data['refine_mask'].to(inputs.device)
	inputs = inputs.transpose(0,1)
	inputs = inputs[:2,...]
	inputs = inputs.transpose(0,1)
	return F.cross_entropy(inputs, refine_targets)


# @register.attach('relax_loss')
# def relax_loss(inputs, targets, weights):
# 	nll = (-F.log_softmax(inputs, dim=1) * targets).sum(1).view(inputs.shape[0], -1).sum(1) / normalizer
# 	return nll.mean()

# @register.attach('relax_loss+refine')
# def refine_relax_loss(inputs, targets, normalizer):
# 	nll = (-F.log_softmax(inputs, dim=1) * targets).sum(1).view(inputs.shape[0], -1).sum(1) / normalizer
# 	return nll.mean()

# def custom_loss(inputs, binary_targets, weights, refine_targets):

# 	inputs = inputs.transpose(0,1)
# 	binary_targets = binary_targets.transpose(0,1)

# 	binary_losses = []
# 	for _inputs, _targets in zip(inputs, binary_targets):
# 		batch_size = _inputs.shape[0]
# 		_binary_loss = F.binary_cross_entropy_with_logits(_inputs, _targets, reduction='none').view(batch_size, -1)
# 		_weights = weights.view(batch_size, -1)
# 		_binary_loss = (_binary_loss * _weights).sum(1) / _weights.sum(1)
# 		binary_losses.append(_binary_loss.mean())

# 	total_binary_loss = sum(binary_losses) / 4

# 	inputs_refine = inputs[:2,...].transpose(0,1)
# 	refine_loss = F.cross_entropy(inputs_refine, refine_targets)

# 	return total_binary_loss + 0.5 * refine_loss









       

