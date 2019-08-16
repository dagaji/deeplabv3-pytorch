from torch.nn import functional as F
import torch
import pdb
from .register import register

def cross_entropy(score, target):
	device = score.device
	target = target.to(device)
	log_prob = F.log_softmax(score, dim=1)
	loss = F.nll_loss(log_prob, target)
	return loss

def norm(x,y):
	return torch.sqrt(((x - y) ** 2).sum(1))

@register.attach('discriminative')
def discriminative_loss(inputs, data):

	margin_v = 0.5
	margin_d = 1.5

	seg_score = inputs['seg']
	embeddings = inputs['coords']
	ndims = embeddings.shape[1]
	label = data['label']
	instance_label = data['instance_label']

	device = embeddings.device
	seg_loss = cross_entropy(seg_score, label)

	loss_var = []
	loss_dist = []
	loss_reg = []

	for idx_n in range(instance_label.shape[0]):

		instance_mask = instance_label[idx_n]
		embeddings_n = embeddings[idx_n].permute(1,2,0)

		instances = torch.unique(instance_mask)
		Nc = len(instances) - 1
		if Nc == 0:
			continue

		instances = instances[1:]
		centroids = torch.zeros((Nc, ndims), dtype=torch.float).to(device)
		instance_mask = instance_mask.to(device)

		_loss_var = []
		_lost_dist = []
		_lost_reg = []

		for idx_i, i in enumerate(instances):
			x_i = embeddings_n[instance_mask == i]
			centroids[idx_i] = x_i.mean(0)
			diff_i = norm(x_i, centroids[idx_i]) - margin_v
			#diff_i = torch.sqrt(((x_i - centroids[i]) ** 2).sum(1)) - margin_v
			diff_i = torch.clamp(diff_i, min=0.0) ** 2
			_loss_var.append(diff_i.mean())
			_loss_reg.append(torch.norm(centroids[idx_i]))

		for idx_c, centroid in enumerate(centroids):
			diff_c = 2 * margin_d -  norm(centroids, centroid)
			diff_c[idx_c] = 0.0
			diff_c = torch.clamp(diff_c, min=0.0) ** 2
			_lost_dist.append(diff_c.sum() / (Nc - 1))

		loss_var.append(sum(_loss_var) / Nc)
		loss_reg.append(sum(_loss_reg) / Nc)
		loss_dist.append(sum(_lost_dist) / Nc)

	instance_loss = (sum(loss_var) + sum(loss_dist) + sum(loss_reg)) / instance_label.shape[0]

	return seg_loss + instance_loss


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









       

