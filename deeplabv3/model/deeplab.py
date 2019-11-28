import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision

import torchvision.models
import torch.utils.data
import pdb
from torchvision.models._utils import IntermediateLayerGetter
import torchvision.models.resnet as resnet
from torchvision.models.segmentation.deeplabv3 import DeepLabV3, ASPP
from torchvision.models.segmentation.fcn import FCN, FCNHead
from torch.nn import functional as F
from collections import OrderedDict
from deeplabv3.model.gabor import GaborConv2d, CoopConv2d
import matplotlib.pyplot as plt
import deeplabv3.lines as lines

class Deeplabv3(nn.Module):
	pass

class WarpNet(nn.Module):
	def __init__(self):
		super(WarpNet, self).__init__()
		self.offset_net = nn.Sequential(nn.Conv2d(2, 16, 1, 1, 0, 1, bias=True),
								   nn.Conv2d(16, 32, 1, 1, 0, 1, bias=True),
								   nn.Conv2d(32, 16, 1, 1, 0, 1, bias=True),
								   nn.Conv2d(16, 2, 1, 1, 0, 1, bias=True))
		self.offset_net.apply(init_conv)
		self.weights = nn.Parameter(torch.zeros((1, 2048), requires_grad=True))

	def forward(self, x_frame, x_mosaic, coords):
		batch_size = x_frame.shape[0]
		coords = F.interpolate(coords.transpose(3,1).transpose(3,2), size=x_frame.shape[-2:], mode='nearest')
		offset = self.offset_net(coords)
		#offset = 0.0
		coords = (coords + offset).transpose(3,1).transpose(1,2)
		

		x_frame_sampled = F.grid_sample(x_mosaic, coords)
		weights_sigmoid = torch.sigmoid(self.weights).repeat(batch_size, 1).view(batch_size, 2048, 1, 1)
		x_fuse = (1 - weights_sigmoid) * x_frame + weights_sigmoid * x_frame_sampled

		return x_fuse, offset


class MosaicNet(nn.Module):

	def __init__(self, frame_net, mosaic_backbone):
		super(MosaicNet, self).__init__()

		self.backbone = frame_net.backbone
		self.aspp = frame_net.aspp
		self.decoder = frame_net.decoder
		self.classifier = frame_net.classifier
		self.predict = frame_net.predict

		self.mosaic_backbone = mosaic_backbone
		self.warp_net = WarpNet()

	def get_device(self,):
		return self.classifier.weight.device

	def forward(self, inputs):

		device = self.get_device()
		frame = inputs['frame_img'].to(device)
		mosaic = inputs['mosaic_img'].to(device)
		grid_coords = inputs['grid_coords'].to(device)

		input_shape = frame.shape[-2:]
		frame_features = self.backbone(frame)
		x_frame = frame_features['out']
		mosaic_features = self.mosaic_backbone(mosaic)
		x_mosaic = mosaic_features['out']
		# print("mosaic_size:{}".format(x_mosaic.shape[-2:]))
		x_mosaic = F.interpolate(x_mosaic, scale_factor=16.0, mode='bilinear', align_corners=False)
		# print("mosaic_size:{}".format(x_mosaic.shape[-2:]))
		# print("frame_size:{}".format(x_frame.shape[-2:]))
		# pdb.set_trace()
		x, offset = self.warp_net(x_frame, x_mosaic, grid_coords)
		x_low = frame_features['skip1']
		x = self.aspp(x)
		x = self.decoder(x, x_low)
		x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
		x = self.classifier(x)
		
		if self.training:
			result = OrderedDict()
			result["out"] = OrderedDict()
			result["out"]["seg"] = x
			result["out"]["offset"] = offset
			return result
		else:
			return self.predict(x)

		


class _Deeplabv3Plus(nn.Module):
	def __init__(self, n_classes, pretrained_model, decoder, predict, aux=False):
		super(_Deeplabv3Plus, self).__init__()
		self.backbone = pretrained_model.backbone
		self.aspp = list(pretrained_model.classifier.children())[0]
		self.decoder = decoder
		self.decoder.apply(init_conv)
		self.classifier = nn.Conv2d(256, n_classes, 1, 1, 0, 1, bias=False)
		init_conv(self.classifier)
		self.predict = predict
		self.aux_clf = None

		if aux:
			aux_clf = pretrained_model.aux_classifier
			aux_clf_out = nn.Conv2d(256, n_classes, 1, 1, 0, 1, bias=False)
			init_conv(aux_clf_out)
			aux_clf_layers = list(aux_clf.children())
			aux_clf_layers[-1] = aux_clf_out
			self.aux_clf = nn.Sequential(*aux_clf_layers)


	def forward(self, x):
		raise NotImplementedError


class Deeplabv3Plus1(_Deeplabv3Plus):
	def __init__(self, n_classes, pretrained_model, predict, aux=False, out_planes_skip=48):
		super(Deeplabv3Plus1, self).__init__(n_classes, 
											pretrained_model, 
											DeepLabDecoder1(256, out_planes=out_planes_skip), 
											predict,
											aux=aux)

	def get_device(self,):
		return self.classifier.weight.device

	def forward(self, inputs):

		x = inputs['image'].to(self.get_device())
		input_shape = x.shape[-2:]
		features = self.backbone(x)
		x = features["out"]
		x_low = features["skip1"]
		x = self.aspp(x)
		x = self.decoder(x, x_low)
		x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
		x = self.classifier(x)
		
		if self.training:
			result = OrderedDict()
			result["out"] = OrderedDict()
			result["out"]["out"] = x
			if self.aux_clf is not None:
				x = features["aux"]
				x = self.aux_clf(x)
				x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
				result["aux"] = x
			return result
		else:
			return self.predict(x, inputs)

class Deeplabv3PlusLines(_Deeplabv3Plus):
	def __init__(self, n_classes, pretrained_model, predict, aux=False, out_planes_skip=48):
		super(Deeplabv3PlusLines, self).__init__(n_classes, 
											pretrained_model, 
											DeepLabDecoder1(256, out_planes=out_planes_skip), 
											predict,
											aux=aux)

		self.lines_clf = nn.Linear(256, 1, bias=False)
		self.lines_clf.weight = None

		angle_step = 15.0
		angles1 = np.deg2rad(np.arange(-30.0, 30.0 + angle_step, angle_step))
		self.angles_v = np.rad2deg(angles1)
		angles2 = angles1 + np.pi/2
		angles = np.array(angles1.tolist() + angles2.tolist())
		self.num_angles = len(angles1)
		self.filter_bank = []
		for angle in angles:
			self.filter_bank.append(GaborConv2d(angle))

		self.line_sampler = lines.LineSampler()
		
	def load_state_dict(self, state_dict, strict=True):
		super(Deeplabv3PlusLines, self).load_state_dict(state_dict, strict)
		if self.lines_clf.weight is None:
			w0, w1 = self.classifier.weight.data[:2]
			init_weight = (w1-w0).squeeze().unsqueeze(0)
			self.lines_clf.weight = nn.Parameter(init_weight)

	def get_device(self,):
		return self.classifier.weight.device

	def compute_angle_range(self, x_seg):

		prob = torch.softmax(x_seg.transpose(0,1)[:2].transpose(0,1), dim=1)
		prob = prob.transpose(0,1)[1].unsqueeze(1)
		pred = x_seg.transpose(0,1)[1].unsqueeze(1)
		bs = pred.shape[0]
		pred = pred - pred.view(bs, -1).mean(1).view(bs,1,1,1)

		res = []
		for gabor_filter in self.filter_bank:
			res.append(gabor_filter(pred))

		res = F.relu(torch.cat(res, dim=1))
		res = res / (res.sum(1).unsqueeze(1) + 1.0)

		res1 = res.transpose(0,1)[:self.num_angles].transpose(0,1)
		res2 = res.transpose(0,1)[self.num_angles:].transpose(0,1)
		res = (res1 + res2) * prob

		for _res in res[0]:
			plt.figure()
			plt.imshow(_res.cpu().detach().numpy().squeeze(), vmax=1.0)
		plt.show()

		hist = res.view(bs, self.num_angles, -1).sum(2)
		hist /= (hist.sum(1).unsqueeze(1).repeat([1, self.num_angles]) + 1.0)
		print(hist)
		hist = hist.cpu().numpy().squeeze()
		angle_indices = np.argsort(hist)[-2:]
		angle_range_v = self.angles_v[angle_indices]
		angle_range_v.sort()
		angle_range_h = angle_range_v + 90

		return angle_range_v, angle_range_h


	def forward(self, inputs):

		x = inputs['image'].to(self.get_device())
		input_shape = x.shape[-2:]
		features = self.backbone(x)
		x = features["out"]
		x_low = features["skip1"]
		x = self.aspp(x)
		x = self.decoder(x, x_low)
		x_features = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
		x_seg = self.classifier(x_features)

		angle_range_v, angle_range_h = self.compute_angle_range(x_seg)
		print(">> {}".format(angle_range_v))
		sampled_points_v, proposed_lines_v = self.line_sampler(angle_range_v, tuple(input_shape))
		sampled_points_h, proposed_lines_h = self.line_sampler(angle_range_h, tuple(input_shape))

		# pdb.set_trace()

		proposed_lines = np.array(proposed_lines_v + proposed_lines_h, dtype=np.float32)
		sampled_points = np.stack(sampled_points_v + sampled_points_h)[np.newaxis,...]
		grid = torch.Tensor(sampled_points).to(self.get_device())

		# grid = inputs['line_points'].to(self.get_device())
		# proposed_lines = inputs['line_coeffs'].cpu().numpy().squeeze()
		sampled_lines = F.grid_sample(x_features, grid)
		line_features = sampled_lines.transpose(1,2).mean(3)
		line_probs = torch.sigmoid(self.lines_clf(line_features)).squeeze()

		if self.training:
			result = OrderedDict()
			result["out"] = OrderedDict()
			result["out"]["out"] = line_probs
		else:
			#return self.predict(line_probs, inputs)
			return self.predict(line_probs, inputs, proposed_lines)


class Deeplabv3PlusLines2(_Deeplabv3Plus):
	def __init__(self, n_classes, pretrained_model, predict, aux=False, out_planes_skip=48):
		super(Deeplabv3PlusLines2, self).__init__(n_classes, 
											pretrained_model, 
											DeepLabDecoder1(256, out_planes=out_planes_skip), 
											predict,
											aux=False)

		self.offset_reg = nn.Linear(256, 4, bias=False)
		init_conv(self.offset_reg)

		self.line_clf = nn.Linear(256, 1, bias=False)

		self.res_net = nn.Sequential(
			nn.Conv2d(256, 64, kernel_size=1, stride=1, bias=False),
			nn.GroupNorm(8, 64),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=12, dilation=12, bias=False),
			nn.GroupNorm(8, 64),
			nn.ReLU(),
			nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
			nn.GroupNorm(32, 256),
			nn.ReLU())
		self.res_net.apply(init_conv)

		angle_step = 15.0
		angles1 = np.deg2rad(np.arange(-30.0, 30.0 + angle_step, angle_step))
		self.angles_v = np.rad2deg(angles1)
		angles2 = angles1 + np.pi/2
		angles = np.array(angles1.tolist() + angles2.tolist())
		self.num_angles = len(angles1)
		self.filter_bank = []
		for angle in angles:
			self.filter_bank.append(GaborConv2d(angle))

		self.line_sampler = lines.LineSampler(angle_step=1.0, rho_step=50)
		
	def load_state_dict(self, state_dict, strict=True):
		super(Deeplabv3PlusLines2, self).load_state_dict(state_dict, strict)
		if 'line_clf.weight' not in state_dict:
			w0, w1 = self.classifier.weight.data[:2]
			init_weight = (w1-w0).squeeze().unsqueeze(0)
			self.line_clf.weight = nn.Parameter(init_weight)

	def get_device(self,):
		return self.classifier.weight.device

	def trainable_parameters(self,):
		params1 = list(self.offset_reg.parameters())
		params2 = list(self.res_net.parameters())
		return params1+params2

	def compute_angle_range(self, x_seg):

		prob = torch.softmax(x_seg.transpose(0,1)[:2].transpose(0,1), dim=1)
		prob = prob.transpose(0,1)[1].unsqueeze(1)
		pred = x_seg.transpose(0,1)[1].unsqueeze(1)
		bs = pred.shape[0]
		pred = pred - pred.view(bs, -1).mean(1).view(bs,1,1,1)

		res = []
		for gabor_filter in self.filter_bank:
			res.append(gabor_filter(pred))

		res = F.relu(torch.cat(res, dim=1))
		res = res / (res.sum(1).unsqueeze(1) + 1.0)

		res1 = res.transpose(0,1)[:self.num_angles].transpose(0,1)
		res2 = res.transpose(0,1)[self.num_angles:].transpose(0,1)
		res = (res1 + res2) * prob

		for _res in res[0]:
			plt.figure()
			plt.imshow(_res.cpu().detach().numpy().squeeze(), vmax=1.0)
		plt.show()

		hist = res.view(bs, self.num_angles, -1).sum(2)
		hist /= (hist.sum(1).unsqueeze(1).repeat([1, self.num_angles]) + 1.0)
		hist = hist.cpu().numpy().squeeze()
		angle_indices = np.argsort(hist)[-2:]
		angle_weights = hist[angle_indices]
		angle_v = (self.angles_v[angle_indices] * angle_weights).sum() / angle_weights.sum()
		angle_range_v = np.array((angle_v, angle_v))
		angle_range_h = angle_range_v + 90

		return angle_range_v, angle_range_h

	def normalize_sampled_features(self, x_sampled_features):
		x_mean = x_sampled_features.mean(2).unsqueeze(2)
		x_std = x_sampled_features.std(2).unsqueeze(2)
		return (x_sampled_features - x_mean) / x_std

	def sample_features(self, x_features, x_features_aux, grid):

		sampled_lines = F.grid_sample(x_features, grid).transpose(1,2).mean(3)
		sampled_lines = self.normalize_sampled_features(sampled_lines)

		sampled_lines_aux = F.grid_sample(x_features_aux, grid).transpose(1,2).mean(3)
		sampled_lines_aux = self.normalize_sampled_features(sampled_lines_aux)

		return torch.cat([sampled_lines, sampled_lines_aux], dim=2)


	def extract_features(self, x):

		input_shape = x.shape[-2:]
		features = self.backbone(x)
		x = features["out"]
		x_low = features["skip1"]
		x = self.aspp(x)
		x_features = self.decoder(x, x_low)
		x_features_up = F.interpolate(x_features, size=input_shape, mode='bilinear', align_corners=False)
		x_seg = self.classifier(x_features_up)

		# x_features_aux = self.aux_net(features["aux"])
		# x_features_aux = F.interpolate(x_features_aux, size=input_shape, mode='bilinear', align_corners=False)

		return x_seg, x_features, x_features_up


	def forward(self, inputs):

		x = inputs['image'].to(self.get_device())
		input_shape = x.shape[-2:]
		x_seg, x_features, x_features_up = self.extract_features(x)

		if self.training:
			grid = inputs['line_points'].to(self.get_device())
		else:
			angle_range_v, angle_range_h = self.compute_angle_range(x_seg)
			sampled_points_v, _, lines_intersect_v = self.line_sampler(angle_range_v, tuple(input_shape))
			sampled_points_h, _, lines_intersect_h = self.line_sampler(angle_range_h, tuple(input_shape))
			sampled_points = np.stack(sampled_points_v + sampled_points_h)[np.newaxis,...]
			grid = torch.Tensor(sampled_points).to(self.get_device())

		# line_features = self.sample_features(x_features, x_features_aux, grid)
		# line_features = self.fc_net(line_features)
		line_features = F.grid_sample(x_features_up, grid).transpose(1,2).mean(3)
		res_features = F.interpolate(self.res_net(x_features), size=input_shape, mode='bilinear', align_corners=False)
		line_res_features = F.grid_sample(res_features, grid).transpose(1,2).mean(3)
		offset = self.offset_reg(line_res_features)
		score = self.line_clf(line_features + line_res_features).squeeze(2)

		if self.training:
			result = OrderedDict()
			result["out"] = OrderedDict()
			result["out"]["score"] = score
			result["out"]["offset"] = offset
			result["out"]["seg"] = x_seg
			return result
		else:
			lines_intersect = np.array(lines_intersect_v + lines_intersect_h, dtype=np.float32)
			return self.predict(lines_intersect, score, offset, inputs)





class Deeplabv3PlusLines3(_Deeplabv3Plus):
	def __init__(self, n_classes, pretrained_model, predict, aux=False, out_planes_skip=48):
		super(Deeplabv3PlusLines3, self).__init__(n_classes, 
											pretrained_model, 
											DeepLabDecoder1(256, out_planes=out_planes_skip), 
											predict,
											aux=False)
	
		# self.reduce_net = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
		# 	    						nn.GroupNorm(8, 64),
		# 	    						nn.ReLU())
		# self.reduce_net.apply(init_conv)
		
		self.line_net = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=0, groups=256, bias=False),
									  nn.GroupNorm(256, 256),
									  nn.ReLU(),
									  nn.Conv2d(256, 256, kernel_size=3, padding=0, groups=256, bias=False),
									  nn.GroupNorm(256, 256),
									  nn.ReLU())
		self.line_net.apply(init_conv)

		self.iou_clf = nn.Conv2d(256, 1, kernel_size=3, padding=0, bias=False)
		init_conv(self.iou_clf)
		self.offset_reg = nn.Conv2d(256, 1, kernel_size=3, padding=0, bias=False)
		init_conv(self.offset_reg)
		self.score_clf = nn.Conv2d(256, 1, kernel_size=1, padding=0, bias=False)
		# init_conv(self.score_clf)

		self.avg_pooling = nn.AdaptiveAvgPool2d(7)
		self.max_pooling = nn.MaxPool2d(7)

		angle_step = 15.0
		angles1 = np.deg2rad(np.arange(-30.0, 30.0 + angle_step, angle_step))
		self.angles_v = np.rad2deg(angles1)
		angles2 = angles1 + np.pi/2
		angles = np.array(angles1.tolist() + angles2.tolist())
		self.num_angles = len(angles1)
		self.filter_bank = []
		for angle in angles:
			self.filter_bank.append(GaborConv2d(angle))

		self.line_sampler = lines.LineSampler(angle_step=1.0, rho_step=50)
		self.ROI_sampler = lines.ROISampler()
		
	def load_state_dict(self, state_dict, strict=True):
		super(Deeplabv3PlusLines3, self).load_state_dict(state_dict, strict)
		if 'score_clf.weight' not in state_dict:
			print(">>>>>>>>>>")
			w0, w1 = self.classifier.weight.data[:2]
			init_weight = (w1-w0).view_as(self.score_clf.weight)
			self.score_clf.weight = nn.Parameter(init_weight)

	def get_device(self,):
		return self.classifier.weight.device

	def trainable_parameters(self,):
		params = []
		# params += list(self.reduce_net.parameters())
		# params += list(self.line_net.parameters())
		params += list(self.iou_clf.parameters())
		params += list(self.score_clf.parameters())
		params += list(self.offset_reg.parameters())
		return params

	def compute_angle_range(self, x_seg):

		prob = torch.softmax(x_seg.transpose(0,1)[:2].transpose(0,1), dim=1)
		prob = prob.transpose(0,1)[1].unsqueeze(1)
		pred = x_seg.transpose(0,1)[1].unsqueeze(1)
		bs = pred.shape[0]
		pred = pred - pred.view(bs, -1).mean(1).view(bs,1,1,1)

		res = []
		for gabor_filter in self.filter_bank:
			res.append(gabor_filter(pred))

		res = F.relu(torch.cat(res, dim=1))
		res = res / (res.sum(1).unsqueeze(1) + 1.0)

		res1 = res.transpose(0,1)[:self.num_angles].transpose(0,1)
		res2 = res.transpose(0,1)[self.num_angles:].transpose(0,1)
		res = (res1 + res2) * prob

		# for _res in res[0]:
		# 	plt.figure()
		# 	plt.imshow(_res.cpu().detach().numpy().squeeze(), vmax=1.0)
		# plt.show()

		hist = res.view(bs, self.num_angles, -1).sum(2)
		hist /= (hist.sum(1).unsqueeze(1).repeat([1, self.num_angles]) + 1.0)
		hist = hist.cpu().numpy().squeeze()
		angle_indices = np.argsort(hist)[-2:]
		angle_weights = hist[angle_indices]
		angle_v = (self.angles_v[angle_indices] * angle_weights).sum() / angle_weights.sum()
		angle_range_v = np.array((angle_v, angle_v))
		angle_range_h = angle_range_v + 90

		return angle_range_v, angle_range_h


	def extract_features(self, x):

		input_shape = x.shape[-2:]
		features = self.backbone(x)
		x = features["out"]
		x_low = features["skip1"]
		x = self.aspp(x)
		x_features = self.decoder(x, x_low)
		x_features_up = F.interpolate(x_features, size=input_shape, mode='bilinear', align_corners=False)
		x_seg = self.classifier(x_features_up)

		return x_seg, x_features

	def debug(self, x_seg, grid):
		x_seg = x_seg.transpose(0,1)[1].unsqueeze(0)
		x_seg = torch.sigmoid(x_seg)
		_line_features = F.grid_sample(x_seg, grid)
		_line_features_1 = _line_features.cpu().detach().numpy().squeeze()
		_line_features = self.avg_pooling(_line_features)
		_line_features_2 = _line_features.cpu().detach().numpy().squeeze()
		plt.figure()
		plt.imshow(_line_features_1)
		plt.figure()
		plt.imshow(_line_features_2)
		plt.show()




	def forward(self, inputs):

		x = inputs['image'].to(self.get_device())
		input_shape = x.shape[-2:]
		x_seg, x_features = self.extract_features(x)
		# x_features = self.reduce_net(x_features)
		x_features_up = F.interpolate(x_features, size=input_shape, mode='bilinear', align_corners=False)

		line_intersects = inputs['intersects_points'].cpu().numpy().squeeze()

		# if self.training:
		# 	intersect_points = inputs['sampled_points'].cpu().numpy().squeeze()
		# else:
		# 	angle_range_v, angle_range_h = self.compute_angle_range(x_seg)
		# 	sampled_points_v, proposed_lines_v, _ = self.line_sampler(angle_range_v, tuple(input_shape))
		# 	sampled_points_h, proposed_lines_h, _ = self.line_sampler(angle_range_h, tuple(input_shape))
		# 	sampled_points = np.stack(sampled_points_v + sampled_points_h)[np.newaxis,...]
		# 	grid = torch.Tensor(sampled_points).to(self.get_device())

		nlines = line_intersects.shape[0]
		line_features = []
		for idx in np.arange(nlines):
			intersect_points = line_intersects[idx]
			grid = self.ROI_sampler(intersect_points, input_shape)
			grid = torch.Tensor(grid).to(self.get_device())
			grid = grid.unsqueeze(0)
			# self.debug(x_seg, grid)
			_line_features = F.grid_sample(x_features_up, grid)
			_line_features = self.avg_pooling(_line_features)
			# _line_features = self.line_net(_line_features)
			line_features.append(_line_features)

		line_features = torch.cat(line_features, 0)

		score = self.score_clf(self.max_pooling(line_features)).squeeze().unsqueeze(0)
		iou = self.iou_clf(line_features).squeeze().unsqueeze(0)
		offset = self.offset_reg(line_features).squeeze().unsqueeze(0)

		if self.training:
			result = OrderedDict()
			result["out"] = OrderedDict()
			result["out"]["score"] = score
			result["out"]["offset"] = offset
			result["out"]["iou"] = iou
			return result
		else:
			proposed_lines = np.array(proposed_lines_v + proposed_lines_h, dtype=np.float32)
			score = torch.sigmoid(score).cpu().numpy().squeeze()
			iou = iou.cpu().numpy().squeeze()
			reg_offset = reg_offset.cpu().numpy().squeeze()
			return self.predict(proposed_lines, score, iou, reg_offset, input_shape)

class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes, atrous_rates):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, atrous_rates),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )


class DeepLabDecoder1(nn.Module):
	def __init__(self, in_planes, out_planes=48):
		super(DeepLabDecoder1, self).__init__()
		self.reduce_conv = nn.Sequential(
			nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False),
			nn.GroupNorm(int(out_planes/8), out_planes),
			nn.ReLU(),
			nn.Dropout(0.5))
		self.fuse_conv = nn.Sequential(
			nn.Conv2d(256 + out_planes, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.GroupNorm(32, 256),
			nn.ReLU(),
			nn.Dropout(0.1),
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.GroupNorm(32, 256),
			nn.ReLU())

	def forward(self, x, x_low):
		low_size = x_low.shape[-2:]
		high_size = x.shape[-2:]
		x_low = self.reduce_conv(x_low)
		if low_size > high_size:
			x = F.interpolate(x, size=low_size, mode='bilinear', align_corners=False)
		x = torch.cat([x_low, x], dim=1)
		x = self.fuse_conv(x)
		return x


class DeepLabDecoder2(nn.Module):
	def __init__(self, in_planes1, in_planes2, out_planes1=32, out_planes2=48):
		super(DeepLabDecoder2, self).__init__()
		self.reduce_conv1 = nn.Sequential(
			nn.Conv2d(in_planes1, out_planes1, kernel_size=1, stride=1, bias=False),
			nn.GroupNorm(int(out_planes1 / 8), out_planes1),
			nn.ReLU(),
			nn.Dropout(0.5))
		self.reduce_conv2 = nn.Sequential(
			nn.Conv2d(in_planes2, out_planes2, kernel_size=1, stride=1, bias=False),
			nn.GroupNorm(int(out_planes2 / 8), out_planes2),
			nn.ReLU(),
			nn.Dropout(0.5))
		self.fuse_conv = nn.Sequential(
			nn.Conv2d(256 + out_planes1 + out_planes2, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.GroupNorm(32, 256),
			nn.ReLU(),
			nn.Dropout(0.1),
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.GroupNorm(32, 256),
			nn.ReLU())

	def forward(self, x, x_low1, x_low2):
		low_size1 = x_low1.shape[-2:]
		low_size2 = x_low2.shape[-2:]
		high_size = x.shape[-2:]
		x_low1 = self.reduce_conv1(x_low1)
		x_low2 = self.reduce_conv2(x_low2)
		if low_size2 < low_size1:
			x_low2 = F.interpolate(x_low2, size=low_size1, mode='bilinear', align_corners=False)
		if high_size < low_size1:
			x = F.interpolate(x, size=low_size1, mode='bilinear', align_corners=False)
		x = torch.cat([x_low1, x_low2, x], dim=1)
		x = self.fuse_conv(x)
		return x


def _segm_resnet(name, backbone_name, num_classes, aux, pretrained_backbone=True, replace_stride_with_dilation=[False, True, True], rates=[12, 24, 36], return_layers = {'layer4': 'out'}):
	backbone = resnet.__dict__[backbone_name](
		pretrained=pretrained_backbone,
		replace_stride_with_dilation=replace_stride_with_dilation)

	backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

	aux_classifier = None
	if aux:
		inplanes = 1024
		aux_classifier = FCNHead(inplanes, num_classes)

	model_map = {
		'deeplabv3': (DeepLabHead, DeepLabV3),
		'fcn': (FCNHead, FCN),
	}
	inplanes = 2048
	classifier = model_map[name][0](inplanes, num_classes, rates)
	base_model = model_map[name][1]

	model = base_model(backbone, classifier, aux_classifier)

	return model

def load_pretrained_model(kw_backbone_args):
	pretrained_model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True, num_classes=21)
	model = _segm_resnet('deeplabv3', 'resnet101', 21, True, **kw_backbone_args)
	model.load_state_dict(pretrained_model.state_dict(), strict=True)
	return model

def init_conv(m):
	if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
		nn.init.xavier_uniform_(m.weight)

