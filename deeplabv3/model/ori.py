import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from .deeplab import _Deeplabv3Plus, init_conv, DeepLabDecoder1
from torch.nn import functional as F
from collections import OrderedDict
import pdb

class Deeplabv3PlusOri(_Deeplabv3Plus):

	min_angle = -35
	max_angle = 125
	bin_len = 4

	def __init__(self, n_classes, pretrained_model, predict, aux=False, out_planes_skip=48):
		super(Deeplabv3PlusOri, self).__init__(n_classes, 
											pretrained_model, 
											DeepLabDecoder1(256, out_planes=out_planes_skip), 
											predict,
											aux=aux)

		n_ori = int((self.max_angle - self.min_angle) / self.bin_len)
		self.ori_clf = nn.Conv2d(256, n_ori + 1, 1, 1, 0, 1, bias=False)
		init_conv(self.ori_clf)

	def forward(self, x):

		input_shape = x.shape[-2:]
		features = self.backbone(x)
		x = features["out"]
		x_low = features["skip1"]
		x = self.aspp(x)
		x = self.decoder(x, x_low)
		x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

		x_seg = self.classifier(x)
		x_ori = self.ori_clf(x)

		result = OrderedDict()
		result["out"] = OrderedDict()
		result["out"]["seg"] = x_seg
		result["out"]["ori"] = x_ori

		return result

class Deeplabv3PlusOri2(_Deeplabv3Plus):

	min_angle = -35
	max_angle = 125
	bin_len = 10

	def __init__(self, n_classes, pretrained_model, predict, aux=False, out_planes_skip=48):
		super(Deeplabv3PlusOri2, self).__init__(n_classes, 
											pretrained_model, 
											DeepLabDecoder1(256, out_planes=out_planes_skip), 
											predict,
											aux=aux)

		self.n_ori = int((self.max_angle - self.min_angle) / self.bin_len)
		self.ori_net = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=7, stride=1, padding=3, bias=False),
			nn.GroupNorm(8*4, 256),
			nn.ReLU(),
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.GroupNorm(8*4, 256),
			nn.ReLU())
		self.ori_net.apply(init_conv)
		self.ori_clf = nn.Conv2d(256, self.n_ori + 1, 1, 1, 0, 1, bias=False)
		init_conv(self.ori_clf)

	def forward(self, x):

		input_shape = x.shape[-2:]
		bs = x.shape[0]
		features = self.backbone(x)
		x = features["out"]
		x_low = features["skip1"]
		x = self.aspp(x)
		x_out = self.decoder(x, x_low)

		x_seg = F.interpolate(x_out, size=input_shape, mode='bilinear', align_corners=False)
		x_seg = self.classifier(x_seg)

		x_ori = self.ori_net(x_out)
		x_ori = F.interpolate(x_ori, size=input_shape, mode='bilinear', align_corners=False)
		x_ori = self.ori_clf(x_ori)

		seg_prob = torch.softmax(x_seg.transpose(0,1)[:2].transpose(0,1), dim=1)
		seg_prob_0 = seg_prob.transpose(0,1)[0]
		seg_prob_1 = seg_prob.transpose(0,1)[1]
		prob_mask = seg_prob_1.view(bs, 1, -1).repeat(1, self.n_ori, 1).view(*((bs, self.n_ori) + input_shape))
		prob_mask = torch.cat([prob_mask, seg_prob_0.view(*((bs, 1) + input_shape))], dim=1)
		x_ori *= prob_mask

		result = OrderedDict()
		result["out"] = OrderedDict()
		result["out"]["seg"] = x_seg
		result["out"]["ori"] = x_ori

		return result

class Deeplabv3PlusOri3(_Deeplabv3Plus):

	min_angle = -35
	max_angle = 125
	bin_len = 10

	def __init__(self, n_classes, pretrained_model, predict, aux=False, out_planes_skip=48):
		super(Deeplabv3PlusOri3, self).__init__(n_classes, 
											pretrained_model, 
											DeepLabDecoder1(256, out_planes=out_planes_skip), 
											predict,
											aux=aux)

		n_ori = int((self.max_angle - self.min_angle) / self.bin_len)

		self.ori_net1 = nn.Sequential(
			nn.Conv2d(256, 64, kernel_size=7, stride=1, padding=3, bias=False),
			nn.GroupNorm(8, 64),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.GroupNorm(8, 64),
			nn.ReLU())
		self.ori_net1.apply(init_conv)

		self.ori_net2 = nn.Sequential(
			nn.Conv2d(2, 64, kernel_size=57, stride=1, padding=28, bias=False),
			nn.GroupNorm(8, 64),
			nn.ReLU())
		self.ori_net2.apply(init_conv)

		self.ori_clf = nn.Conv2d(64 * 2, n_ori + 1, 1, 1, 0, 1, bias=False)
		init_conv(self.ori_clf)

	def forward(self, x):

		input_shape = x.shape[-2:]
		features = self.backbone(x)
		x = features["out"]
		x_low = features["skip1"]
		x = self.aspp(x)
		x_out = self.decoder(x, x_low)

		x_seg = F.interpolate(x_out, size=input_shape, mode='bilinear', align_corners=False)
		x_seg = self.classifier(x_seg)

		x_ori1 = self.ori_net1(x_out)
		x_ori1 = F.interpolate(x_ori1, size=input_shape, mode='bilinear', align_corners=False)
		x_ori2 = self.ori_net2(x_seg.transpose(0,1)[:2].transpose(0,1))
		x_ori = torch.cat([x_ori1, x_ori2], dim=1)
		x_ori = self.ori_clf(x_ori)

		result = OrderedDict()
		result["out"] = OrderedDict()
		result["out"]["seg"] = x_seg
		result["out"]["ori"] = x_ori

		return result

