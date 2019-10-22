import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from .deeplab import _Deeplabv3Plus, init_conv, DeepLabDecoder1
from torch.nn import functional as F
from collections import OrderedDict
import pdb
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from .gabor import GaborConv2d
import pdb

class Deeplabv3PlusOri_v2(_Deeplabv3Plus):

	min_angle = -45
	max_angle = 45
	bin_len = 10
	n_ori = int((max_angle - min_angle) / bin_len)

	def __init__(self, n_classes, pretrained_model, predict, aux=False, out_planes_skip=48):
		super(Deeplabv3PlusOri_v2, self).__init__(n_classes, 
											pretrained_model, 
											DeepLabDecoder1(256, out_planes=out_planes_skip), 
											predict,
											aux=aux)
		self.ori_net1 = nn.Sequential(
			nn.Conv2d(2, 64, kernel_size=31, stride=2, padding=17, bias=False),
			nn.GroupNorm(8, 64),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
			nn.GroupNorm(8, 64),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
			nn.GroupNorm(8, 64),
			nn.ReLU(),
			)
		self.ori_net1.apply(init_conv)

		self.ori_net2 = nn.Sequential(
			nn.Linear(53 * 105 * 64, 1000),
			nn.ReLU(),
			nn.Linear(1000, self.n_ori),
			)
		self.ori_net2.apply(init_conv)

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
		x_seg_prob = torch.softmax(x_seg, dim=1)
		x_seg_1 = x_seg_prob.transpose(0,1)[1].view(*((bs, 1) + input_shape))
		plt.figure()
		plt.imshow(x_seg_1.cpu().detach().numpy().squeeze())
		for angle in [-30, -15, 0, 15, 30]:
			x_gabor = GaborSmallConv2d(in_channels=1, out_channels=1, kernel_size=51, padding=21, angle=angle)(x_seg_1)
			print("{}, {}".format(angle, x_gabor.sum().cpu().detach().numpy()))
			plt.figure()
			plt.imshow(x_gabor.cpu().detach().numpy().squeeze())
		# x_10 = GaborSmallConv2d(in_channels=1, out_channels=1, kernel_size=51, padding=25, angle=10)(x_seg_1)
		# GaborSmallConv2d(in_channels=1, out_channels=1, kernel_size=51, padding=25, angle=20)(x_seg_1)
		# GaborSmallConv2d(in_channels=1, out_channels=1, kernel_size=51, padding=25, angle=30)(x_seg_1)
		# plt.imshow(x_seg_1)
		plt.show()
		pdb.set_trace()
		x_ori = self.ori_net1(x_seg.transpose(0,1)[:2].transpose(0,1))
		x_ori = self.ori_net2(x_ori.view(bs,-1))

		result = OrderedDict()
		result["out"] = OrderedDict()
		result["out"]["seg"] = x_seg
		result["out"]["ori"] = x_ori

		return result


class Deeplabv3PlusHist(_Deeplabv3Plus):

	min_angle = -40
	max_angle = 40
	angle_step = 10
	angles1 = np.arange(min_angle, max_angle + angle_step, angle_step)
	angles2 = angles1 + 90
	
	def __init__(self, n_classes, pretrained_model, predict, aux=False, out_planes_skip=48):
		super(Deeplabv3PlusHist, self).__init__(n_classes, 
											pretrained_model, 
											DeepLabDecoder1(256, out_planes=out_planes_skip), 
											predict,
											aux=aux)
		self.gabor_filter = GaborConv2d()
		self.relu = nn.ReLU()
		self.tresh = 100.0

	def filter_bank(self, x, angles):
		res = []
		for angle in angles:
			res.append(self.gabor_filter(x, angle))
		res = torch.cat(res, dim=1)
		return self.relu(res - self.tresh)

	def forward(self, x):

		input_shape = x.shape[-2:]
		bs = x.shape[0]
		features = self.backbone(x)
		x = features["out"]
		x_low = features["skip1"]
		x = self.aspp(x)
		x = self.decoder(x, x_low)
		x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
		x_seg = self.classifier(x)
		x_seg_prob = torch.softmax(x_seg.transpose(0,1)[:2].transpose(0,1), dim=1)
		x_seg_prob = x_seg_prob.transpose(0,1)[1].view(*((bs, 1) + input_shape))
		x_hist = self.filter_bank(x_seg_prob, np.deg2rad(self.angles1).tolist()) + self.filter_bank(x_seg_prob,  np.deg2rad(self.angles2).tolist())
		x_hist = x_hist.view(bs, len(self.angles1), -1).sum(2)
		x_hist /= x_hist.sum(1).repeat([1, len(self.angles1)])


		result = OrderedDict()
		result["out"] = OrderedDict()
		result["out"]["seg"] = x_seg
		result["out"]["hist"] = x_hist

		return result

	def plot_filters(self):

		for angle in np.deg2rad(self.angles1).tolist()[4:7]:
			self.gabor_filter.plot_filter(angle)


class Deeplabv3PlusAngleClf(_Deeplabv3Plus):

	min_angle = 0
	max_angle = 180
	angle_step = 7.5
	bank_angles = np.deg2rad(np.arange(min_angle, max_angle, angle_step))

	n_classes_angle = len(np.arange(-45, 45, 7.5))
	
	def __init__(self, n_classes, pretrained_model, predict, aux=False, out_planes_skip=48):
		super(Deeplabv3PlusAngleClf, self).__init__(n_classes, 
											pretrained_model, 
											DeepLabDecoder1(256, out_planes=out_planes_skip), 
											predict,
											aux=aux)
		self.gabor_filter = GaborConv2d()
		self.angle_net = nn.Sequential(
			nn.Conv2d(len(self.bank_angles), 128, kernel_size=7, stride=2, padding=12, dilation=4, bias=False),
			nn.GroupNorm(8, 128),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=7, stride=2, padding=3),
			nn.Conv2d(128, 256, kernel_size=7, stride=2, padding=3, bias=False),
			nn.GroupNorm(16, 256),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=7, stride=2, padding=3),
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(256, 1000, kernel_size=1),
			nn.ReLU(),
			nn.Conv2d(1000, self.n_classes_angle, kernel_size=1)
			)
		self.angle_net.apply(init_conv)


	def filter_bank(self, x):
		res = []
		for angle in self.bank_angles:
			res.append(self.gabor_filter(x, angle))
		return torch.cat(res, dim=1)

	def forward(self, x):

		input_shape = x.shape[-2:]
		bs = x.shape[0]
		features = self.backbone(x)
		x = features["out"]
		x_low = features["skip1"]
		x = self.aspp(x)
		x = self.decoder(x, x_low)
		x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
		x_seg = self.classifier(x)
		x_seg_prob = torch.softmax(x_seg, dim=1)
		x_seg_prob = x_seg_prob.transpose(0,1)[1].view(*((bs, 1) + input_shape))
		x_hist = self.filter_bank(x_seg_prob)
		x_angle = self.angle_net(x_hist)

		result = OrderedDict()
		result["out"] = OrderedDict()
		result["out"]["seg"] = x_seg
		result["out"]["ori"] = x_angle

		return result






