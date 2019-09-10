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

	def hist(self, x_seg):
		bs = x_seg.shape[0]
		input_shape = x_seg.shape[-2:]
		x_seg_prob = torch.softmax(x_seg, dim=1)
		# plt.figure()
		# plt.imshow(x_seg_prob.transpose(0,1)[1].cpu().detach().numpy().squeeze())
		x_seg_prob = x_seg_prob.transpose(0,1)[1].view(*((bs, 1) + input_shape))
		x_hist = self.filter_bank(x_seg_prob, np.deg2rad(self.angles1).tolist()) + self.filter_bank(x_seg_prob,  np.deg2rad(self.angles2).tolist())
		x_hist = x_hist.view(bs, len(self.angles1), -1).sum(2)
		x_hist /= (x_hist.sum(1).view(bs,1).repeat([1, len(self.angles1)]) + 1)
		return x_hist



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
		x_hist = self.hist(x_seg)
		# x_seg_prob = torch.softmax(x_seg, dim=1)
		# x_seg_prob = x_seg_prob.transpose(0,1)[1].view(*((bs, 1) + input_shape))
		# x_hist = self.filter_bank(x_seg_prob, np.deg2rad(self.angles1).tolist()) + self.filter_bank(x_seg_prob,  np.deg2rad(self.angles2).tolist())
		# x_hist = x_hist.view(bs, len(self.angles1), -1).sum(2)
		# x_hist /= (x_hist.sum(1).view(bs,1).repeat([1, len(self.angles1)]) + 1)

		if self.training:
			result = OrderedDict()
			result["out"] = OrderedDict()
			result["out"]["seg"] = x_seg
			result["out"]["hist"] = x_hist
			if self.aux_clf is not None:
				x = features["aux"]
				x = self.aux_clf(x)
				x_seg_aux = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
				x_hist_aux = self.hist(x_seg_aux)
				result["aux"] = OrderedDict()
				result["aux"]["seg"] = x_seg_aux
				result["aux"]["hist"] = x_hist_aux
				# plt.show()
			return result
		else:
			return self.predict(x)

		return result

	def plot_filters(self):

		for angle in np.deg2rad(self.angles1).tolist()[4:7]:
			self.gabor_filter.plot_filter(angle)