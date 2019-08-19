import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from .deeplab import _Deeplabv3Plus, init_conv, DeepLabDecoder1
from torch.nn import functional as F
from collections import OrderedDict

class Deeplabv3PlusInstance1(_Deeplabv3Plus):
	def __init__(self, n_classes, pretrained_model, predict, aux=False, out_planes_skip=48):
		super(Deeplabv3PlusInstance1, self).__init__(n_classes, 
											pretrained_model, 
											DeepLabDecoder1(256, out_planes=out_planes_skip), 
											predict,
											aux=aux)
		self.instance_net = nn.Sequential(
			nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.GroupNorm(8, 64),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.GroupNorm(8, 64),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.GroupNorm(8, 64),
			nn.ReLU())
		self.instance_net.apply(init_conv)

		self.embeddings_conv = nn.Conv2d(64, 8, kernel_size=3, stride=1, padding=1, bias=False)
		init_conv(self.embeddings_conv)

	def forward(self, x):

		input_shape = x.shape[-2:]
		features = self.backbone(x)
		x = features["out"]
		x_low = features["skip1"]
		x = self.aspp(x)
		x_out = self.decoder(x, x_low)

		x_instance = self.instance_net(x_out)
		x_instance = F.interpolate(x_instance, size=input_shape, mode='bilinear', align_corners=False)
		embeddings = self.embeddings_conv(x_instance)

		x_seg = F.interpolate(x_out, size=input_shape, mode='bilinear', align_corners=False)
		seg = self.classifier(x_seg)

		result = OrderedDict()
		result["out"] = OrderedDict()
		result["out"]["seg"] = seg
		result["out"]["coords"] = embeddings

		return result


class Deeplabv3PlusInstance2(_Deeplabv3Plus):
	def __init__(self, n_classes, pretrained_model, predict, aux=False, out_planes_skip=48):
		super(Deeplabv3PlusInstance2, self).__init__(n_classes, 
											pretrained_model, 
											DeepLabDecoder1(256, out_planes=out_planes_skip), 
											predict,
											aux=aux)
		self.instance_net = nn.Sequential(
			nn.Conv2d(2, 64, kernel_size=21, stride=2, padding=10, bias=False),
			nn.GroupNorm(8, 64),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3, bias=False),
			nn.GroupNorm(8, 64),
			nn.ReLU())
		self.instance_net.apply(init_conv)

		self.embeddings_conv = nn.Conv2d(64, 16, 1, 1, 0, 1, bias=False)
		init_conv(self.embeddings_conv)

	def forward(self, x):

		input_shape = x.shape[-2:]
		features = self.backbone(x)
		x = features["out"]
		x_low = features["skip1"]
		x = self.aspp(x)
		x_out = self.decoder(x, x_low)
		x_seg = F.interpolate(x_out, size=input_shape, mode='bilinear', align_corners=False)
		seg = self.classifier(x_seg)

		instance_input = seg.transpose(0,1)[:2].transpose(0,1)
		instance_out = self.instance_net(instance_input)
		instance_out = F.interpolate(instance_out, size=input_shape, mode='bilinear', align_corners=False)
		embeddings = self.embeddings_conv(instance_out)

		result = OrderedDict()
		result["out"] = OrderedDict()
		result["out"]["seg"] = seg
		result["out"]["coords"] = embeddings

		return result


class Deeplabv3PlusInstance3(_Deeplabv3Plus):
	def __init__(self, n_classes, pretrained_model, predict, aux=False, out_planes_skip=48):
		super(Deeplabv3PlusInstance3, self).__init__(n_classes, 
											pretrained_model, 
											DeepLabDecoder1(256, out_planes=out_planes_skip), 
											predict,
											aux=aux)

		self.instance_net = nn.Sequential(
			nn.Conv2d(256, 64, kernel_size=7, stride=1, padding=3, bias=False),
			nn.GroupNorm(8, 64),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.GroupNorm(8, 64),
			nn.ReLU())

		self.embeddings_conv = nn.Conv2d(64, 16, 1, 1, 0, 1, bias=False)
		init_conv(self.embeddings_conv)

	def forward(self, x):

		input_shape = x.shape[-2:]
		features = self.backbone(x)
		x = features["out"]
		x_low = features["skip1"]
		x = self.aspp(x)
		x_out = self.decoder(x, x_low)

		x_instance = self.instance_net(x_out)
		x_instance = F.interpolate(x_instance, size=input_shape, mode='bilinear', align_corners=False)
		embeddings = self.embeddings_conv(x_instance)

		x_seg = F.interpolate(x_out, size=input_shape, mode='bilinear', align_corners=False)
		seg = self.classifier(x_seg)

		result = OrderedDict()
		result["out"] = OrderedDict()
		result["out"]["seg"] = seg
		result["out"]["coords"] = embeddings

		return result
