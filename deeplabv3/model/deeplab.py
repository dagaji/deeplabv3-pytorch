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
		# self.weight = 0.25

	def forward(self, x_frame, x_mosaic, coords):
		batch_size = x_frame.shape[0]
		coords = F.interpolate(coords.transpose(3,1).transpose(3,2), size=x_frame.shape[-2:], mode='nearest')
		coords = coords.transpose(3,1).transpose(1,2)
		# offset = self.offset_net(coords)
		offset = 0.0

		x_frame_sampled = F.grid_sample(x_mosaic, coords + offset)
		weights_sigmoid = torch.sigmoid(self.weights).repeat(batch_size, 1).view(batch_size, 2048, 1, 1)
		x_fuse = (1 - weights_sigmoid) * x_frame + weights_sigmoid * x_frame_sampled
		#x_fuse = (1- self.weight) * x_frame + self.weight * x_frame_sampled

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
		scale_factor = input_shape[0] / x_frame.shape[-2]
		x_mosaic = F.interpolate(x_mosaic, scale_factor=scale_factor, mode='bilinear', align_corners=False)
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
			return self.predict(x, inputs)

		


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
			return self.predict(x)




class Deeplabv3Plus2(_Deeplabv3Plus):
	def __init__(self, n_classes, pretrained_model, predict):
		super(Deeplabv3Plus2, self).__init__(n_classes, 
											pretrained_model, 
											DeepLabDecoder2(256, 512), 
											predict)

	def forward(self, x):

		input_shape = x.shape[-2:]
		features = self.backbone(x)
		x = features["out"]
		x_low1 = features["skip1"]
		x_low2 = features["skip2"]
		x = self.aspp(x)
		x = self.decoder(x, x_low1, x_low2)
		x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
		x = self.classifier(x)
		
		if self.training:
			result = OrderedDict()
			result["out"] = x
			return result
		else:
			return self.predict(x)


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

