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
from .deeplab import _Deeplabv3Plus, DeepLabDecoder1, load_pretrained_model, init_conv

output_stride_params = { 16: dict(replace_stride_with_dilation=[False, False, True], rates=[6, 12, 18]),
						 8:  dict(replace_stride_with_dilation=[False, True, True],  rates=[12, 24, 36]),
						 4 : dict(replace_stride_with_dilation=[True, True, True],   rates=[24, 48, 72]),
						}


class DeepLabv3Plus(_Deeplabv3Plus):
	def __init__(self, n_classes, pretrained_model, out_planes_skip=48):
		super(DeepLabv3Plus, self).__init__(n_classes, 
											pretrained_model, 
											DeepLabDecoder1(256, out_planes=out_planes_skip), 
											None,
											aux=False)

	def forward(self, x):
		input_shape = x.shape[-2:]
		features = self.backbone(x)
		x = features["out"]
		x_low = features["skip1"]
		x = self.aspp(x)
		x = self.decoder(x, x_low)
		x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
		x = self.classifier(x).transpose(0,1)[1]
		return x

class LineDetectionNet(nn.Module):
	def __init__(self, in_planes=256):
		super(LineDetectionNet, self).__init__()
		self.clf = nn.Linear(in_planes, 1)
		init_conv(self.clf)

	def forward(self, features, grid):
		features_sample = torch.squeeze(F.grid_sample(features, grid), dim=3)
		features_mean = features_sample.mean(2)
		score = self.clf(features_mean)
		return score

def get_seg_model():

	return_layers = dict(layer4='out', layer1='skip1', layer3='aux')
	kw_backbone_args = dict(output_stride_params[8])
	kw_backbone_args.update(return_layers=return_layers)
	pretrained_model = load_pretrained_model(kw_backbone_args)
	seg_model = DeepLabv3Plus(4, pretrained_model, out_planes_skip=48)
	return seg_model

