import torch
import torch.nn as nn
import numpy as np

import pdb
from torch.nn import functional as F
from collections import OrderedDict
import matplotlib.pyplot as plt
import deeplabv3.lines as lines
from deeplabv3.save import ResultsLogger
from deeplabv3.model.deeplab import _Deeplabv3Plus, DeepLabDecoder1

def gabor(theta, sigma_x=0.075, sigma_y=0.75, Lambda=0.2, psi=0.0, kernel_size=51):

    y, x = np.meshgrid(np.linspace(-0.5, 0.5, kernel_size), np.linspace(-0.5, 0.5, kernel_size))

    # Rotation
    x_theta = x * np.cos(theta) - y * np.sin(theta)
    y_theta = x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    
    return gb.astype(np.float32)

class GaborBank(nn.Module):
	def __init__(self, thetas, kernel_size=51, Lambda=0.2):
		super(GaborBank, self).__init__()
		self.thetas = torch.Tensor(thetas).float()
		self.kernel_size = kernel_size
		self.Lambda = Lambda
		self.alfa = nn.Parameter(0.1 * torch.ones(1))
		self.sigma_y = 0.75

	def get_device(self,):
		return self.alfa.device

	def plot_filters(self,):
		print(">>>>>> sigma_x: {}".format(self.alfa * self.sigma_y))
		print(">>>>>> sigma_y: {}".format(self.sigma_y))
		# gabor_filters = self.compute_weigths()
		# for _filter in gabor_filters[:int(len(self.thetas)/2)]:
		# 	plt.figure()
		# 	plt.imshow(_filter.cpu().detach().numpy().squeeze())
		# plt.show()


	def compute_weigths(self,):

		y, x = torch.meshgrid([torch.linspace(-0.5, 0.5, self.kernel_size), torch.linspace(-0.5, 0.5, self.kernel_size)])
		x = x.to(self.get_device())
		y = y.to(self.get_device())
		gabor_filters = []
		for _theta in self.thetas.to(self.get_device()):
			rotx = x * torch.cos(_theta) - y * torch.sin(_theta)
			roty = x * torch.sin(_theta) + y * torch.cos(_theta)
			gf = torch.exp(-0.5 * (rotx ** 2 / (self.sigma_y * self.alfa + 1e-3) ** 2 + roty ** 2 / (self.sigma_y + 1e-3) ** 2)) * torch.cos(2 * 3.14 * rotx / self.Lambda) 
			gabor_filters.append(gf.unsqueeze(0))
		gabor_filters = torch.cat(gabor_filters, 0).unsqueeze(1)

		return gabor_filters


	def forward(self, _input):
		# print(">>>>>> sigma_x: {}".format(self.sigma_x))
		gabor_filters = self.compute_weigths()
		return F.conv2d(_input, gabor_filters)




class AngleDetect(_Deeplabv3Plus):
	def __init__(self, n_classes, pretrained_model, predict,
		aux=False, out_planes_skip=48, angle_step=15.0, min_angle=-30.0, max_angle=30.0, kernel_size=51, train_gabor=False):
		
		super(AngleDetect, self).__init__(n_classes, 
											pretrained_model, 
											DeepLabDecoder1(256, out_planes=out_planes_skip), 
											predict,
											aux=aux)
		self.reduce = nn.Sequential(
			nn.Conv2d(256, 1, kernel_size=1, stride=1, bias=False))
		self.relu = nn.ReLU()

		angles1 = np.deg2rad(np.arange(min_angle, max_angle + angle_step, angle_step))
		self.angles_v = np.rad2deg(angles1)
		angles2 = angles1 + np.pi/2
		angles = np.array(angles1.tolist() + angles2.tolist())
		self.num_angles = len(angles1)

		self.train_gabor = train_gabor
		if train_gabor:
			self.gabor_bank = GaborBank(angles)
		else:
			self.gabor_bank = nn.Conv2d(1, 2 * self.num_angles, kernel_size=kernel_size, stride=1, bias=False)
			filter_weights = []
			for angle in (np.pi/2 - angles).tolist():
				filter_weights.append(gabor(angle, kernel_size=kernel_size))
			filter_weights = np.dstack(filter_weights)
			filter_weights = np.transpose(filter_weights, (2,0,1))
			filter_weights = torch.Tensor(filter_weights).view_as(self.gabor_bank.weight.data)
			self.gabor_bank.weight = nn.Parameter(filter_weights)
			self.gabor_bank.requires_grad=False

		# self.plot_gabor()

		# self.results_logger = ResultsLogger('reunion/results_25_25')

	def plot_gabor(self, indices=None):

		gabor_filters = self.gabor_bank.weight.data[:self.num_angles]
		gabor_angles = self.angles_v.copy()
		if indices is not None:
			gabor_filters = gabor_filters[indices]
			gabor_angles = gabor_angles[indices]

		for gf, theta in zip(gabor_filters, gabor_angles):
			gf = gf.squeeze().cpu().numpy()
			plt.figure()
			plt.imshow(gf)
			plt.title("Theta={}".format(theta))
		plt.show()

	def plot_gabor_response(self, res):

		for _res in res.squeeze(0)[:self.num_angles]:
			_res = _res.cpu().detach().numpy().squeeze()
			plt.figure()
			plt.imshow(_res)
		plt.show()


		
	def load_state_dict(self, state_dict, strict=True):
		super(AngleDetect, self).load_state_dict(state_dict, strict)
		if 'reduce.weight' not in state_dict:
			w0, w1 = self.classifier.weight.data[:2]
			self.reduce[0].weight = nn.Parameter((w1 - w0).unsqueeze(0))

	def get_device(self,):
		return self.classifier.weight.device

	def compute_angle_range(self, x):

		bs = x.shape[0]
		x = self.reduce(x)
		x = self.gabor_bank(x)
		x = self.relu(x)
		x = x.transpose(0,1)
		x1 = x[:self.num_angles].transpose(0,1)
		x2 = x[self.num_angles:].transpose(0,1)
		x = x1 + x2
		x = x.view(bs, self.num_angles, -1).sum(2)
		x = x.transpose(0,1)
		x = (x[:-1] + x[1:]).transpose(0,1)
		return x / x.sum(1)

	def trainable_parameters(self,):
		params = list(self.reduce.parameters())
		# if self.train_gabor:
		# 	params += list(self.gabor_bank.parameters())
		# params += list(super(AngleDetect, self).parameters())
		return params

	def forward(self, inputs):

		x = inputs['image'].to(self.get_device())
		input_shape = x.shape[-2:]
		features = self.backbone(x)
		x = features["out"]
		x_low = features["skip1"]
		x = self.aspp(x)
		x = self.decoder(x, x_low)
		angle_ranges = self.compute_angle_range(x)

		result = OrderedDict()
		result["out"] = OrderedDict()
		result["out"]["angle_ranges"] = angle_ranges
		return result

