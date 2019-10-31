# Use with pytorch version >= 1.1.0

import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
import torch.nn as nn
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import pdb
from scipy.ndimage import gaussian_filter
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GaborConv2d(_ConvNd):

	# def __init__(self, angle, sigma_x=0.075, sigma_y=0.75, freq=2.0, in_channels=1, out_channels=1, kernel_size=51, stride=1, 
	# 	dilation=1, groups=1, bias=False, padding_mode='zeros'):

	# 	padding = (kernel_size - 1) // 2

	# 	kernel_size = _pair(kernel_size)
	# 	stride = _pair(stride)
	# 	padding = _pair(padding)
	# 	dilation = _pair(dilation)

	# 	super(GaborConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, _pair(0), groups, bias, padding_mode)
	# 	self.sigma_x = sigma_x * torch.ones(out_channels, in_channels, requires_grad=False).to(device)
	# 	self.sigma_y = sigma_y * torch.ones(out_channels, in_channels, requires_grad=False).to(device)
	# 	self.freq = freq * torch.ones(out_channels, in_channels, requires_grad=False).to(device)

	# 	_weights_1 = self._compute_weights(angle)
	# 	_weights_2 = self._compute_weights(angle + np.pi/2)
	# 	self._weight = self.merge_weights(_weights_1, _weights_2)

	# 	# mask = torch.ones(joint_mask.shape).float().to(device)
	# 	# mask -= joint_mask.float()
	# 	# plt.imshow(mask.cpu().detach().numpy().squeeze())
	# 	# plt.show()
	# 	# self._weight = (_weights_1 + _weights_2) * mask
	# 	# self._weight = torch.clamp(self._weight, max=1.0)

	def __init__(self, angle, sigma_x=0.075, sigma_y=0.75, freq=2.0, in_channels=1, out_channels=1, kernel_size=51, stride=1, 
		dilation=1, groups=1, bias=False, padding_mode='zeros'):

		padding = (kernel_size - 1) // 2

		kernel_size = _pair(kernel_size)
		stride = _pair(stride)
		padding = _pair(padding)
		dilation = _pair(dilation)

		super(GaborConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, _pair(0), groups, bias, padding_mode)
		self.sigma_x = sigma_x * torch.ones(out_channels, in_channels, requires_grad=False).to(device)
		self.sigma_y = sigma_y * torch.ones(out_channels, in_channels, requires_grad=False).to(device)
		self.freq = freq * torch.ones(out_channels, in_channels, requires_grad=False).to(device)

		self._weight = self._compute_weights(angle)

		# mask = torch.ones(joint_mask.shape).float().to(device)
		# mask -= joint_mask.float()
		# plt.imshow(mask.cpu().detach().numpy().squeeze())
		# plt.show()
		# self._weight = (_weights_1 + _weights_2) * mask
		# self._weight = torch.clamp(self._weight, max=1.0)

	def merge_weights(self, _weights_1, _weights_2):
		_weights_1 = _weights_1.cpu().detach().numpy().squeeze()
		_weights_2 = _weights_2.cpu().detach().numpy().squeeze()
		# mask = ((_weights_1 > tresh) * (_weights_2 > tresh)).astype(bool)
		n = np.zeros((51,51))
		n[25,25] = 1
		k = gaussian_filter(n, sigma=4)
		k = 1.0 - 0.5 * (k / k.max())
		_weights = (_weights_1 + _weights_2) * k
		_weights = np.clip(_weights, a_min=0, a_max=1.0)
		_weights = _weights[np.newaxis, np.newaxis, :, :]
		return torch.from_numpy(_weights).float().to(device)
		# plt.figure()
		# plt.imshow(_weights)
		# plt.figure()
		# plt.imshow(k)
		# plt.figure()
		# plt.imshow(mask)
		# plt.show()


	def _compute_weights(self, angle):

		_theta = angle * torch.ones((1, 1)).to(device)
		y, x = torch.meshgrid([torch.linspace(-0.5, 0.5, self.kernel_size[0]), torch.linspace(-0.5, 0.5, self.kernel_size[1])])
		x = x.to(device)
		y = y.to(device)
		_weight = torch.empty(self.weight.shape, requires_grad=False).to(device)
		for i in range(self.out_channels):
			for j in range(self.in_channels):
				sigma_x = self.sigma_x[i, j].expand_as(y)
				sigma_y = self.sigma_y[i, j].expand_as(y)
				freq = self.freq[i, j].expand_as(y)
				theta = _theta[i, j].expand_as(y)

				rotx = x * torch.cos(theta) + y * torch.sin(theta)
				roty = -x * torch.sin(theta) + y * torch.cos(theta)

				g = torch.zeros(y.shape).to(device)

				g = torch.exp(-0.5 * (rotx ** 2 / (sigma_x + 1e-3) ** 2 + roty ** 2 / (sigma_y + 1e-3) ** 2))
				g = g * torch.cos(2 * 3.14 * freq * rotx)
				_weight[i, j] = g

		return _weight

	def forward(self, _input):
		return F.conv2d(_input, self._weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


	def plot_filter(self):
		plt.figure()
		plt.imshow(self._weight.cpu().detach().numpy().squeeze())



