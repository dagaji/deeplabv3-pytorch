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


def gabor(sigma, theta, Lambda, psi, gamma, kernel_size):
    """Gabor feature extraction."""
    sigma_x = sigma
    sigma_y = float(sigma) / gamma


    y, x = np.meshgrid(np.linspace(-0.5, 0.5, kernel_size), np.linspace(-0.5, 0.5, kernel_size))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb

class GaborConv2d(_ConvNd):


	def __init__(self, angle, sigma_x=0.025, sigma_y=0.75, freq=2.0, in_channels=1, out_channels=1, kernel_size=201, stride=1, 
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


class CoopConv2d(_ConvNd):


	def __init__(self, angle, sigma_1=0.1, sigma_2=0.25, sigma_3=0.75, in_channels=1, out_channels=1, kernel_size=201, stride=1, 
		dilation=1, groups=1, bias=False, padding_mode='zeros'):

		padding = (kernel_size - 1) // 2

		kernel_size = _pair(kernel_size)
		stride = _pair(stride)
		padding = _pair(padding)
		dilation = _pair(dilation)

		super(CoopConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, _pair(0), groups, bias, padding_mode)
		self.sigma_1 = sigma_1 * torch.ones(out_channels, in_channels, requires_grad=False).to(device)
		self.sigma_2 = sigma_2 * torch.ones(out_channels, in_channels, requires_grad=False).to(device)
		self.sigma_3 = sigma_3 * torch.ones(out_channels, in_channels, requires_grad=False).to(device)

		self._weight = self._compute_weights(angle)


	def _compute_weights(self, angle):

		def _compute_g(sigma, rot):
			return torch.exp(-0.5*(rot/sigma)**2) / (sigma * np.sqrt(2*np.pi))

		_theta = (np.pi - angle) * torch.ones((1, 1)).to(device)
		y, x = torch.meshgrid([torch.linspace(-0.5, 0.5, self.kernel_size[0]), torch.linspace(-0.5, 0.5, self.kernel_size[1])])
		x = x.to(device)
		y = y.to(device)
		_weight = torch.empty(self.weight.shape, requires_grad=False).to(device)
		for i in range(self.out_channels):
			for j in range(self.in_channels):
				_sigma_1 = self.sigma_1[i, j]
				_sigma_2 = self.sigma_2[i, j]
				_sigma_3 = self.sigma_3[i, j]

				theta = _theta[i, j].expand_as(y)
				rot_3 = y * torch.cos(theta) - x * torch.sin(theta)
				rot_12 =  y * torch.sin(theta) - x * torch.cos(theta)

				g1 = _compute_g(_sigma_1, rot_12)
				g2 = _compute_g(_sigma_2, rot_12)
				g3 = _compute_g(_sigma_3, rot_3)

				_weight[i, j] = (g1 - g2) * g3

			_weight /= _weight.max() 

		return _weight

	def forward(self, _input):
		return F.conv2d(_input, self._weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

	def plot_filter(self):
		plt.figure()
		plt.imshow(self._weight.cpu().detach().numpy().squeeze())

if __name__ == "__main__":

	theta = np.deg2rad(-30.0)
	sigma = 0.075
	gamma = sigma / 0.75
	kernel_size = 201
	psi = 0

	for Lambda in np.linspace(0.15, 0.20, 5).tolist():
		gf = gabor(sigma, theta, Lambda, psi, gamma, kernel_size)
		plt.figure()
		plt.imshow(gf)
		plt.title("Lambda={}; mean={}".format(Lambda, gf.mean()))
	plt.show()








