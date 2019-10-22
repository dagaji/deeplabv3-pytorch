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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GaborConv2d(_ConvNd):

    def __init__(self, angle, sigma_x=0.025, sigma_y=0.5, freq=2.0, in_channels=1, out_channels=1, kernel_size=51, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super(GaborConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, _pair(0), groups, bias, padding_mode)
        self.sigma_x = sigma_x * torch.ones(out_channels, in_channels, requires_grad=False).to(device)
        self.sigma_y = sigma_y * torch.ones(out_channels, in_channels, requires_grad=False).to(device)
        self.freq = freq * torch.ones(out_channels, in_channels, requires_grad=False).to(device)

         _theta = angle * torch.ones((1, 1)).to(device)
        y, x = torch.meshgrid([torch.linspace(-0.5, 0.5, self.kernel_size[0]), torch.linspace(-0.5, 0.5, self.kernel_size[1])])
        x = x.to(device)
        y = y.to(device)
        self._weight = torch.empty(self.weight.shape, requires_grad=False).to(device)
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
                self._weight[i, j] = g
                self.weight.data[i, j] = g

    def forward(self, input):
        return F.conv2d(input, self._weight, self.bias, self.stride, self.padding, self.dilation, self.groups)



    def plot_filter(self, angle):
        plt.figure()
        plt.imshow(self._weight.cpu().detach().numpy().squeeze())



