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

    def __init__(self, in_channels=1, out_channels=1, kernel_size=51, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super(GaborConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, _pair(0), groups, bias, padding_mode)
        self.sigma_x = 0.025 * torch.ones(out_channels, in_channels).to(device)
        self.sigma_y = 0.5 * torch.ones(out_channels, in_channels).to(device)
        self.freq = 2 * torch.ones(out_channels, in_channels).to(device)

    def forward(self, input, angle):

        _theta = angle * torch.ones((1, 1)).to(device)
        y, x = torch.meshgrid([torch.linspace(-0.5, 0.5, self.kernel_size[0]), torch.linspace(-0.5, 0.5, self.kernel_size[1])])
        x = x.to(device)
        y = y.to(device)
        weight = torch.empty(self.weight.shape, requires_grad=False).to(device)
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
                weight[i, j] = g
                self.weight.data[i, j] = g

        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def plot_filter(self, angle):

        _theta = angle * torch.ones((1, 1)).to(device)
        y, x = torch.meshgrid([torch.linspace(-0.5, 0.5, self.kernel_size[0]), torch.linspace(-0.5, 0.5, self.kernel_size[1])])
        x = x.to(device)
        y = y.to(device)
        weight = torch.empty(self.weight.shape, requires_grad=False).to(device)
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
                weight[i, j] = g
                plt.figure()
                plt.imshow(weight.cpu().detach().numpy().squeeze())



# class GaborFilterBank(nn.Module):

#     gabor_filter = GaborConv2d()
#     tresh = nn.Parameter(torch.Tensor([100.0]).float()).to(device)

#     def __init__(self, angles):
#         super(GaborFilterBank, self).__init__()
#         self.angles = np.deg2rad(angles).tolist()
#         self.relu = nn.ReLU()

#     def forward(self, x):

#         res = []
#         for angle in self.angles:
#             res.append(self.gabor_filter(x, angle))
#         res = torch.cat(res, dim=1)

#         return self.relu(res - self.tresh)




