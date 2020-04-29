import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pylab as plt
from math import isnan, sqrt
import torch.nn as nn
import os
import time
import torch
from torch.autograd import Variable
from torch.optim import Adam
from skimage import io
# from sklearn.metrics import mean_squared_error
# from helper_functions import *

################################################# network class #################################################
class dncnn(nn.Module):

    def __init__(self, all_params):
        super(dncnn, self).__init__()

        self.padding_mid = all_params['padding_mid']
        self.num_mid_kernels = all_params['num_mid_kernels']
        self.kernel_size_mid = all_params['kernel_size_mid']
        self.num_mid_layers = all_params['num_mid_layers']

        #### initialize the layers
        mid_layers = nn.ModuleList([])
        mid_layers.append(nn.Conv2d(1,self.num_mid_kernels, self.kernel_size_mid, padding=self.padding_mid , bias=True))
        mid_layers.append(nn.ReLU(inplace=True))
        for l in range(1,self.num_mid_layers-1):
            mid_layers.append(nn.Conv2d(self.num_mid_kernels ,self.num_mid_kernels, self.kernel_size_mid, padding=self.padding_mid , bias=True))
            mid_layers.append(nn.BatchNorm2d(self.num_mid_kernels) )
            mid_layers.append(nn.ReLU(inplace=True))

        mid_layers.append(nn.Conv2d(self.num_mid_kernels,1, self.kernel_size_mid, padding=self.padding_mid , bias=True))
        self.model = nn.Sequential(*mid_layers)



    def forward(self, x):

        ## pass through the denoising function/network

        out = self.model(x)
        return out

