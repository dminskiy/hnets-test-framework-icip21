# This code is based on the implementation of InvariantLayerj1
# Most of the functions are copied directly
# Only the numbers were adapted to work with Scattering2D

from kymatio.torch import Scattering2D
from .helper_funcs import calculate_channels_after_scat

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as init
import numpy as np


def random_postconv_impulse(C, F):
    """ Creates a random filter with +/- 1 in one location for a
    3x3 convolution. The idea being that we can randomly offset filters from
    each other_readers"""
    z = torch.zeros((F, C, 3, 3))
    x = np.random.randint(-1, 2, size=(F, C))
    y = np.random.randint(-1, 2, size=(F, C))
    for i in range(F):
        for j in range(C):
            z[i, j, y[i,j], x[i,j]] = 1
    return z


def random_postconv_smooth(C, F, σ=1):
    """ Creates a random filter by shifting a gaussian with std σ. Meant to
    be a smoother version of random_postconv_impulse."""
    x = np.arange(-2, 3)
    a = 1/np.sqrt(2*np.pi*σ**2) * np.exp(-x**2/σ**2)
    b = np.outer(a, a)
    z = np.zeros((F, C, 3, 3))
    x = np.random.randint(-1, 2, size=(F, C))
    y = np.random.randint(-1, 2, size=(F, C))
    for i in range(F):
        for j in range(C):
            z[i, j] = np.roll(b, (y[i,j], x[i,j]), axis=(0,1))[1:-1,1:-1]
        z[i] /= np.sqrt(np.sum(z[i]**2))
    return torch.tensor(z, dtype=torch.float32)

class Scattering2DMixed(nn.Module):
    """ Also can be called the learnable scatternet layer.

    Takes a single order scatternet layer, and mixes the outputs to give a new
    set of outputs. You can select the style of mixing, the default being a
    single 1x1 convolutional layer, but other_readers options include a 3x3
    convolutional mixing and a 1x1 mixing with random offsets.

    Inputs:
        C (int): The number of input channels
        F (int): The number of output channels. None by default, in which case
            the number of output channels is 7*C.
        stride (int): The downsampling factor
        k (int): The mixing kernel size
        alpha (str): A fixed kernel to increase the spatial size of the mixing.
            Can be::

                - None (no expansion),
                - 'impulse' (randomly shifts bands left/right and up/down by 1
                    pixel),
                - 'smooth' (randomly shifts a gaussian left/right and up/down
                    by 1 pixel and uses the mixing matrix to expand this.

    Returns:
        y (torch.tensor): The output

    """
    def __init__(self, in_channels, J, shape, max_order, L=8, k=1, alpha = None):
        super().__init__()

        self.scatNet = Scattering2D(J=J, shape=shape, max_order=max_order, L=L)

        channels_after_scat = calculate_channels_after_scat(in_channels=in_channels, order=max_order, J=J, L=L)

        if k > 1 and alpha is not None:
            raise ValueError("Only use alpha when k=1")

        # Create the learned mixing weights and possibly the expansion kernel
        self.A = nn.Parameter(torch.randn(channels_after_scat, channels_after_scat, k, k))
        self.b = nn.Parameter(torch.zeros(channels_after_scat,))
        if alpha == 'impulse':
            self.alpha = nn.Parameter(
                random_postconv_impulse(channels_after_scat, channels_after_scat), requires_grad=False)
            self.pad = 1
        elif alpha == 'smooth':
            self.alpha = nn.Parameter(
                random_postconv_smooth(channels_after_scat, channels_after_scat, σ=1), requires_grad=False)
            self.pad = 1
        elif alpha == 'random':
            self.alpha = nn.Parameter(
                torch.randn(channels_after_scat, channels_after_scat, 3, 3), requires_grad=False)
            init.xavier_uniform(self.alpha)
            self.pad = 1
        elif alpha is None:
            self.alpha = 1
            self.pad = (k-1) // 2
        else:
            raise ValueError

    def forward(self, x):
        z = self.scatNet(x)
        z = z.view(z.shape[0], -1, z.shape[z.dim() - 2], z.shape[z.dim() - 1])
        As = self.A * self.alpha
        y = func.conv2d(z, As, self.b, padding=self.pad)
        y = func.relu(y)
        return y

    def init(self, gain=1, method='xavier_uniform'):
        if method == 'xavier_uniform':
            init.xavier_uniform_(self.A, gain=gain)
        else:
            init.xavier_normal_(self.A, gain=gain)