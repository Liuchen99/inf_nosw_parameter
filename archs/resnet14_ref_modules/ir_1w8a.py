import torch.nn as nn
from typing import Optional, Any

from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init
# from torch.nn.modules._functions import SyncBatchNorm as sync_batch_norm
# from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.modules.module import Module
from . import binaryfunction
import torch
from torch import Tensor
import math
# from quantizeInt8 import activation_quantize_fn
from torch.nn.parameter import Parameter


class IRConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(IRConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()
        self.bw = 0

    def forward(self, input):
        w = self.weight
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)

        sw = torch.pow(torch.tensor([2] * bw.size(0)).cuda().float(),
                       (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(
            bw.size(0), 1, 1, 1).detach()

        bw = binaryfunction.BinaryQuantize().apply(bw, self.k, self.t)
        # bw = bw * sw
        output = F.conv2d(input, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)

        return output


class IRlinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super(IRlinear, self).__init__(in_features, out_features, bias=False)
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()

    def forward(self, input, Tmax=[1.0, 0.0]):
        w = self.weight
        x = input
        Tmax = torch.max(input).detach()  # Tmax[0]  # torch.max(input).detach()#
        Tmin = torch.min(input).detach()  # Tmax[1]  # torch.min(input).detach()#
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1)
        sw = torch.pow(torch.tensor([2] * bw.size(0)).cuda().float(),
                       (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(
            bw.size(0), 1).detach()

        bw = binaryfunction.BinaryQuantize().apply(bw, self.k, self.t)
        bw = bw * sw

        T = torch.max(torch.abs(Tmin), torch.abs(Tmax))
        T = torch.clamp(T, 1e-10, 255.)
        x = torch.clamp(x, 0 - T, T)
        x_s = x / T
        activation_q = binaryfunction.qfn().apply(x_s, 4)
        activation_q = activation_q * T
        output = F.linear(activation_q, bw, self.bias)
        return output
