import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .binary_function import STE, BinaryQuantize

##### STE
class Conv2d_STE_1w4a(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d_STE_1w4a, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        # quant weight
        bw = STE().apply(self.weight, 1)

        # quant act
        x = STE().apply(x, 3)

        return self._conv_forward(x, bw, self.bias)
    

class BatchNorm2d_STE_1w4a(nn.BatchNorm2d):
    def __init__(self, num_features, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
        super(BatchNorm2d_STE_1w4a, self).__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)
        assert affine == True, Exception("Only support affine=True")

        self.qw = 3
        self.qb = 12
        self.qa = 10

        self.Qw = 2 ** self.qw
        self.Qb = 2 ** self.qb
        self.Qa = 2 ** self.qa

    def forward(self, x):
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats: 
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1             
                if self.momentum is None:     
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:      
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = x.mean([0, 2, 3])
            var = x.var([0, 2, 3], unbiased=False)
            n = x.numel() / x.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean + (1 - exponential_average_factor) * self.running_mean
                self.running_var = exponential_average_factor * var * n / (n - 1) + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        std = torch.sqrt(var + self.eps)
        weight = self.weight / std
        bias = self.bias - weight * mean

        # quant weight
        Tw_max = weight.max().detach().abs()
        Tw_min = weight.min().detach().abs()
        Tw = torch.max(Tw_max, Tw_min)
        weight = weight / Tw * self.Qw
        weight = STE().apply(weight, 3)
        weight = weight * Tw / self.Qw

        # quant bias v1
        # Tb_max = bias.max().detach().abs()
        # Tb_min = bias.min().detach().abs()
        # Tb = torch.max(Tb_max, Tb_min)
        # bias = bias / Tb * self.Qb
        # bias = STE().apply(bias, 12)
        # bias = bias * Tb / self.Qb

        # quant bias v2
        bias = bias / Tw * self.Qw
        bias = STE().apply(bias, 12)
        bias = bias * Tw / self.Qw

        # print(x.max(), x.min(), bias.max(), bias.min())

        # quant act
        # x = STE().apply(x, 10)  # conv's output, should not be scaled

        x = x * weight[None, :, None, None] + bias[None, :, None, None]
        
        return x


##### LSQ   
##### https://github.com/hustzxd/EfficientPyTorch/blob/clean/models/_modules/lsq.py#L49
def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad


class Conv2d_LSQ_1w4a(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d_LSQ_1w4a, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.nbits = 4 
        self.Qp = 2 ** (self.nbits - 1) - 1
        self.Qn = -2 ** (self.nbits - 1)

        self.alpha = nn.Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))
        
    def forward(self, x):
        # quant weight
        bw = STE().apply(self.weight, 1)

        # quant act LSQ
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(self.Qp))
            self.init_state.fill_(1)

            print("alpha:", self.alpha.data)

        g = 1.0 / math.sqrt(x.numel() * self.Qp)
        alpha = grad_scale(self.alpha, g)

        x =  round_pass((x / alpha).clamp(self.Qn, self.Qp)) * alpha

        return self._conv_forward(x, bw, self.bias)


class Conv2d_IR_LSQ_1w4a(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d_IR_LSQ_1w4a, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        # IR
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()
        self.bw = 0

        # LSQ
        self.nbits = 4 
        self.Qp = 2 ** (self.nbits - 1) - 1
        self.Qn = -2 ** (self.nbits - 1)

        self.alpha = nn.Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))
        
    def forward(self, x):
        # quant weight
        w = self.weight
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)

        sw = torch.pow(torch.tensor([2] * bw.size(0)).cuda().float(),
                       (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(
            bw.size(0), 1, 1, 1).detach()

        bw = BinaryQuantize().apply(bw, self.k, self.t)
        bw = bw * sw
        print(sw)
        # quant act LSQ
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(self.Qp))
            self.init_state.fill_(1)

            print("alpha:", self.alpha.data)

        g = 1.0 / math.sqrt(x.numel() * self.Qp)
        alpha = grad_scale(self.alpha, g)

        x =  round_pass((x / alpha).clamp(self.Qn, self.Qp)) * alpha

        return self._conv_forward(x, bw, self.bias)
    

class Conv2d_sigmoid_IR_LSQ_1w4a(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d_sigmoid_IR_LSQ_1w4a, self).__init__(in_channels, out_channels,
                                                 kernel_size, stride, padding, dilation, groups, bias)

        # IR
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()
        self.bw = 0

        # LSQ
        self.nbits = 4
        self.Qp = 2 ** (self.nbits - 1) - 1
        self.Qn = -2 ** (self.nbits - 1)

        self.alpha = nn.Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))

    def forward(self, x):
        # quant weight
        w = self.weight
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)

        sw = torch.pow(torch.tensor([2] * bw.size(0)).cuda().float(),
                       (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(
            bw.size(0), 1, 1, 1).detach()

        bw = BinaryQuantize().apply(bw, self.k, self.t)
        bw = bw * sw

        # quant act LSQ
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(self.Qp))
            self.init_state.fill_(1)

            print("alpha:", self.alpha.data)

        g = 1.0 / math.sqrt(x.numel() * self.Qp)
        alpha = grad_scale(self.alpha, g)

        x = x / alpha
        x = F.sigmoid(x)*8

        x = round_pass((x).clamp(self.Qn, self.Qp)) * alpha

        return self._conv_forward(x, bw, self.bias)
