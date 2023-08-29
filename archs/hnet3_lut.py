import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

if __name__=="__main__":
    from binary_utils.module_v1 import *
else:
    from utils.registry import ARCH_REGISTRY
    from .binary_utils.module_v1 import *


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)
    

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        return x
    

class BasicBlock_1w4a_LUT(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.planes = planes

        self.conv1 = Conv2d_IR_LSQ_1w4a(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act = nn.ReLU(inplace=True)

    def set_next_scale(self, val):
        self.next_scale = val

    def _init_lut(self):
        assert hasattr(self, "next_scale"), print("Not set next_scale")

        # conv scales
        alpha1 = self.conv1.alpha
        alpha2 = self.next_scale

        # BatchNorm2d
        mean = self.bn1.running_mean
        var = self.bn1.running_var
        weight = self.bn1.weight
        bias = self.bn1.bias

        std = torch.sqrt(var + self.bn1.eps)
        weight = weight / std
        bias = bias - weight * mean

        self.weight = weight
        self.bias = bias

        # Generate LUT
        lut = torch.linspace(0.5, 6.5, 7).view(1, 7).repeat(self.planes, 1).to(alpha1.device) # shape: [planes, 8]
        lut = (lut * alpha2 - bias[:, None]) / (alpha1 * weight[:, None])
        lut = lut.round()

        self.lut = lut

    def _forward_conv(self, x):
        # quant weight
        w = self.conv1.weight
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)

        sw = torch.pow(torch.tensor([2] * bw.size(0)).cuda().float(),
                       (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(
            bw.size(0), 1, 1, 1).detach()

        bw = BinaryQuantize().apply(bw, self.conv1.k, self.conv1.t)
        bw = bw * sw

        return self.conv1._conv_forward(x, bw, self.conv1.bias)

    def _forward_lut(self, x):
        if not hasattr(self, "lut"):
            self._init_lut()

        for i in range(self.planes):

            # get i-th channel data
            data = x[:, i, :, :]

            # calculate flags
            # the number will be rounded to the nearest even integer
            flag_0 = (data <= self.lut[i][0])    # 0.5
            flag_1 = (data <  self.lut[i][1])    # 1.5
            flag_2 = (data <= self.lut[i][2])    # 2.5
            flag_3 = (data <  self.lut[i][3])    # 3.5
            flag_4 = (data <= self.lut[i][4])    # 4.5
            flag_5 = (data <  self.lut[i][5])    # 5.5
            flag_6 = (data <= self.lut[i][6])    # 6.5

            flag_0_not = (data >  self.lut[i][0])
            flag_1_not = (data >= self.lut[i][1])
            flag_2_not = (data >  self.lut[i][2])
            flag_3_not = (data >= self.lut[i][3])
            flag_4_not = (data >  self.lut[i][4])
            flag_5_not = (data >= self.lut[i][5])
            flag_6_not = (data >  self.lut[i][6])

            find_flag_0 = flag_0
            find_flag_1 = (flag_0_not * flag_1)
            find_flag_2 = (flag_1_not * flag_2)
            find_flag_3 = (flag_2_not * flag_3)
            find_flag_4 = (flag_3_not * flag_4)
            find_flag_5 = (flag_4_not * flag_5)
            find_flag_6 = (flag_5_not * flag_6)
            find_flag_7 = flag_6_not

            x[:, i, :, :] =  torch.where(find_flag_0, 0, x[:, i, :, :])
            x[:, i, :, :] =  torch.where(find_flag_1, 1, x[:, i, :, :])
            x[:, i, :, :] =  torch.where(find_flag_2, 2, x[:, i, :, :])
            x[:, i, :, :] =  torch.where(find_flag_3, 3, x[:, i, :, :])
            x[:, i, :, :] =  torch.where(find_flag_4, 4, x[:, i, :, :])
            x[:, i, :, :] =  torch.where(find_flag_5, 5, x[:, i, :, :])
            x[:, i, :, :] =  torch.where(find_flag_6, 6, x[:, i, :, :])
            x[:, i, :, :] =  torch.where(find_flag_7, 7, x[:, i, :, :])

        return x
    
    def forward_normal(self, x):
        # x = x * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        x = self.bn1(x)
        x = self.act(x)
        return x

    def forward(self, x):
        if not hasattr(self, "lut"):
            self._init_lut()

        # x_normal = self.conv1(x)
        # x_normal = self.forward_normal(x_normal)

        x_lut = self._forward_conv(x)
        x_lut = self._forward_lut(x_lut)


        return x_lut


class HNet_1w4a_LUT(nn.Module):
    def __init__(self, in_channels, num_classes, block, num_blocks, firstconv_channels, num_channels):
        super().__init__()
        self.num_channels = num_channels

        self.conv1 = nn.Conv2d(in_channels, firstconv_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(firstconv_channels)
        self.act1 = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, firstconv_channels, num_channels[0], num_blocks[0], stride=1) # 2 in_shape: 112x112; out_channels: 16
        self.layer2 = self._make_layer(block, num_channels[0], num_channels[1], num_blocks[1], stride=2)    # 3 in_shape: 112x112; out_channels: 24
        self.layer3 = self._make_layer(block, num_channels[1], num_channels[2], num_blocks[2], stride=2)    # 4 in_shape: 56x56;   out_channels: 40
        self.layer4 = self._make_layer(block, num_channels[2], num_channels[3], num_blocks[3], stride=1)    # 5 in_shape: 28x28;   out_channels: 72
        self.layer5 = self._make_layer(block, num_channels[3], num_channels[4], num_blocks[4], stride=1)    # 6 in_shape: 28x28;   out_channels: 108
        self.layer6 = self._make_layer(block, num_channels[4], num_channels[5], num_blocks[5], stride=2)    # 7 in_shape: 14x14;   out_channels: 172
        self.layer7 = self._make_layer(block, num_channels[5], num_channels[6], num_blocks[6], stride=2)    # 8 in_shape: 7x7;     out_channels: 236

        self.Qp = 7
        self.Qn = -8
        self.alpha = nn.Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))

        self.lut_init = False

        self.conv2 = nn.Conv2d(num_channels[6], num_channels[7], kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(num_channels[7])
        self.act2 = nn.ReLU(inplace=True)

        self.linear = nn.Linear(num_channels[7], num_classes, bias=True)
        
        self.apply(_weights_init)

    def _make_layer(self, block, in_channels, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(in_channels, channels, stride))
            in_channels = channels
        return nn.Sequential(*layers)
    
    def forward_features(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)

        # print("conv1: ", x.max().cpu(), x.min().cpu())
        x = (x / self.scales[0]).round().clamp(self.Qn, self.Qp)
        # print("conv1_quant: ", x.max().cpu(), x.min().cpu())

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        # print("layer7: ", x.max().cpu(), x.min().cpu())
        x = x * self.alpha
        # print("layer7_dequant: ", x.max().cpu(), x.min().cpu(), "\n")

        # # quant act LSQ
        # if self.training and self.init_state == 0:
        #     self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(self.Qp))
        #     self.init_state.fill_(1)

        #     print("alpha:", self.alpha.data)
        
        # g = 1.0 / math.sqrt(x.numel() * self.Qp)
        # alpha = grad_scale(self.alpha, g)
        # x = round_pass((x / alpha).clamp(self.Qn, self.Qp)) * alpha

        return x
    
    def forward(self, x):
        # check whether init the LUT
        if not self.lut_init:
            print("Init Look-UP-Table for all convs")
            self.lut_init = True

            # collect all scales
            scales = []
            for i in range(1, 8):
                layer_name = "layer%d" % (i)
                layer = getattr(self, layer_name)

                for block in layer:
                    scales.append(block.conv1.alpha.data)
            scales.append(self.alpha.data)
            self.scales = scales

            # set next_scale for each Block
            cnt = 1
            for i in range(1, 8):
                layer_name = "layer%d" % (i)
                layer = getattr(self, layer_name)

                for block in layer:
                    block.set_next_scale(scales[cnt])
                    cnt += 1

        # Normal Feed Forward
        x = self.forward_features(x)
        x = self.act2(self.norm2(self.conv2(x)))
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x
    
    def set_kt(self, k, t):
        for i in range(1,8):
            layer_name = "layer%d" % (i)
            layer = getattr(self, layer_name)
            for block in layer:
                block.conv1.k = k
                block.conv1.t = t


if __name__!="__main__":

    @ARCH_REGISTRY.register()
    class HNet3_b18_1w4a_LUT(HNet_1w4a_LUT):
        def __init__(self, num_classes):
            super().__init__(3, num_classes, BasicBlock_1w4a_LUT, [1, 2, 2, 3, 3, 4, 1], 32, [16, 24, 40, 72, 108, 172, 236, 256])

