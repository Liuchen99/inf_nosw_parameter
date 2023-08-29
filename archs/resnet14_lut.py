import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .binary_utils.module_v1 import *
from utils.registry import ARCH_REGISTRY

import numpy as np
from sim.para_save import intToBin, diff_base_save, lut_all_save

def _weights_init(m):
    classname = m.__class__.__name__
    print(classname)
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
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class BasicBlock_1w4a(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_1w4a, self).__init__()

        self.conv1 = Conv2d_IR_LSQ_1w4a(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d_IR_LSQ_1w4a(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class BasicBlock_1w4a_LUT(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, idx=0):
        super(BasicBlock_1w4a_LUT, self).__init__()
        self.planes = planes

        self.conv1 = Conv2d_IR_LSQ_1w4a(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d_IR_LSQ_1w4a(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.layer_idx = 0
        self.idx = idx
        self.count = 0

    def set_next_scale(self, val):
        self.next_scale = val

    def _init_lut(self, idx):
        if idx == 1:
            bn = self.bn1
            alpha1 = self.conv1.alpha
            alpha2 = self.conv2.alpha
        elif idx == 2:
            bn = self.bn2
            alpha1 = self.conv2.alpha
            alpha2 = self.next_scale

        # BatchNorm2d
        mean = bn.running_mean
        var = bn.running_var
        weight = bn.weight
        bias = bn.bias

        std = torch.sqrt(var + bn.eps)
        weight = weight / std
        bias = bias - weight * mean

        # Generate LUT. This only support 4a and ReLU.
        lut = torch.linspace(0.5, 6.5, 7).view(1, 7).repeat(self.planes, 1).to(alpha1.device)  # shape: [planes, 8]
        lut = (lut * alpha2 - bias[:, None]) / (alpha1 * weight[:, None])
        lut = lut.round()

        if idx == 1:
            self.lut1 = lut
        else:
            self.lut2 = lut

    def gen_weight(self, weight, name):

        for channel_in in range(weight.size(1) // 16):
            for channel_out in range(weight.size(0) // 16):
                resu = []
                count = 0
                str_data = ''
                for j in range(channel_in * 16, (channel_in + 1) * 16):
                    for row in range(3):
                        for col in range(3):
                            for i in range(channel_out * 16, (channel_out + 1) * 16):
                                str_data = str(int(weight[i, j, row, col])) + str_data
                                count = count + 1
                                if count == 8:
                                    resu.append('0x' + hex(int(str_data, 2))[2:].rjust(2, '0'))
                                    str_data = ''
                                    count = 0

                file_path = 'sim/mid_data/' + name + "weight_in" + str((channel_in + 1) * 16) + "_out" + str(
                    (channel_out + 1) * 16) + ".coe"
                with open(file_path, mode='w', encoding='utf-8') as file_obj:
                    for i in resu:
                        file_obj.write(i + ',\n')

    def gen_feature(self, input, name):
        resu = []
        for i in range(input[0].size(2)):
            for j in range(input[0].size(3)):
                str = ''
                for c in range(input[0].size(1)):
                    val = input[0][0, c, i, j]
                    val_int = (int(val)) & 0xF
                    val_hex = hex(val_int)
                    val_hex = val_hex[2:].rjust(1, '0')
                    str = val_hex + str
                resu.append('0x' + str + ',')

        file_path = 'sim/mid_data/' + name + ".coe"
        with open(file_path, mode='w', encoding='utf-8') as file_obj:
            file_obj.write('{')
            for i in resu:
                file_obj.write(i + '\n')
            file_obj.write('},\n')

    def _forward_conv(self, x, conv):
        w = conv.weight
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)

        sw = torch.pow(torch.tensor([2] * bw.size(0)).cuda().float(),
                       (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(
            bw.size(0), 1, 1, 1).detach()

        bw = BinaryQuantize().apply(bw, conv.k, conv.t)

        # bw = bw * sw
        weight = bw.clone()
        weight[weight == -1] = torch.tensor(0)

        # self.gen_weight(weight, '')

        return conv._conv_forward(x, bw, self.conv1.bias)

    def _forward_lut(self, x, lut):
        self.count += 1
        self.layer_idx =  (self.planes // 32 + 1) * 4 + self.idx * 2 + self.count - 7
        print(self.layer_idx)
        lut_base = torch.zeros(64)
        lut_diff = torch.zeros(64)

        for i in range(self.planes):

            # get i-th channel data
            # print(x.shape)  # 100,16,32,32
            data = x[:, i, :, :]

            # calculate flags
            # the number is rounded to the nearest even integer
            diff = lut[i, 1] - lut[i, 0]
            for j in range(1, 7):
                lut[i, j] = lut[i, j-1] + diff
            # print('channel:', str(i), 'diff is ', diff, 'base is ', lut[i, 0])
            if self.layer_idx == 1:
                lut_base[i+16] = lut[i, 0]
                lut_diff[i+16] = diff
            elif self.layer_idx == 2 or self.layer_idx == 6:
                lut_base[i+32] = lut[i, 0]
                lut_diff[i+32] = diff
            elif self.layer_idx == 3:
                lut_base[i + 48] = lut[i, 0]
                lut_diff[i + 48] = diff
            else:
                lut_base[i] = lut[i, 0]
                lut_diff[i] = diff
            print(self.layer_idx, lut[i], diff)
            # print('  // channel ', str(i), '    ', lut[i])
            # for j in range(0, 7):
            #     if lut[i][j] < 0:
            #         print('  hard_lut_', str(i) + '[' + str(j) + ']', ' <= -8\'d', '{:.0f}'.format(lut[i, j].abs()),
            #               ';', sep="")
            #     elif lut[i, j] == 0:
            #         print('  hard_lut_', str(i) + '[' + str(j) + ']', ' <= 8\'d0;', sep="")
            #     else:
            #         print('  hard_lut_', str(i), '[', str(j), ']', ' <= 8\'d' '{:.0f}'.format(lut[i, j]), ';', sep="")
            flag_0 = (data <= lut[i][0])  # 0.5
            flag_1 = (data < lut[i][1])  # 1.5
            flag_2 = (data <= lut[i][2])  # 2.5
            flag_3 = (data < lut[i][3])  # 3.5
            flag_4 = (data <= lut[i][4])  # 4.5
            flag_5 = (data < lut[i][5])  # 5.5
            flag_6 = (data <= lut[i][6])  # 6.5

            flag_0_not = (data > lut[i][0])
            flag_1_not = (data >= lut[i][1])
            flag_2_not = (data > lut[i][2])
            flag_3_not = (data >= lut[i][3])
            flag_4_not = (data > lut[i][4])
            flag_5_not = (data >= lut[i][5])
            flag_6_not = (data > lut[i][6])

            find_flag_0 = flag_0
            find_flag_1 = (flag_0_not * flag_1)
            find_flag_2 = (flag_1_not * flag_2)
            find_flag_3 = (flag_2_not * flag_3)
            find_flag_4 = (flag_3_not * flag_4)
            find_flag_5 = (flag_4_not * flag_5)
            find_flag_6 = (flag_5_not * flag_6)
            find_flag_7 = flag_6_not

            x[:, i, :, :] = torch.where(find_flag_0, torch.tensor(0, dtype=torch.float32).cuda(), x[:, i, :, :])
            x[:, i, :, :] = torch.where(find_flag_1, torch.tensor(1, dtype=torch.float32).cuda(), x[:, i, :, :])
            x[:, i, :, :] = torch.where(find_flag_2, torch.tensor(2, dtype=torch.float32).cuda(), x[:, i, :, :])
            x[:, i, :, :] = torch.where(find_flag_3, torch.tensor(3, dtype=torch.float32).cuda(), x[:, i, :, :])
            x[:, i, :, :] = torch.where(find_flag_4, torch.tensor(4, dtype=torch.float32).cuda(), x[:, i, :, :])
            x[:, i, :, :] = torch.where(find_flag_5, torch.tensor(5, dtype=torch.float32).cuda(), x[:, i, :, :])
            x[:, i, :, :] = torch.where(find_flag_6, torch.tensor(6, dtype=torch.float32).cuda(), x[:, i, :, :])
            x[:, i, :, :] = torch.where(find_flag_7, torch.tensor(7, dtype=torch.float32).cuda(), x[:, i, :, :])

        # self.gen_feature([x], 'output')
        # diff_base_save(lut_base, lut_diff, self.layer_idx)
        lut_all_save()
        # exit()
        return x

    def forward(self, x):
        if not hasattr(self, "lut1"):
            self._init_lut(1)
            self._init_lut(2)

        x = self._forward_conv(x, self.conv1)
        x = self._forward_lut(x, self.lut1)
        x = self._forward_conv(x, self.conv2)
        x = self._forward_lut(x, self.lut2)
        return x


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, layer_idx=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, layer_idx=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, layer_idx=3)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, layer_idx):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class ResNet_1w4a(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_1w4a, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, layer_idx=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, layer_idx=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, layer_idx=3)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

        self.Qp = 7
        self.Qn = -8
        self.alpha = nn.Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))

    def _make_layer(self, block, planes, num_blocks, stride, layer_idx):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def set_kt(self, k, t):
        for i in range(1, 4):
            layer_name = "layer%d" % (i)
            layer = getattr(self, layer_name)
            for block in layer:
                block.conv1.k = k
                block.conv1.t = t
                block.conv2.k = k
                block.conv2.t = t

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # quant act LSQ
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(self.Qp))
            self.init_state.fill_(1)
            print("alpha:", self.alpha.data)

        g = 1.0 / math.sqrt(x.numel() * self.Qp)
        alpha = grad_scale(self.alpha, g)
        x = round_pass((x / alpha).clamp(self.Qn, self.Qp)) * alpha

        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class ResNet_1w4a_LUT(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_1w4a_LUT, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

        self.Qp = 7
        self.Qn = -8
        self.alpha = nn.Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))
        self.lut_init = False

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        idx = 0
        for stride in strides:
            idx = idx + 1
            layers.append(block(self.in_planes, planes, stride, idx))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def gen_feature(self, input, name):
        resu = []
        for i in range(input[0].size(2)):
            for j in range(input[0].size(3)):
                str = ''
                for c in range(input[0].size(1)):
                    val = input[0][0, c, i, j]
                    val_int = (int(val)) & 0xF
                    val_hex = hex(val_int)
                    val_hex = val_hex[2:].rjust(1, '0')
                    str = val_hex + str
                resu.append('0x' + str + ',')

        file_path = 'sim/mid_data/' + name + "_input.coe"
        with open(file_path, mode='w', encoding='utf-8') as file_obj:
            file_obj.write('{')
            for i in resu:
                file_obj.write(i + '\n')
            file_obj.write('},\n')

    def forward(self, x):
        if not self.lut_init:
            print("\nInit Look-UP-Table for all convs")
            self.lut_init = True

            # collect all scales
            scales = []
            for i in range(1, 4):
                layer_name = "layer%d" % (i)
                layer = getattr(self, layer_name)

                for block in layer:
                    scales.append(block.conv1.alpha.data)
            scales.append(self.alpha.data)
            self.scales = scales

            # set next_scale for each Block
            cnt = 1
            for i in range(1, 4):
                layer_name = "layer%d" % (i)
                layer = getattr(self, layer_name)

                for block in layer:
                    block.set_next_scale(scales[cnt])
                    cnt += 1

        # conv1 FP32
        x = F.relu(self.bn1(self.conv1(x)))  # FP32

        # FP32 -> 4a
        x = (x / self.scales[0]).round().clamp(self.Qn, self.Qp)

        self.gen_feature([x], 'input')
        # layers 1w4a 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # 4a -> FP32
        x = x * self.alpha

        # linear FP32
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        exit()
        return x


@ARCH_REGISTRY.register()
class ResNet14_new(ResNet):
    def __init__(self, num_classes):
        super(ResNet14_new, self).__init__(BasicBlock, [2, 2, 2], num_classes)


@ARCH_REGISTRY.register()
class ResNet14_new_1w4a(ResNet_1w4a):
    def __init__(self, num_classes):
        super(ResNet14_new_1w4a, self).__init__(BasicBlock_1w4a, [2, 2, 2], num_classes)


@ARCH_REGISTRY.register()
class ResNet14_new_1w4a_LUT(ResNet_1w4a_LUT):
    def __init__(self, num_classes):
        super(ResNet14_new_1w4a_LUT, self).__init__(BasicBlock_1w4a_LUT, [2, 2, 2], num_classes)
