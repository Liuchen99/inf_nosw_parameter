'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


__all__ = ['resnet12_1w8a']


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
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        
        if stride != 1:
            if in_channels != out_channels:
                self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, out_channels // 4, out_channels // 4), "constant",0))
            else:
                self.shortcut = LambdaLayer(lambda x: x[:, :, ::2, ::2])
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        # plan A
        out += self.shortcut(x)
        out = F.hardtanh(out)

        # plan B
        # out = F.hardtanh(out)
        # out += self.shortcut(x)
        return out


class ResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16                                                 
                                                                                        # [3,  224, 224], 24bit = 3612672
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)   # [16, 112, 112]
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(16, num_blocks[0], stride=1)     # out: [16, 112, 112]
        self.layer2 = self._make_layer(32, num_blocks[1], stride=2)     # out: [32, 56, 56]
        self.layer3 = self._make_layer(64, num_blocks[2], stride=2)     # out: [64, 28, 28]
        self.layer4 = self._make_layer(128, num_blocks[3], stride=2)    # out: [128, 14, 14]; stride: 16; rate:6,  rateb:36
        self.layer5 = self._make_layer(128, num_blocks[4], stride=2)    # out: [256, 7, 7];   stride: 32; rate:12, rateb:72
        self.linear = nn.Linear(128, num_classes, bias=True)

        self.apply(_weights_init)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.hardtanh(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


from utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class ResNetv3_18(ResNet):
    def __init__(self, num_classes):
        super().__init__(num_blocks=[2, 4, 6, 2, 2], num_classes=num_classes)
    # [4, 4, 4, 4, 0]

def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params = total_params + np.prod(x.data.numpy().shape)
    print("Total number of params %f M" % (total_params/1e6))
    print("Total layers", len(list(filter(
        lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))
    

if __name__ == "__main__":
    test(ResNetv3_18(1000))
    print()

