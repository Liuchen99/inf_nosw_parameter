import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from utils.registry import ARCH_REGISTRY


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
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.hardtanh(out)

        x1 = out
        out = self.conv2(x1)
        out = self.bn2(out)
        out = F.hardtanh(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, in_planes=64):
        super(ResNet, self).__init__()

        # in:  224*224*3 = 150528
        # out: 14*14*128 = 25088  (1/6)*(1/6)
        # out: 14*14*512 = 100325 (2/3)*(1/6)

        self.in_planes = in_planes

        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=2, padding=1, bias=False)            # out: 112x112
        self.bn1 = nn.BatchNorm2d(in_planes)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=2, padding=1, bias=False)     # out: 56x56
        self.bn2 = nn.BatchNorm2d(in_planes)

        self.layer1 = self._make_layer(block, in_planes * 1, num_blocks[0], stride=1, layer_idx=1)      # out: 56x56
        self.layer2 = self._make_layer(block, in_planes * 2, num_blocks[1], stride=2, layer_idx=2)      # out: 28x28
        self.layer3 = self._make_layer(block, in_planes * 4, num_blocks[2], stride=2, layer_idx=3)      # out: 14x14
        self.layer4 = self._make_layer(block, in_planes * 8, num_blocks[3], stride=2, layer_idx=4)      # out: 7x7

        self.linear = nn.Linear(in_planes * 8, num_classes, bias=True)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, layer_idx):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.hardtanh(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.hardtanh(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.avg_pool2d(out, out.size()[3])

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


@ARCH_REGISTRY.register()
class ResNetF18L(ResNet):
    def __init__(self, num_classes):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes, in_planes=64)


@ARCH_REGISTRY.register()
class ResNetF18S(ResNet):
    def __init__(self, num_classes):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes, in_planes=16)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params = total_params + np.prod(x.data.numpy().shape)
    print("Total number of params", total_params / 1e6)
    print("Total layers", len(list(filter(
        lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))
    print()


if __name__ == "__main__":
    import timm
    test(timm.create_model("resnet18"))
    test(ResNetF18L(1000))
    test(ResNetF18S(1000))
