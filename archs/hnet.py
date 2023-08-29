import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

if __name__=="__main__":
    pass
else:
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


### BasicBlock: version 0
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if in_planes != planes:
                self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
            else:
                self.shortcut = LambdaLayer(lambda x: x[:, :, ::2, ::2])

    def forward(self, x):
        x = self.bn1(self.conv1(x)) + self.shortcut(x)
        x = F.relu(x)

        x = self.bn2(self.conv2(x)) + x
        x = F.relu(x)
        
        return x


### BasicBlock: version 1
class BasicBlock1(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = F.relu(x)

        x = self.bn2(self.conv2(x)) 
        x = F.relu(x)
        
        return x


class HNet(nn.Module):
    def __init__(self, in_channels, num_classes, block, num_blocks, in_planes):
        super().__init__()
        self.in_planes = in_planes

        self.conv1 = nn.Conv2d(in_channels, in_planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(in_planes)
        self.act1 = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, in_planes * 1, num_blocks[0], stride=2)      # in_shape: 112x112; out_channels: 64
        self.layer2 = self._make_layer(block, in_planes * 2, num_blocks[1], stride=2)      # out: 28x28
        self.layer3 = self._make_layer(block, in_planes * 4, num_blocks[2], stride=2)      # out: 14x14
        self.layer4 = self._make_layer(block, in_planes * 8, num_blocks[3], stride=2)      # out: 7x7

        self.linear = nn.Linear(in_planes * 8, num_classes, bias=True)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward_features(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
    
    def forward(self, x):
        x = self.forward_features(x)

        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)

        x = self.linear(x)

        return x


if __name__!="__main__":
    @ARCH_REGISTRY.register()
    class HNet0_b18(HNet):
        def __init__(self, num_classes):
            super().__init__(3, num_classes, BasicBlock, [2, 2, 2, 2], in_planes=64)


    @ARCH_REGISTRY.register()
    class HNet0_s18(HNet):
        def __init__(self, num_classes):
            super().__init__(3, num_classes, BasicBlock, [2, 2, 2, 2], in_planes=16)
    

    @ARCH_REGISTRY.register()
    class HNet1_b18(HNet):
        def __init__(self, num_classes):
            super().__init__(3, num_classes, BasicBlock1, [2, 2, 2, 2], in_planes=64)


    @ARCH_REGISTRY.register()
    class HNet1_s18(HNet):
        def __init__(self, num_classes):
            super().__init__(3, num_classes, BasicBlock1, [2, 2, 2, 2], in_planes=32)


    @ARCH_REGISTRY.register()
    class HNet1_n18(HNet):
        def __init__(self, num_classes):
            super().__init__(3, num_classes, BasicBlock1, [2, 2, 2, 2], in_planes=16)

else:
    model = HNet(3, 1000, BasicBlock1, [2, 2, 2, 2], in_planes=32)

    total = sum([param.nelement() for param in model.parameters()])

    print("Total parameters: %.2f M" % (total / 1e6))
