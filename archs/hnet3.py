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
    

class BasicBlock_1w4a(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = Conv2d_IR_LSQ_1w4a(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        return x


class BasicBlock_1w4a_QBN(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = Conv2d_IR_LSQ_1w4a(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d_STE_1w4a(planes)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        return x


class HNet(nn.Module):
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

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        return x
    
    def forward(self, x):
        x = self.forward_features(x)

        x = self.act2(self.norm2(self.conv2(x)))

        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)

        x = self.linear(x)

        return x



class HNet_1w4a(nn.Module):
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

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        # quant act LSQ
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(self.Qp))
            self.init_state.fill_(1)

            print("alpha:", self.alpha.data)
        
        g = 1.0 / math.sqrt(x.numel() * self.Qp)
        alpha = grad_scale(self.alpha, g)
        x = round_pass((x / alpha).clamp(self.Qn, self.Qp)) * alpha

        return x
    
    def forward(self, x):
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
    class HNet3_b18(HNet):
        def __init__(self, num_classes):
            super().__init__(3, num_classes, BasicBlock, [1, 2, 2, 3, 3, 4, 1], 32, [16, 24, 40, 72, 108, 172, 236, 256])
    

    @ARCH_REGISTRY.register()
    class HNet3_b18_1w4a(HNet_1w4a):
        def __init__(self, num_classes):
            super().__init__(3, num_classes, BasicBlock_1w4a, [1, 2, 2, 3, 3, 4, 1], 32, [16, 24, 40, 72, 108, 172, 236, 256])
    

    @ARCH_REGISTRY.register()
    class HNet3_b18_1w4a_QBN(HNet_1w4a):
        def __init__(self, num_classes):
            super().__init__(3, num_classes, BasicBlock_1w4a_QBN, [1, 2, 2, 3, 3, 4, 1], 32, [16, 24, 40, 72, 108, 172, 236, 256])
    

else:
    model = HNet(3, 1000, BasicBlock_1w4a, [1, 2, 2, 3, 3, 4, 1], 32, [16, 24, 40, 72, 108, 172, 236, 256])

    total = sum([param.nelement() for param in model.parameters()])

    print("Total parameters: %.2f M" % (total / 1e6))
    


