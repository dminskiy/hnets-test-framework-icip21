import torch.nn as nn
import math

__all__ = ['resnet50_2']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

##Bottleneck resBlock, ref Zagoruyko WRN.
class BasicBlock_bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, conv_planes, out_planes, upsample):
        super(BasicBlock_bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, conv_planes, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(conv_planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(conv_planes, conv_planes)
        self.bn2 = nn.BatchNorm2d(conv_planes)

        self.conv3 = nn.Conv2d(conv_planes, out_planes, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.upsample = upsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out = out + residual
        out = self.relu(out)

        return out

class ResNet_8layers(nn.Module):
    ## ref: https://github.com/szagoruyko/wide-residual-networks/blob/master/pretrained/README.md
    def __init__(self, block, in_shape, color_dim, num_classes, k=1):
        self.inplanes = 64
        self.color_dim = color_dim

        super(ResNet_8layers, self).__init__()
        self.conv1 = nn.Conv2d(self.color_dim, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3)

        size_after_1_stage = int(in_shape/(2*3)) #kernel size=7 canseled by padding=2, but stride = 2, maxpool = 3, hence divide by 2*3
        self.pooling_coef = int(size_after_1_stage / 5)  # 7 by default for input = 224

        self.layer1 = self._make_layer(block, self.inplanes, 128, 128*k, 1)
        self.layer2 = self._make_layer(block, 128*k, 128, 128*k, 2)
        self.layer3 = self._make_layer(block, 128*k, 256, 256*k, 1)
        self.layer4 = self._make_layer(block, 256*k, 256, 256*k, 3)
        self.layer5 = self._make_layer(block, 256*k, 512, 512*k, 1)
        self.layer6 = self._make_layer(block, 512*k, 512, 512*k, 4)
        self.layer7 = self._make_layer(block, 512*k, 1024, 1024*k, 1)
        self.layer8 = self._make_layer(block, 1024*k, 1024, 1024*k, 2)

        self.avgpool = nn.AvgPool2d(self.pooling_coef)
        self.fc = nn.Linear(1024*k*5*5, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, conv_planes, out_planes, blocks):
        upsample = None
        if planes < out_planes:
            upsample = nn.Sequential(
                nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_planes),
            )

        layers = []
        for i in range(blocks):
            layers.append(block(planes, conv_planes, out_planes, upsample))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


def resnet50(in_shape, color_dim, num_classes, k):
    model = ResNet_8layers(BasicBlock_bottleneck, in_shape, color_dim, num_classes, k)
    return model

def resnet50_2(in_shape, color_dim, num_classes):
    model = resnet50(in_shape, color_dim, num_classes, 2)
    return model