import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from torch.autograd import Variable

__all__ = ['resnet16_2', 'resnet16_8', 'resnet16_16', 'resnet16_32']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = dropout
        self.dropout_layer = nn.Dropout2d(p=0.2)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.dropout is True:
            out = self.dropout_layer(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, color_dim,  k=1, n=2, num_classes=10, dropout=False):
        self.inplanes = 16
        self.color_dim = color_dim
        self.block_dropout = dropout

        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(self.color_dim, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16 * k, n)
        self.layer2 = self._make_layer(block, 32 * k, n, stride=2)
        self.layer3 = self._make_layer(block, 64 * k, n, stride=2)

        #different for cifar and mnist
        #color dim = 3 for cifar, for mnist and alike color dim = 1
        self.avgpool = nn.AvgPool2d(8)
        if color_dim == 1:
            self.avgpool = nn.AvgPool2d(7)

        self.fc_mult = 64
        if num_classes == 3832: #hack - do properly
            self.fc_mult = 256

        # This is how all the experiments were performed - Nov & Dec 2020
        # However, the average pool should've been increased with the increasing resolution
        if num_classes == 200: #tiny imagenet
            self.fc_mult = 256

        self.fc = nn.Linear(self.fc_mult * k, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.block_dropout))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dropout = self.block_dropout))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x


def resnet16(n, k, color_dim, num_classes, dropout = False):
    model = ResNet(BasicBlock,color_dim, k, n, num_classes=num_classes, dropout=dropout)
    return model


def resnet16_32(color_dim, nc, dropout = False):
    return resnet16(2, 32, color_dim, num_classes=nc, dropout=dropout)


def resnet16_16(color_dim,nc, dropout = False):
    return resnet16(2, 16, color_dim, num_classes=nc, dropout=dropout)


def resnet16_8(color_dim,nc, dropout = False):
    return resnet16(2, 8,color_dim, num_classes=nc, dropout=dropout)


def resnet16_2(color_dim, nc, dropout = False):
    return resnet16(2, 2,color_dim, num_classes=nc, dropout=dropout)
