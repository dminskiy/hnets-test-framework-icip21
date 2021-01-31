import torch.nn as nn
import numpy as np

__all__ = ['wrnscat_50_2', 'wrnscat_short_50_2']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

##Bottleneck resBlock, ref Zagoruyko WRN.
class BasicBlock_bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, conv_planes, out_planes, upsample, downsample):
        super(BasicBlock_bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, conv_planes, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(conv_planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(conv_planes, conv_planes)
        self.bn2 = nn.BatchNorm2d(conv_planes)

        self.conv3 = nn.Conv2d(conv_planes, out_planes, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.upsample = upsample
        self.downsample = downsample
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

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    ## ref: https://github.com/szagoruyko/wide-residual-networks/blob/master/pretrained/README.md
    def __init__(self, block, scatter_size, color_dim, num_classes, k=1):
        self.color_dim = color_dim
        self.scatter_size = scatter_size
        self.pooling_coef = int(scatter_size / 5) #was 5 by default for input of 224 with J=3, scat size = 28

        super(ResNet, self).__init__()
        self.bn0 = nn.BatchNorm2d(self.color_dim, eps=1e-5, affine=False)
        self.layer3 = self._make_layer(block, self.color_dim, 256, 256*k, 1)
        self.layer4 = self._make_layer(block, 256*k, 256, 256*k, 3)
        self.layer5 = self._make_layer(block, 256*k, 512, 512*k, 1)
        self.layer6 = self._make_layer(block, 512*k, 512, 512*k, 4)
        self.layer7 = self._make_layer(block, 512*k, 1024, 1024*k, 1)
        self.layer8 = self._make_layer(block, 1024*k, 1024, 1024*k, 2)

        self.avgpool = nn.AvgPool2d(self.pooling_coef)
        self.after_pooling_size = int(self.scatter_size / self.pooling_coef)

        self.fc = nn.Linear(1024*k*self.after_pooling_size*self.after_pooling_size, num_classes)

    def _make_layer(self, block, planes, conv_planes, out_planes, blocks):
        upsample = None
        if planes < out_planes:
            upsample = nn.Sequential(
                nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_planes),
            )

        downsample = None
        if planes > out_planes and out_planes % planes != 0:
            downsample = nn.Sequential(
                nn.Conv2d(self.color_dim, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes),
            )

        layers = []
        for i in range(blocks):
            layers.append(block(planes, conv_planes, out_planes, upsample, downsample))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), self.color_dim, self.scatter_size, self.scatter_size)
        x = self.bn0(x)
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

class ResNetShort(nn.Module):
    ## ref: https://github.com/szagoruyko/wide-residual-networks/blob/master/pretrained/README.md
    def __init__(self, block, scatter_size, color_dim, num_classes, k=1):
        self.color_dim = color_dim
        self.scatter_size = scatter_size
        self.pooling_coef = int(scatter_size / 5) #was 5 by default for input of 224 with J=3, scat size = 28

        super(ResNetShort, self).__init__()
        self.bn0 = nn.BatchNorm2d(self.color_dim, eps=1e-5, affine=False)
        self.layer3 = self._make_layer(block, self.color_dim, 256 * k, 256 * k, 1)  # replace layer 3
        self.layer4 = self._make_layer(block, 256 * k, 256, 256*k, 3)
        self.layer5 = self._make_layer(block, 256*k, 512, 512*k, 1)
        self.layer6 = self._make_layer(block, 512*k, 512, 512*k, 4)
        self.layer7 = self._make_layer(block, 512*k, 1024, 1024*k, 1)
        self.layer8 = self._make_layer(block, 1024*k, 1024, 1024*k, 2)

        self.avgpool = nn.AvgPool2d(self.pooling_coef)
        self.after_pooling_size = int(self.scatter_size / self.pooling_coef)

        self.fc = nn.Linear(1024*k*self.after_pooling_size*self.after_pooling_size, num_classes)

    def _make_layer(self, block, planes, conv_planes, out_planes, blocks):
        upsample = None
        if planes < out_planes:
            upsample = nn.Sequential(
                nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_planes),
            )

        downsample = None
        if planes > out_planes and out_planes % planes != 0:
            downsample = nn.Sequential(
                nn.Conv2d(planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes),
            )

        layers = []
        for i in range(blocks):
            layers.append(block(planes, conv_planes, out_planes, upsample, downsample))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), self.color_dim, self.scatter_size, self.scatter_size)
        x = self.bn0(x)
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


def resnet50_scatter(scatter_size, color_dim, num_classes, k):
    model = ResNet(BasicBlock_bottleneck, scatter_size, color_dim, num_classes, k)
    return model

def resnet50_short(scatter_size, color_dim, num_classes, k):
    model = ResNetShort(BasicBlock_bottleneck, scatter_size, color_dim, num_classes, k)
    return model

def wrnscat_50_2(scatter_size, color_dim, num_classes):
    model = resnet50_scatter(scatter_size, color_dim, num_classes, 2)
    return model

def wrnscat_short_50_2(scatter_size, color_dim, num_classes):
    model = resnet50_short(scatter_size, color_dim, num_classes, 2)
    return model