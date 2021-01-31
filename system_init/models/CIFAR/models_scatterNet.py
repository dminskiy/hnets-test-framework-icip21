import torch.nn as nn

__all__ = ['resnet12_8_scat', 'resnet12_16_scat', 'resnet12_32_scat']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class View(nn.Module):
    def __init__(self, *args):
        super(View, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(-1, *self.shape)


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

        if self.dropout:
            out = self.dropout_layer(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, size_after_scat, nfscat, k=1, n=2, num_classes=10, dropout=False):
        self.inplanes = 16 * k
        self.ichannels = 16 * k
        super(ResNet, self).__init__()

        self.block_dropout = dropout
        self.nspace = size_after_scat
        self.nfscat = nfscat
        self.bn0 = nn.BatchNorm2d(self.nfscat, eps=1e-5, affine=False)
        self.conv1 = nn.Conv2d(self.nfscat, self.ichannels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.ichannels)
        self.relu = nn.ReLU(inplace=True)

        self.layer2 = self._make_layer(block, 32 * k, n)
        self.layer3 = self._make_layer(block, 64 * k, n)

        self.avgpool = nn.AvgPool2d(size_after_scat)

        self.fc = nn.Linear(64 * k, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dropout=self.block_dropout))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), self.nfscat, self.nspace, self.nspace)
        x = self.bn0(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x


def resnet12_scat(size_after_scat, nfscat, n, k, num_classes=10, dropout = False):

    model = ResNet(BasicBlock, size_after_scat, nfscat, k, n, num_classes=num_classes, dropout = dropout)
    return model


def resnet12_32_scat(size_after_scat, nfscat, num_classes, dropout = False):
    return resnet12_scat(size_after_scat, nfscat, 2, 32, num_classes=num_classes, dropout = dropout)


def resnet12_16_scat(size_after_scat, nfscat, num_classes, dropout = False):
    return resnet12_scat(size_after_scat, nfscat, 2, 16, num_classes=num_classes, dropout = dropout)


def resnet12_8_scat(size_after_scat, nfscat, num_classes, dropout = False):
    return resnet12_scat(size_after_scat, nfscat, 2, 8, num_classes=num_classes, dropout= dropout)

