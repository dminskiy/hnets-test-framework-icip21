import torch
import torch.nn as nn
__all__ = ["OneConvLayerNet", "ThreeFullyConnectedLayers"]


class View(nn.Module):
    def __init__(self, *args):
        super(View, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(-1,*self.shape)

''''
A simple 3x3 conv layer followed by 512 fully connected layer, performs batch normalisation
padding = 1, stride = 1, conv channels = 256 by default (can be changed)
in_size       - size of the input features (in_size x in_size)
in_channels   - number of channels at the beginning
num_classes   - number of classes to be classified
conv_channels - number of channels at the end of convolution
'''
class OneConvLayerNet(nn.Module):
    def __init__(self, in_size, in_channels, num_classes, conv_channels = 64, kernel_sz=3):
        super(OneConvLayerNet,self).__init__()
        self.conv = nn.Conv2d(in_channels, conv_channels, kernel_sz, padding=1)
        self.kernel_sz = kernel_sz
        self.fc1  = nn.Linear(conv_channels*in_size*in_size,512)
        self.fc2  = nn.Linear(512,num_classes)
        self.bn2d = nn.BatchNorm2d(in_channels)
        self.in_channels = in_channels
        self.in_size = in_size
        self.conv_channels = conv_channels
        self.relu = nn.ReLU()

    def forward(self, x):
        if torch.cuda.is_available():
            x.cuda()

        out = x.view(x.size(0), self.in_channels, self.in_size, self.in_size)

        if torch.cuda.is_available():
            out.cuda()

        out = self.bn2d(out)
        out = self.conv(out)
        out = self.relu(out)
        #just for tests, very specific for mallat_l testing with J=2
        out = out.view(out.size(0),self.conv_channels * self.in_size * self.in_size)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out
 
class ThreeFullyConnectedLayers(nn.Module):
    def __init__(self, in_size, in_channels, num_classes, fc1_coefs=2048, fc2_coefs=1024):
        super(ThreeFullyConnectedLayers, self).__init__()
        self.in_channels = in_channels
        self.in_size = in_size
        self.bn2d = nn.BatchNorm2d(in_channels)
        self.fc1 = nn.Linear(in_channels*in_size*in_size,fc1_coefs)
        self.fc2 = nn.Linear(fc1_coefs, fc2_coefs)
        self.fc3 = nn.Linear(fc2_coefs,num_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        if torch.cuda.is_available():
            x.cuda()

        if x.dim() != 4:
            x = x.view(x.size(0), -1, self.in_size, self.in_size)
        x = self.bn2d(x)

        out = x.view(x.size(0), self.in_channels* self.in_size * self.in_size)
        if torch.cuda.is_available():
            out.cuda()
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out