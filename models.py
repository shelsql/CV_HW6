import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG(nn.Module):
    def __init__(self, in_height=32, in_width=32, in_channels=3, dropout=0.5):
        super().__init__()
        print("VGG16 initializing...")
        # 1. define multiple convolution and downsampling layers
        # 3. define full-connected layer to classify
        self.config = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
        convs = []
        for layer in self.config:
            if type(layer) == int:
                out_channels = layer
                convs.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1))
                convs.append(nn.BatchNorm2d(out_channels))
                convs.append(nn.ReLU())
                in_channels = out_channels
            elif layer == "M":
                convs.append(nn.MaxPool2d(2, 2))
        self.convs = nn.Sequential(*convs)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, 4096),
            nn.ReLU(),
            nn.Dropout(p = dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p = dropout),
            nn.Linear(4096, 10)
        )
                

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # extract features
        # classification
        out = self.convs(x)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out


class ResBlock(nn.Module):
    ''' residual block'''
    def __init__(self, in_channels, out_channels, stride, downsample = None):
        super().__init__()
        '''
        in_channel: number of channels in the input image.
        out_channel: number of channels produced by the convolution.
        stride: stride of the convolution.
        '''
        # 1. define double convolution
             # convolution
             # batch normalization
             # activate function
             # ......
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU()
        self.use_conv3 = in_channels != out_channels or stride != 1
        # 2. if in_channel != out_channel or stride != 1, deifine 1x1 convolution layer to change the channel or size.
        if self.use_conv3:
            self.conv3 = nn.Conv2d(in_channels, out_channels, 1, stride)
        # Note: we are going to implement 'Basic residual block' by above steps, you can also implement 'Bottleneck Residual block'

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # 1. convolve the input
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        # 2. if in_channel != out_channel or stride != 1, change the channel or size of 'x' using 1x1 convolution.
        if self.use_conv3:
            x = self.conv3(x)
        # 3. Add the output of the convolution and the original data (or from 2.)
        out = out + x
        # 4. relu
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    '''residual network'''
    def __init__(self):
        super().__init__()
        print("ResNet18 initializing...")
        self.config = [(2, 64, False), (2, 128, True), (2, 256, True), (2, 512, True)]
        # 1. define convolution layer to process raw RGB image
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )
        # 2. define multiple residual blocks
        blocks = []
        in_channels = 64
        for num, channels, down in self.config:
            block = []
            for i in range(num):
                out_channels = channels
                stride = 2 if i == 0 and down else 1
                block.append(ResBlock(in_channels, out_channels, stride))
                in_channels = out_channels
            blocks.append(nn.Sequential(*block))
        self.blocks = nn.Sequential(*blocks)
        # 3. define full-connected layer to classify
        self.maxpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # extract features
        # classification
        out = self.conv1(x)
        out = self.blocks(out)
        out = self.maxpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.linear(out)
        return out
    

class ResNextBlock(nn.Module):
    '''ResNext block'''
    def __init__(self, in_channels, out_channels, bottle_neck, group, stride):
        super().__init__()
        # in_channel: number of channels in the input image
        # out_channel: number of channels produced by the convolution
        # bottle_neck: int, bottleneck= out_channel / hidden_channel 
        # group: number of blocked connections from input channels to output channels
        # stride: stride of the convolution.

        # 1. define convolution
             # 1x1 convolution
             # batch normalization
             # activate function
             # 3x3 convolution
             # ......
             # 1x1 convolution
             # ......
        hidden_channels = out_channels // bottle_neck
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size = 1, stride = 1),
            nn.BatchNorm2d(hidden_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride = stride,
                      padding = 1, groups = group),
            nn.BatchNorm2d(hidden_channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size = 1, stride = 1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU()
        self.use_conv4 = in_channels != out_channels or stride != 1
        # 2. if in_channel != out_channel or stride != 1, deifine 1x1 convolution layer to change the channel or size.
        if self.use_conv4:
            self.conv4 = nn.Conv2d(in_channels, out_channels, 1, stride)

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # 1. convolve the input
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        # 2. if in_channel != out_channel or stride != 1, change the channel or size of 'x' using 1x1 convolution.
        if self.use_conv4:
            x = self.conv4(x)
        # 3. Add the output of the convolution and the original data (or from 2.)
        out = out + x
        # 4. relu
        out = self.relu(out)
        return out


class ResNext(nn.Module):
    def __init__(self):
        super().__init__()
        print("ResNext18 initializing...")
        self.config = [(2, 64, False), (2, 128, True), (2, 256, True), (2, 512, True)]
        # 1. define convolution layer to process raw RGB image
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )
        # 2. define multiple residual blocks
        blocks = []
        in_channels = 64
        for num, channels, down in self.config:
            block = []
            for i in range(num):
                out_channels = channels
                stride = 2 if i == 0 and down else 1
                block.append(ResNextBlock(in_channels, out_channels, bottle_neck = 4,
                                          group = 8 ,stride = stride))
                in_channels = out_channels
            blocks.append(nn.Sequential(*block))
        self.blocks = nn.Sequential(*blocks)
        # 3. define full-connected layer to classify
        self.maxpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # extract features
        # classification
        out = self.conv1(x)
        out = self.blocks(out)
        out = self.maxpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.linear(out)
        return out

