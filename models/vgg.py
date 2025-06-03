import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.projection = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.projection = None

    def forward(self, x):
        identity = x
        out = F.leaky_relu(self.conv1(x))
        out = F.leaky_relu(self.bn2(self.conv2(out)))
        
        if self.projection is not None:
            identity = self.projection(identity)
        
        out += identity
        return F.leaky_relu(out)

class VGG8like(nn.Module):
    def __init__(self):
        super(VGG8like, self).__init__()
        # Convolutional layers
        self.projection0 = nn.Conv2d(3, 16, 1)
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv3 = nn.Conv2d(16, 16, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(16)

        self.BlockA = ResidualBlock(16, 16)
        
        self.BlockB = ResidualBlock(16, 16)

        self.BlockC = ResidualBlock(16, 16)
        
        self.BlockD = ResidualBlock(16, 32)

        self.BlockE = ResidualBlock(32, 32)

        self.BlockF = ResidualBlock(32, 64)

        self.BlockG = ResidualBlock(64, 64)

        self.BlockH = ResidualBlock(64, 128)

        self.BlockI = ResidualBlock(128, 128)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc1 = nn.Linear(128, 16)
        self.fc2 = nn.Linear(16, 2)

        self.spatial_dropout = nn.Dropout2d(p=0.2)  

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        identity = self.projection0(x)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.bn1(self.conv3(x)))
        x = x + identity 

        x = self.BlockA(x)

        x = self.BlockB(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.BlockC(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.BlockD(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.BlockE(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.BlockF(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.BlockG(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.BlockH(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.BlockI(x)
        
        x = self.global_avg_pool(x)  # Global average pooling

        x = x.view(x.size(0), -1)  # Flatten the output
        if self.training:
            x = F.dropout(x, p=0.5)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))

        return x