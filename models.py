from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=16, out_dim=10, bias_flag = False):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(out_channels, 2 * out_channels, 3, padding=1,bias=bias_flag)
        self.fc1 = nn.Linear(2 * out_channels * 8 * 8, 108, bias=bias_flag)
        self.fc2 = nn.Linear(108, 64, bias=bias_flag)
        self.fc3 = nn.Linear(64, out_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class VGG8(nn.Module):
    def __init__(self, in_channels=3,out_channels=16,out_dim=10,bias_flag = False):
        super(VGG8, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias_flag),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels*2, kernel_size=3, padding=1,bias=bias_flag),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, padding=1,bias=bias_flag),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(out_channels*2, out_channels*4, kernel_size=3, padding=1,bias=bias_flag),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels*4, out_channels*4, kernel_size=3, padding=1,bias=bias_flag),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(out_channels*4 * 4 * 4, 180,bias=bias_flag),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(180, 64,bias=bias_flag),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(64, out_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
class IdentityLayer(nn.Module):
    def forward(self, x):
        return x
        
class ResNetBlock(nn.Module):
    def __init__(self, in_size=16, out_size=16, bias_flag=False, downsample = False):
        super(ResNetBlock,self).__init__()
        self.out_size = out_size
        self.in_size = in_size
        if downsample:
            self.stride1 = 2
            self.reslayer = nn.Conv2d(in_channels=self.in_size, out_channels=self.out_size, stride=2, kernel_size=1)
        else:
            self.stride1 = 1
            self.reslayer = IdentityLayer()
        self.conv1 =nn.Conv2d(in_channels=self.in_size,out_channels=self.out_size,kernel_size=3,stride=self.stride1,padding=1,bias=bias_flag)
        self.conv2 =nn.Conv2d(in_channels=self.out_size,out_channels=self.out_size,kernel_size=3,padding=1,bias=bias_flag)
        self.bn1 = nn.BatchNorm2d(out_size)
        self.bn2 = nn.BatchNorm2d(out_size)

    def forward(self, x):
        residual = self.reslayer(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out
    
class ResNet10(nn.Module):
    def __init__(self,in_channels=3,out_dim=10,bias_flag = False):
        super(ResNet10, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.filter_size = [[16,16], [16,16], [16,16], [16,32], [32,32], [32,32], [32,64], [64,64], [64,64], [64,64]]
        layers = []
        for i in range(10):
            down_sample = False
            if i == 3 or i == 6:
                down_sample = True
            layers.append(ResNetBlock(self.filter_size[i][0], self.filter_size[i][1], bias_flag, downsample=down_sample))
        self.res_net = nn.Sequential(*layers)
        self.global_avg = nn.AvgPool2d(8)
        self.final = nn.Linear(64,out_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res_net(x)
        x = self.global_avg(x)
        x = self.final(x.view(-1,64))
        return x

class ResNet12(nn.Module):
    def __init__(self,in_channels=3,out_dim=10,bias_flag = False):
        super(ResNet12, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.filter_size = [[16,16], [16,32], [32,32], [32,32], [32,32], [32,32], 
                    [32,64], [64,64], [64,64], [64,64], [64,64], [64,64]]
        layers = []
        for i in range(12):
            down_sample = False
            if i == 1 or i == 6:
                down_sample = True
            layers.append(ResNetBlock(self.filter_size[i][0], self.filter_size[i][1],bias_flag,downsample=down_sample))
        self.res_net = nn.Sequential(*layers)
        self.global_avg = nn.AvgPool2d(8)
        self.final = nn.Linear(64,out_dim)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res_net(x)
        x = self.global_avg(x)
        x = self.final(x.view(-1,64))
        return x
    
class ResNet18(nn.Module):
    def __init__(self,in_channels=3,out_dim=10,bias_flag = False):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.filter_size = [[16,16], [16,16], [16,16], [16,16], [16,16], [16,16], [16,32], [32,32], [32,32], [32,32],
                            [32,32], [32,32], [32,64], [64,64], [64,64], [64,64], [64,64], [64,64]]
        layers = []
        for i in range(18):
            down_sample = False
            if i > 5 and i % 6 == 0:
                down_sample = True
            layers.append(ResNetBlock(self.filter_size[i][0], self.filter_size[i][1], bias_flag, downsample=down_sample))
        self.res_net = nn.Sequential(*layers)
        self.global_avg = nn.AvgPool2d(8)
        self.final = nn.Linear(64,out_dim)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res_net(x)
        x = self.global_avg(x)
        x = self.final(x.view(-1,64))
        return x

class MLP(nn.Module):
    def __init__(self, in_channels=3, out_dim=10, bias_flag = False):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear( in_channels * 32 * 32, 128, bias=bias_flag)
        self.fc2 = nn.Linear(128, 64, bias=bias_flag)
        self.fc3 = nn.Linear(64, out_dim)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


