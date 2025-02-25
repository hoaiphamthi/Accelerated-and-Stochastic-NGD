'''ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Implementation:
    https://github.com/kuangliu/pytorch-cifar
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, dataset='mnist'):
        super(ResNet, self).__init__()
        self.in_planes = 64
        if dataset in ['mnist', 'FashionMNIST']:
            self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(dataset='mnist'):
    return ResNet(BasicBlock, [2,2,2,2], dataset=dataset)

def ResNet34(dataset='mnist'):
    return ResNet(BasicBlock, [3,4,6,3], dataset=dataset)

def ResNet50(dataset='mnist'):
    return ResNet(Bottleneck, [3,4,6,3], dataset=dataset)

def ResNet101(dataset='mnist'):
    return ResNet(Bottleneck, [3,4,23,3], dataset=dataset)

def ResNet152(dataset='mnist'):
    return ResNet(Bottleneck, [3,8,36,3], dataset=dataset)

class DenseNet(nn.Module):
    def __init__(self, pretrained='densenet121', num_classes=10, dataset='mnist'):
        super(DenseNet, self).__init__()
        if dataset == 'mnist':
            self.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model = torch.hub.load('pytorch/vision:v0.10.0', pretrained, pretrained=True)
        self.model.features.conv0 = self.conv0
        self.model.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        return self.model(x)

def DenseNet121(dataset='mnist'):
    return DenseNet(pretrained='densenet121', dataset=dataset)

def Densenet161(dataset='mnist'):
    return DenseNet(pretrained='densenet161', dataset=dataset)

def Densenet169(dataset='mnist'):
    return DenseNet(pretrained='densenet169', dataset=dataset)

def Densenet201(dataset='mnist'):
    return DenseNet(pretrained='densenet201', dataset=dataset)

class LeNet(nn.Module):

    def __init__(self, num_classes=10, dataset='mnist'):
        super(LeNet, self).__init__()
        if dataset == 'mnist':
            n_dim = 1
            n_linear = 256
        else:
            n_dim = 3
            n_linear = 400
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=n_dim, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(n_linear, 120),  # in_features = 16 x4x4
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
            nn.Softmax()

        )

    def forward(self, x):
        a1 = self.feature_extractor(x)
        # print(a1.shape)
        a1 = torch.flatten(a1, 1)
        a2 = self.classifier(a1)
        return a2