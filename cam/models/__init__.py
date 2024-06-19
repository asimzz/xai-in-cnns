import torch
import torch.nn as nn
from torchvision import models
from .visualizer import CAMVisualizer


class ResNet18WithCAM(nn.Module):
    def __init__(self, num_classes=16):
        super(ResNet18WithCAM, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)

    def forward(self, images):
        layers = [
            self.resnet18.conv1,
            self.resnet18.bn1,
            self.resnet18.relu,
            self.resnet18.maxpool,
            *self.resnet18.layer1,
            *self.resnet18.layer2,
            *self.resnet18.layer3,
            *self.resnet18.layer4
        ]
        output = images
        for layer in layers:
            output = layer(output)

        self.feature_maps = output

        output = self.resnet18.avgpool(output)
        output = torch.flatten(output, 1)
        output = self.resnet18.fc(output)
        return output


class GoogLeNetWithCAM(nn.Module):
    def __init__(self, num_classes=16):
        super(GoogLeNetWithCAM, self).__init__()
        self.googlenet = models.googlenet(pretrained=True)
        self.googlenet.fc = nn.Linear(self.googlenet.fc.in_features, num_classes)

    def forward(self, images):
        layers = [
            self.googlenet.conv1,
            self.googlenet.maxpool1,
            self.googlenet.conv2,
            self.googlenet.conv3,
            self.googlenet.maxpool2,
            self.googlenet.inception3a,
            self.googlenet.inception3b,
            self.googlenet.maxpool3,
            self.googlenet.inception4a,
            self.googlenet.inception4b,
            self.googlenet.inception4c,
            self.googlenet.inception4d,
            self.googlenet.inception4e,
            self.googlenet.maxpool4,
            self.googlenet.inception5a,
            self.googlenet.inception5b,
        ]
        output = images
        for layer in layers:
            output = layer(output)

        self.feature_maps = output

        output = self.googlenet.avgpool(output)
        output = torch.flatten(output, 1)
        output = self.googlenet.fc(output)
        return output

class AlexNetWithCAM(nn.Module):
  def __init__(self, num_classes, image_channels=3):
    super(AlexNetWithCAM, self).__init__()
    self.layer1 = nn.Sequential(
      nn.Conv2d(image_channels, 96, kernel_size=11, stride=4,padding=0),
      nn.BatchNorm2d(96),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2)
    )
    self.layer2 = nn.Sequential(
      nn.Conv2d(96, 256,kernel_size=5, stride=1, padding=2),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3,stride=2)
    )
    self.layer3 = nn.Sequential(
     nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
     nn.BatchNorm2d(384),
     nn.ReLU(inplace=True),
     nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
     nn.BatchNorm2d(384),
     nn.ReLU(inplace=True),
     nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
     nn.BatchNorm2d(256),
     nn.ReLU(inplace=True),
     nn.MaxPool2d(kernel_size=3,stride=2)
    )
    self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(256, num_classes)

  def forward(self, image):
    output = self.layer1(image)
    output = self.layer2(output)
    feature_maps = self.layer3(output)
    output = self.global_avg_pool(feature_maps)
    output = output.reshape(output.size(0), -1)
    output = self.fc(output)

    return output, feature_maps