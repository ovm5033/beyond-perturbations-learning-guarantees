from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock
import torch.nn.functional as F


class MnistResNet(ResNet):
    def __init__(self, num_labels=10):
        super(MnistResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_labels)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=2, bias=False)
        
    def forward(self, x):
        return super(MnistResNet, self).forward(x) #torch.softmax(super(MnistResNet, self).forward(x), dim=-1)

class MnistNet(nn.Module):
    def __init__(self, num_labels=10):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_labels)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        #output = F.log_softmax(x, dim=1)
        return x

class SmallFNN(nn.Module):
    def __init__(self, num_labels=10):
        super(SmallFNN, self).__init__()
        self.hidden_layer1 = nn.Sequential(
                                nn.Linear(784, 100, bias=False),
                                nn.Dropout(p=0.0),
                                nn.ReLU()
                            )
        self.hidden_layer2 = nn.Sequential(
                                nn.Linear(100, 100, bias=False),
                                nn.Dropout(p=0.0),
                                nn.ReLU()
                            )

        self.output_layer = nn.Linear(100, num_labels, bias=False)

    def forward(self, x):
        x = x.view(x.shape[0],-1)
        hidden1 = self.hidden_layer1(x)
        hidden2 = self.hidden_layer2(hidden1)
        output = self.output_layer(hidden1)

        return output

class SmallCNN(nn.Module):
    def __init__(self, num_labels=10, drop=0.5):
        super(SmallCNN, self).__init__()

        # Mnist is 1, CIFAR is 3
        self.num_channels = 1
        self.num_labels = num_labels

        activ = nn.ReLU(True)

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 32, 3, padding=1)),
            ('relu1', activ),
            ('conv2', nn.Conv2d(32, 32, 3, padding=1)),
            ('relu2', activ),
            ('maxpool1', nn.AvgPool2d(2, 2)),
            ('conv3', nn.Conv2d(32, 64, 3, padding=1)),
            ('relu3', activ),
            ('conv4', nn.Conv2d(64, 64, 3, padding=1)),
            ('relu4', activ),
            ('maxpool2', nn.AdaptiveAvgPool2d(1)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64, 200)),
            ('relu1', activ),
            ('fc3', nn.Linear(200, self.num_labels)),
        ]))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.kaiming_normal_(self.classifier.fc3.weight, 0)
        nn.init.constant_(self.classifier.fc3.bias, 0)

    def forward(self, input):
        features = self.feature_extractor(input)
        logits = self.classifier(features.view(-1, 64))
        return logits

class LinearModel(nn.Module):
    def __init__(self, input_dim = 3072, num_labels=2):
        super(LinearModel, self).__init__()

        self.output_layer = nn.Linear(input_dim, num_labels, bias=True)

    def forward(self, x):
        x = x.view(x.shape[0],-1)
        output = self.output_layer(x)

        return output