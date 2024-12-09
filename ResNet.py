import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))

        x = self.relu(self.batch_norm2(self.conv2(x)))

        x = self.conv3(x)
        x = self.batch_norm3(x)

        # downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        # add identity
        x += identity
        x = self.relu(x)

        return x


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.batch_norm2(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512 * ResBlock.expansion, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)
        features = []

        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        e1 = self.fc1(x)
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)

        # for i in features:
        #     print(i.shape)

        return x, e1, features

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * ResBlock.expansion)
            )

        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion

        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)

    def get_embedding_dim(self):
        return 50


class ResNet_rank(nn.Module):
    def __init__(self, num_classes, num_channels=3):
        super(ResNet_rank, self).__init__()

        self.convert_layer0 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(512), nn.ReLU(),
                                            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                                            nn.BatchNorm2d(512), nn.ReLU(),
                                            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
                                            nn.BatchNorm2d(512), nn.ReLU(),
                                            )

        self.convert_layer1 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(512), nn.ReLU(),
                                            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                                            nn.BatchNorm2d(512), nn.ReLU(),
                                            nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0),
                                            nn.BatchNorm2d(1024), nn.ReLU(),
                                            )

        self.convert_layer2 = nn.Sequential(nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(1024), nn.ReLU(),
                                            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
                                            nn.BatchNorm2d(1024), nn.ReLU(),
                                            nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0),
                                            nn.BatchNorm2d(2048), nn.ReLU(),
                                            )

        self.convert_layer3 = nn.Sequential(nn.Conv2d(4096, 2048, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(2048), nn.ReLU())

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048*7*7, 2048)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(2048, num_classes)

    def forward(self, x):
        x_1 = self.convert_layer0(x[0])
        x_2 = self.convert_layer1(torch.cat((x_1, x[1]), 1))
        x_3 = self.convert_layer2(torch.cat((x_2, x[2]), 1))
        x_4 = self.convert_layer3(torch.cat((x_3, x[3]), 1))

        x_4 = x_4.reshape(x_4.shape[0], -1)
        x_4 = self.dropout(self.fc1(x_4))
        x_4 = self.fc2(x_4)

        return x_4


class LossNet(nn.Module):
    def __init__(self, feature_sizes=[32, 16, 8, 4], num_channels=[256, 512, 1024, 2048], interm_dim=128):
        super(LossNet, self).__init__()

        self.GAP1 = nn.AvgPool2d(feature_sizes[0])
        self.GAP2 = nn.AvgPool2d(feature_sizes[1])
        self.GAP3 = nn.AvgPool2d(feature_sizes[2])
        self.GAP4 = nn.AvgPool2d(feature_sizes[3])

        self.FC1 = nn.Linear(num_channels[0], interm_dim)
        self.FC2 = nn.Linear(num_channels[1], interm_dim)
        self.FC3 = nn.Linear(num_channels[2], interm_dim)
        self.FC4 = nn.Linear(num_channels[3], interm_dim)

        self.linear = nn.Linear(4 * interm_dim, 1)

    def forward(self, features):
        out1 = self.GAP1(features[0])
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        out2 = self.GAP2(features[1])
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out3 = self.GAP3(features[2])
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.FC3(out3))

        out4 = self.GAP4(features[3])
        out4 = out4.view(out4.size(0), -1)
        out4 = F.relu(self.FC4(out4))

        out = self.linear(torch.cat((out1, out2, out3, out4), 1))
        return out


def ResNet50(num_classes, channels=3):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, channels)


def ResNet101(num_classes, channels=3):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, channels)


def ResNet152(num_classes, channels=3):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, channels)


def Res_rank(num_classes, channels=3):
    return ResNet_rank(num_classes, channels)


