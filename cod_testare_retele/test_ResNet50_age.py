'''
/****************************************************************************
 *                                                                          *
 *  File:        test_ResNet50_age.py                                       *
 *  Copyright:   (c) 2020, Maria Frentescu                                  *
 *  Description: This script is used to test a convolutional neural network *
 *               for age prediction using ResNet50 arhitecture.             *
 *                                                                          *
 ***************************************************************************/
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import datetime
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
datasetDir = '/kaggle/input/64batch/CROPED_AGE_FINAL/train/'
datasetDir_test = '/kaggle/input/64batch/CROPED_AGE_FINAL/test/'
transform = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = torchvision.datasets.ImageFolder(datasetDir, transform=transform)
dataset_test = torchvision.datasets.ImageFolder(datasetDir_test, transform=transform)

batch_size = 32
batch_size_test = 1
loader = {
    'train': DataLoader(dataset, batch_size=batch_size, shuffle=True),
    'test': DataLoader(dataset_test, batch_size=batch_size_test, shuffle=True),
}

# ResNet50 arhitecture
class block(nn.Module):
    def __init__(
            self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )


        self.in_channels = intermediate_channels * 4

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)


def ResNet50(img_channel=3, num_classes=1):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)


model = ResNet50(3, 1).cuda()
model.load_state_dict(torch.load("/kaggle/input/64batch/model_resnet50_age.pt", map_location=device))

timea = datetime.datetime.now()
sum_error = 0
total_images = 0
pic_plot = []
loss_test_plot = []

# testing
model.eval()
with torch.no_grad():
    for batch_idx, (image, label) in enumerate(loader['test']):
        image = image.cuda()
        # label in range of [18:50]
        # real class must be [0.18:0.50] 
        label = .18 + label.reshape((batch_size_test, 1)).cuda().type(torch.float32) / 100
        # model return a value in range of [0:1]
        predicted = model(image)
        
        real_class = (label * 100).type(torch.int64)
        predicted_class = (predicted * 100).type(torch.int64)
        x=real_class.item()
        y=predicted_class.item()
        
        loss = (y - x)**2 
        sum_error += loss
        total_images += 1
        
        # saving values for loss graph
        if total_images % 100 == 0:
            pic_plot += [total_images]
            loss_test_plot += [loss]

    print("Error for Test_dataset:")
    print(f'Formula = radical({sum_error} / {total_images}) with RMSE error {math.sqrt(float(sum_error) / float(total_images)) :.2f}')
pic_plot += [total_images]
loss_test_plot += [loss]
    
plt.figure()
plt.plot(pic_plot, loss_test_plot)
plt.title('Evolution of RMSE error for test dataset')
plt.xlabel('Test_dataset')
plt.ylabel('Error')
plt.show()
plt.savefig('loss_test_plot1.jpg', dpi=300, quality=80, optimize=True, progressive=True)
      
timeb = datetime.datetime.now()
print("time in seconds: ", (timeb - timea).seconds)