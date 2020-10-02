'''
/****************************************************************************
 *                                                                          *
 *  File:        test_ResNet50_gender.py                                    *
 *  Copyright:   (c) 2020, Maria Frentescu                                  *
 *  Description: This script is used to test a convolutional neural network *
 *               for gender prediction using ResNet50 arhitecture.          *
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
import tensorflow as tf
from sklearn import metrics
import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

datasetDir = '/kaggle/input/genderresnetfull/CROPED_GENDER_FINAL/train/'
datasetDir_test = '/kaggle/input/genderresnetfull/CROPED_GENDER_FINAL/test/'
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

        return F.softmax(x, dim=1)

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


def ResNet50(img_channel=3, num_classes=2):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)


model = ResNet50(3, 2).cuda()
model.load_state_dict(torch.load("/kaggle/input/genderfull/model_resnet50_gender.pt", map_location=device))

lossFunc = nn.MSELoss()
timea = datetime.datetime.now()

num_correct = 0
num_samples = 0
sum_loss = 0
pic_plot = []
acc_plot = []
class_real = []
loss_plot = []
class_predicted = []

# testing
model.eval()
with torch.no_grad():
    for batch_idx, (image, label) in enumerate(loader['test']):
        image = image.cuda()
        # real_class = 0 for female or 1 for male
        real_class = label.cuda().type(torch.float32)
        # scores = an array with 2 values
        scores = model(image)
        # predicted_class = max of the 2 values
        predicted_class = torch.argmax(scores, dim=1).type(torch.float32)
        #convert to int
        real_int = real_class.type(torch.int32)
        pred_int = predicted_class.type(torch.int32)

        # saving information for confusion matrix
        class_real += [real_int.item()]
        class_predicted += [pred_int.item()]
        
        loss = lossFunc(predicted_class, real_class)
        
        num_correct += (real_class == predicted_class).sum()
        num_samples += 1
        
        if num_samples % 50 == 0:
            pic_plot += [num_samples]
            acc = (float(num_correct) / float(num_samples)) * 100
            acc_plot += [acc]
            loss_plot += [loss.item()]
            
    print("Accuracy for Test_dataset:")
    print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')

    pic_plot += [num_samples]
    acc = (float(num_correct) / float(num_samples)) * 100
    acc_plot += [acc]
    loss_plot += [loss.item()]

# accuracy plot
plt.figure()
plt.plot(pic_plot, acc_plot)
plt.title('Test accuracy information')
plt.xlabel('Test_dataset')
plt.ylabel('Accuracy')
plt.show()
plt.savefig('acc_plot.jpg', dpi=300, quality=80, optimize=True, progressive=True)

# loss plot
plt.figure()
plt.plot(pic_plot, loss_plot)
plt.title('Test loss information')
plt.xlabel('Test_dataset')
plt.ylabel('Loss')
plt.show()
plt.savefig('loss_test_plot.jpg', dpi=300, quality=80, optimize=True, progressive=True)

# confusion matrix
alphabets = ['Female', 'Male']
matrix = metrics.confusion_matrix(class_real, class_predicted, labels=[0, 1])
data = matrix

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(data, cmap='Greens', interpolation='nearest')
fig.colorbar(cax)

ax.set_xticklabels([''] + alphabets)
ax.set_yticklabels([''] + alphabets)
for (i, j), z in np.ndenumerate(data):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
plt.title('Confusion matrix')
plt.xlabel('Predicted label')
plt.ylabel('Real label')
plt.show()
plt.savefig('matrix_plot.jpg', dpi=300, quality=80, optimize=True, progressive=True)


timeb = datetime.datetime.now()
print("time in seconds: ", (timeb - timea).seconds)
