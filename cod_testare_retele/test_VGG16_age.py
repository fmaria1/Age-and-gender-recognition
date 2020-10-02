'''
/****************************************************************************
 *                                                                          *
 *  File:        test_VGG16_age.py                                          *
 *  Copyright:   (c) 2020, Maria Frentescu                                  *
 *  Description: This script is used to test a convolutional neural network *
 *               for age prediction using VGG16 arhitecture.                *
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
    [transforms.Resize((64, 64)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = torchvision.datasets.ImageFolder(datasetDir, transform=transform)
dataset_test = torchvision.datasets.ImageFolder(datasetDir_test, transform=transform)

batch_size = 64
batch_size_test = 1
loader = {
    'train': DataLoader(dataset, batch_size=batch_size, shuffle=True),
    'test': DataLoader(dataset_test, batch_size=batch_size_test, shuffle=True),
}
# VGG16 arhitecture
VGG_16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
class Model(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_16)

        self.fcs = nn.Sequential(
            # 2 = 64 / (2^5) (5-maxpool layers)
            nn.Linear(512 * 2 * 2, 4096),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                           nn.BatchNorm2d(x),
                           nn.LeakyReLU()]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)


model = Model().cuda()
model.load_state_dict(torch.load("/kaggle/input/64batch/model_vgg16_age.pt", map_location=device))

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