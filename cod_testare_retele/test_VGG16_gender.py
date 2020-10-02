'''
/****************************************************************************
 *                                                                          *
 *  File:        test_VGG16_gender.py                                       *
 *  Copyright:   (c) 2020, Maria Frentescu                                  *
 *  Description: This script is used to test a convolutional neural network *
 *               for gender prediction using VGG16 arhitecture.             *
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

datasetDir = '/kaggle/input/gender-all-img-64/CROPED_GENDER_FINAL/train/'
datasetDir_test = '/kaggle/input/gender-all-img-64/CROPED_GENDER_FINAL/test/'
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
    def __init__(self, in_channels=3, num_classes=2):
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
        return F.softmax(x, dim=1)

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
model.load_state_dict(torch.load("/kaggle/input/genderfull/model_vgg16_gender.pt", map_location=device))

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
