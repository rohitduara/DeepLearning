#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 17:23:47 2019

@author: markpeddle
"""

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import time

try:
    transform = transforms.Compose([transforms.Resize((64, 64), interpolation=Image.BICUBIC), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
except:
    transform = transforms.Compose([transforms.Scale((64, 64), interpolation=Image.BICUBIC), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
trainset = datasets.CIFAR10(root='datasets/data_cifar10', train=True, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=4,\
                                    shuffle=True,drop_last=True, num_workers=4)

dataiter = iter(train_loader)
images, labels = dataiter.next()

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

images.size()

imshow(torchvision.utils.make_grid(images))

print(labels)
for i in labels:
    print(classes[i])
    
imshow(torchvision.utils.make_grid(images[2]))

import torch.nn as nn

## first argumennt is the channel input : 3
## second argument is the channel output you want: 6
## last is the kernel size

conv1 = nn.Conv2d(3, 6, 5)

images[2].unsqueeze(0).size()

input=images[2].unsqueeze(0)

"""
Dimension of the output is ceil( (64+2*0-5) /1 ) +1 =60

OUTPUT_DIM= ceil( input+2*padding-kernel / stride ) +1
"""

conv1(input).size()

pool = nn.MaxPool2d(2, 2)

pool(conv1(input)).size()

model = nn.Sequential(
          nn.Conv2d(3, 6, 5),
          nn.MaxPool2d(2, 2),
        )

model(input).size()

last_layer=nn.Linear(6*30*30, 10)

last_layer

last_layer(model(input).view(1,6*30*30))

import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.max_pool1 = nn.MaxPool2d(2, 2)
        self.last_layer = nn.Linear(6*30*30, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        
        x = self.max_pool1(x)
        
        x = x.view(-1, 6*30*30) ## IMPORTANT TO RESHAPE
        
        x = self.last_layer(x)
        softmax_layer=nn.Softmax(dim=1)(x) ## optional
        return softmax_layer
    
cnn=CNN()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

count_parameters(cnn)







