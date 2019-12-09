#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 17:12:23 2019

@author: abdullah
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn.metrics import confusion_matrix
from torch.utils.data import random_split
from torch.utils.data.sampler import SubsetRandomSampler
import pretrainedmodels
from efficientnet_pytorch import EfficientNet


def train_model(model, criterion, optimizer, scheduler, 
                num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    iters = len(dataloaders['train'])
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i,(inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                if phase == 'train':
                    scheduler.step(epoch + i / iters)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    #torch.save(model.state_dict(), modelPath)
    return model

def balanceClasses(images,subset_idx, nclasses):                        
    count = [0] * nclasses                                                      
    for i in subset_idx:                                                        
        count[images.imgs[i][1]] += 1                                                    
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                  
    for i in range(nclasses):                                                  
        weight_per_class[i] = N/float(count[i])                                
    weight = [0] * len(subset_idx)                                              
    for i, j in enumerate(subset_idx):                                          
        weight[i] = weight_per_class[images.imgs[j][1]]                                  
    return weight      


path = os.path.join(os.path.expanduser('~'), 'Documents', 'Datasets','aptos2019-blindness-detection','train')
modelPath = os.path.join(os.path.expanduser('~'), 'Documents', 'Datasets','aptos2019-blindness-detection','pretrainedResnet151_0.pt')
'RGB Mean - 117.416, 62.874, 20.417 - all divided by 255'
'RGB Stdev - 63.443, 35.275, 20.565 - all divided by 255'
'transforms.RandomAffine(degrees = (-360,360), shear = (-45,45), translate=(0,.1)),'

transform = transforms.Compose(
        [transforms.RandomResizedCrop((320,320)),
        transforms.RandomHorizontalFlip(.5),
        transforms.RandomVerticalFlip(.5),
        transforms.RandomApply([
        transforms.ColorJitter(brightness=(1,1.5), contrast=(1,1.5), saturation=(1,1.5)),
        transforms.RandomAffine(degrees = (-360,360), shear = (-30,30))],
        p=.5),
        transforms.ToTensor(),
        transforms.Normalize([0.460, 0.247, 0.080], [0.249, 0.138, 0.081]),
        transforms.RandomErasing(inplace=True, value='random', p=.25)
        ])

image_datasets = datasets.ImageFolder(path, transform)
train, val = random_split(image_datasets, (2930, 732))
weights = balanceClasses(image_datasets,train.indices,5)



trainSize = len(train)
valSize = len(val)


weights = torch.DoubleTensor(weights)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, trainSize)
trainLoader =  torch.utils.data.DataLoader(train, sampler=sampler,
                                              num_workers=1, batch_size=32)
        
valLoader = torch.utils.data.DataLoader(val, shuffle=True,
                                              num_workers=1, batch_size=32)

dataloaders = {'train': trainLoader,
               'val': valLoader}

dataset_sizes = {'train':trainSize,
                 'val': valSize}

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.457, 0.247, 0.082])
    std = np.array([0.251, 0.140, 0.083])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.1)  # pause a bit so that plots are updated


inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
class_names = image_datasets.classes
imshow(out, title=[class_names[x] for x in classes])
print(inputs.shape)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

   
   
#model = convNet()


model = torchvision.models.resnet152(pretrained=False)
weights = torch.load(modelPath)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs,5)
model.load_state_dict(weights)
#model = EfficientNet.from_pretrained('efficientnet-b5')
#model = torchvision.models.densenet121(pretrained=True)
#model = pretrainedmodels.se_resnext101_32x4d()

#ct = 0
#for child in model.children():
 #   ct += 1
  #  if ct < 5:
   #     for param in child.parameters():
    #        param.requires_grad = False



#num_ftrs = model.last_linear.in_features
#model.last_linear = nn.Linear(num_ftrs, 5)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=3e-4, momentum=0.9)
#optimizer = optim.Adam(model.parameters(), lr= 3e-4)
# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
exp_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

model = train_model(model, criterion, optimizer, exp_lr_scheduler,
                       num_epochs=25)

torch.save(model.state_dict(),  os.path.join(os.path.expanduser('~'), 'Documents', 
                            'Datasets','aptos2019-blindness-detection','FinalResnet152_0.pt'))
