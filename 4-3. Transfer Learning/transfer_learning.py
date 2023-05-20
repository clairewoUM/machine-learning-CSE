import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms


def hello():
    print("Hello from transfer_learning.py!")

def train_model(device, dataloaders, dataset_sizes, model, criterion,
                optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # If there is no training happening
    if num_epochs == 0:
        model.eval()
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # statistics
            running_corrects += torch.sum(preds == labels.data)

        best_acc = running_corrects.double() / dataset_sizes['val']

    # Training for num_epochs steps
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    loss = None
                    preds = None
                    """
                    Inputs:
                    - inputs : tensor (N x C x H x W)                                                
                    - labels : tensor (N)                                                           
                    Outputs:                                                                         
                    - preds : int tensor (N)
                    - loss : torch scalar

                    outputs = model(inputs)
                    _, preds = torch.max(outputs,1)
                    loss = criterion(outputs,labels)                    
                    """
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

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    if num_epochs > 0:
        model.load_state_dict(best_model_wts)
    return model


def visualize_model(device, dataloaders, model, class_names, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    images = []
    captions = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = None
            # Perform feedforward operation using model,
            # and get the labels using torch.max
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                images.append(inputs.cpu().data[j])
                captions.append(f'{class_names[labels.cpu()[j].item()]}\npredicted: {class_names[preds[j]]}')
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return images, captions
        model.train(mode=was_training)
        return images, captions

def finetune(device, dataloaders, dataset_sizes, class_names, num_epochs=10):

    # This is a pretrained Resnet 18 network that we are going to finetune.
    model_ft = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Number of features in FC layer model
    num_ft = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ft,2)
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(),lr=0.001,momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # Show the performance with the pretrained-model, not finetuned yet
    print('Performance of pre-trained model without finetuning')
    _ = train_model(device, dataloaders, dataset_sizes, model_ft,
                    criterion, optimizer_ft, exp_lr_scheduler, num_epochs=0)

    # Finetune the model for 25 epoches
    print('Finetune the model')
    model_ft = train_model(device, dataloaders, dataset_sizes, model_ft,
                           criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)

    return model_ft

def freeze(device, dataloaders, dataset_sizes, class_names, num_epochs=10):
    model_conv = torchvision.models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in model_conv.parameters():
        param.requires_grad = False

    # Replace last layer in with a linear layer having 2 output classes          
    # Parameters of newly constructed modules have requires_grad=True by default       
    num_ft = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ft,2)
    model_conv = model_conv.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_conv = optim.SGD(model_conv.parameters(), lr = 0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    print('Performance of pre-trained model without finetuning')
    _ = train_model(device, dataloaders, dataset_sizes, model_conv, criterion,
                    optimizer_conv, exp_lr_scheduler, num_epochs=0)

    print('\nFinetune the model')
    model_conv = train_model(device, dataloaders, dataset_sizes, model_conv,
                             criterion, optimizer_conv, exp_lr_scheduler, num_epochs=num_epochs)

    return model_conv
