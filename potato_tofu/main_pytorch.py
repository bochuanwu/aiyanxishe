# coding=utf-8
# author=yphacker

import os
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim import lr_scheduler

import numpy as np
import matplotlib.pyplot as plt

import torchvision
from torchvision import datasets, models, transforms
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import cv2
import PIL

class MedianBlur:
    """ Perform median filtering to remove salt and pepper noise from image. """
    def __init__(self, ksize):
        assert isinstance(ksize, int) and ksize > 1 and ksize % 2 == 1
        self.ksize = ksize

    def __call__(self, sample):
        return PIL.Image.fromarray(cv2.medianBlur(np.asarray(sample), self.ksize))

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

xent = CrossEntropyLabelSmooth(num_classes=2)
medianBlur = MedianBlur(3)
data_transforms = {
    'train': transforms.Compose([
        transforms.Scale(299),
        transforms.RandomRotation(15),
        transforms.RandomChoice([transforms.RandomResizedCrop(224),transforms.CenterCrop(224)]),
        #transforms.RandomResizedCrop(224),
        #transforms.Resize([ 299, 299]),
        #transforms.CenterCrop(224),
        transforms.RandomChoice([transforms.RandomHorizontalFlip(),medianBlur]),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validation': transforms.Compose([
        # transforms.Resize(256),
        transforms.Scale(299),
        #transforms.RandomResizedCrop(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}



image_datasets = {x: datasets.ImageFolder(x, data_transforms[x])
                  for x in ['train', 'validation']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True)
               for x in ['train', 'validation']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}

class_names = image_datasets['train'].classes

device = torch.device("cuda:0,1" if torch.cuda.is_available() else "cpu")

inputs, classes = next(iter(dataloaders['train']))

out = torchvision.utils.make_grid(inputs)


def train_model(model, criterion, optimizer, scheduler, num_epochs=1):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict)
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    #loss = criterion(outputs, labels)
                    loss = xent(outputs, labels)

                    print(loss.item(),end='')

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)

    torch.save(model, 'model.pkl')

    return model


# def set_parameter_requires_grad(model, feature_extracting):
#    if feature_extracting:
#        for param in model.parameters():
#            param.requires_grad = False
#
# model_ft = models.densenet121(pretrained=True)
# set_parameter_requires_grad(model_ft, True)
# num_ftrs = model_ft.classifier.in_features
# model_ft.classifier = nn.Linear(num_ftrs, 10)

model_ft = models.resnext50_32x4d(pretrained=True)
#model_ft = models.resnet101(pretrained=True)
#model_ft = models.inception_v3(pretrained=True)
#model_ft = models.wide_resnet50_2(pretrained=True) 98.08
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

#optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
optimizer_ft = optim.Adagrad(model_ft.parameters(), lr=0.001)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, 15)
