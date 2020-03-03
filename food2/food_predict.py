import os, sys, glob, argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

import time, datetime
import pdb, traceback

import cv2
# import imagehash
from PIL import Image

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b4')

import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from collections import OrderedDict
class QRDataset(Dataset):
    def __init__(self, img_path, transform=None):
        self.img_path = img_path
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        start_time = time.time()
        img = Image.open(self.img_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, torch.from_numpy(np.array(int('PNEUMONIA' in self.img_path[index])))

    def __len__(self):
        return len(self.img_path)


class VisitNet(nn.Module):
    def __init__(self):
        super(VisitNet, self).__init__()
        #model = models.resnext50_32x4d(pretrained=True)
        #model = models.resnext101_32x8d(pretrained=True)
        model = models.wide_resnet50_2(pretrained=True)
        #model = models.resnet152(pretrained=True)
        #model.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 4)
        self.resnet = model
        #model = EfficientNet.from_pretrained('efficientnet-b7')
        #model._fc = nn.Linear(1280, 18)
        #model._fc = nn.Linear(2560, 4)
        #self.resnet = model

    def forward(self, img):
        out = self.resnet(img)
        return out


class VisitNet2(nn.Module):
    def __init__(self):
        super(VisitNet2, self).__init__()
        #model = models.resnext50_32x4d(pretrained=True)
        #model = models.wide_resnet50_2(pretrained=True)
        #model = models.resnet101(pretrained=True)
        #model.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)
        self.resnet = model
        #model = EfficientNet.from_pretrained('efficientnet-b4')
        #model._fc = nn.Linear(1280, 18)
        #model._fc = nn.Linear(1792, 3)
        #self.resnet = model

    def forward(self, img):
        out = self.resnet(img)
        return out


def predict(test_loader, model, tta=5):
    # switch to evaluate mode
    model.eval()

    test_pred_tta = None
    for _ in range(tta):
        test_loader = torch.utils.data.DataLoader(
            QRDataset(test_jpg,
                  transforms.Compose([
                      transforms.Resize([400, 400]),
                      # transforms.Scale(299),
                      # transforms.RandomResizedCrop(224),
                      # transforms.CenterCrop(224),
                      #transforms.RandomHorizontalFlip(),
                      #transforms.ColorJitter(brightness=0.1),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                  ])
                  ), batch_size=128, shuffle=False, num_workers=10, pin_memory=True
        )

        test_pred = []
        with torch.no_grad():
            end = time.time()
            for i, (input, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
                input = input.cuda()
                target = target.cuda()

                # compute output
                output = model(input)
                output = output.data.cpu().numpy()
                #print(output)
                test_pred.append(output)
        test_pred = np.vstack(test_pred)

        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred

    return test_pred_tta


test_jpg = ['./test/{0}.jpg'.format(x) for x in range(0, 856)]
test_jpg = np.array(test_jpg)

test_pred = None


#for model_path in ['./model/invnet_fold0.pt','./model/invnet_fold1.pt','./model/invnet_fold2.pt','./model/invnet_fold3.pt','./model/invnet_fold4.pt']:
#for model_path in ['./resnet_fold0.pt','./resnet_fold1.pt','./resnet_fold2.pt','./resnet_fold3.pt','./resnet_fold4.pt']:
for model_path in ['./0_wide_fold0.pt','./0_wide_fold1.pt','./0_wide_fold2.pt','./0_wide_fold3.pt','./0_wide_fold4.pt']:
#for model_path in ['./wide_fold2.pt']:
#for model_path in ['./model/resnext_fold0.pt','./model/resnext_fold1.pt','./model/resnext_fold2.pt','./model/resnext_fold3.pt','./model/resnext_fold4.pt']:
    test_loader = torch.utils.data.DataLoader(
        QRDataset(test_jpg,
                  transforms.Compose([
                      transforms.Resize([400, 400]),
                      # transforms.Scale(299),
                      # transforms.RandomResizedCrop(224),
                      # transforms.CenterCrop(224),
                      #transforms.RandomHorizontalFlip(),
                      #transforms.ColorJitter(brightness=0.1),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                  ])
                  ), batch_size=128, shuffle=False, num_workers=10, pin_memory=True
    )

    model = VisitNet().cuda()
    model = nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(model_path))
    #model = myOwnLoad(model,torch.load(model_path))
    #model = nn.DataParallel(model).cuda()
    if test_pred is None:
        test_pred = predict(test_loader, model, 1)
    else:
        test_pred += predict(test_loader, model,1)

test_csv = pd.DataFrame()
test_csv[0] = list(range(0, 856))
test_csv[1] = np.argmax(test_pred, 1)
test_csv.to_csv('tmp5.csv', index=None, header=None)
#test_pred.astype(int).iloc[:].to_csv('tmp.csv', index=None, header=None)
