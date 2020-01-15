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
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, models, transforms
import PIL
from torch.utils.data.dataset import Dataset

train_df = pd.read_csv('./train.csv',header=None)
train_df[0] = train_df[0].apply(lambda x: './train/{0}'.format(x) + '.jpg')


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        logpt = F.log_softmax(input, dim=1)
        #logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class QRDataset(Dataset):
    def __init__(self, df, transform=None, cut_ratio=0.2):
        self.df = df
        self.cut_ratio = cut_ratio
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        start_time = time.time()
        img = Image.open(self.df[0].iloc[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        #print(np.array(self.df.iloc[index, 1]))
        return img, torch.from_numpy(np.array(self.df.iloc[index, 1]))

    def __len__(self):
        return len(self.df)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = ""

    def pr2int(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class VisitNet(nn.Module):
    def __init__(self):
        super(VisitNet, self).__init__()

        model = models.wide_resnet50_2(pretrained=True)
        #model.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 5)
        self.resnet = model

        #model = EfficientNet.from_pretrained('efficientnet-b0')
        #model._fc = nn.Linear(1280, 18)
        #self.resnet = model

    def forward(self, img):
        out = self.resnet(img)
        return out


def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':2.2f')
    top5 = AverageMeter('Acc@5', ':2.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        # TODO: this should also be done with the ProgressMeter
        print(losses.avg)
        return losses


def predict(test_loader, model, tta=10):
    # switch to evaluate mode
    model.eval()

    test_pred_tta = None
    for _ in range(tta):
        test_pred = []
        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(test_loader):
                input = input.cuda()
                target = target.cuda()

                # compute output
                output = model(input, path)
                output = output.data.cpu().numpy()

                test_pred.append(output)
        test_pred = np.vstack(test_pred)

        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred

    return test_pred_tta


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    # data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':2.2f')
    top5 = AverageMeter('Acc@5', ':2.2f')
    #progress = ProgressMeter(len(train_loader), batch_time, losses)
    progress = ProgressMeter(len(train_loader), batch_time, losses,top1,top5)
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        #print(len(output))
        #print(len(target))
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            progress.pr2int(i)


skf = KFold(n_splits=5, random_state=233, shuffle=True)
for flod_idx, (train_idx, val_idx) in enumerate(skf.split(train_df[0].values, train_df[0].values)):
    # print(flod_idx, train_idx, val_idx)

    train_loader = torch.utils.data.DataLoader(
        QRDataset(train_df.iloc[train_idx],
                  transforms.Compose([
                      transforms.Resize([299, 299]),
                      transforms.RandomRotation(15),
                      transforms.RandomChoice([transforms.Resize([256, 256]), transforms.CenterCrop([256, 256])]),
                      # transforms.RandomResizedCrop(224),
                      # transforms.Resize([ 256, 256]),
                      # transforms.CenterCrop(224),
                      # transforms.RandomChoice([transforms.RandomHorizontalFlip(),medianBlur]),
                      transforms.ColorJitter(brightness=0.2, contrast=0.2),
                      transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                  ]), 0
                  ), batch_size=128, shuffle=True, num_workers=10, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        QRDataset(train_df.iloc[val_idx],
                  transforms.Compose([
                      transforms.Resize([256, 256]),
                      # transforms.Scale(299),
                      # transforms.RandomResizedCrop(224),
                      # transforms.CenterCrop(224),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                  ]), 0
                  ), batch_size=128, shuffle=False, num_workers=10, pin_memory=True
    )

    model = VisitNet().cuda()
    model = nn.DataParallel(model).cuda()
    # model.load_state_dict(torch.load('resnet18_pretrain_fold0.pt'))
    # model.resnet.fc = nn.Linear(512, 100)
    model = model.cuda()

    # model = nn.DataParallel(model).cuda()
    criterion = FocalLoss(0.5)
    #optimizer = torch.optim.Adam(model.parameters(), 0.001)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.85)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    best_acc = 10.0
    for epoch in range(15):
        scheduler.step()
        print('Epoch: ', epoch)

        train(train_loader, model, criterion, optimizer, epoch)
        val_acc = validate(val_loader, model, criterion)

        if val_acc.avg < best_acc:
            best_acc = val_acc.avg
            torch.save(model.state_dict(), './resnet18_fold{0}.pt'.format(flod_idx))
