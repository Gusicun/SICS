import datetime
import json
import random
import math
from math import pow, acos, sqrt
from itertools import chain
from collections import Counter
import torch
from torch import nn
from torch.utils.data.dataset import TensorDataset
from torch.autograd import Variable
import torchvision
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.datasets import make_blobs, make_regression, make_classification
from sklearn.model_selection import train_test_split
import numpy as np
from random import sample
import copy
from matplotlib import pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import asyncio
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import os
from PIL import Image
from tqdm import tqdm
import sys
import pickle
import torch
from torch.utils.data import Dataset
import numpy

def train_one_epoch(model, dataset, batch_size, loss_function, optimizer, device, epoch):
    '''
    功能：训练函数
    参数：
    Model:传入的模型
    optimizer:传入的优化器
    dataloader:传输的数据迭代器
    device:传入使用设备
    epoch:计算是第几个循环
    返回：
    当前训练精度,当前训练损失
    '''
    data_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0)
    model.train()

    #loss_function = loss_function

    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num1 = torch.zeros(1).to(device)   # 累计预测正确的样本数


    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        optimizer.zero_grad()

        pred = model(images.to(device))
        _,pred_classes = torch.max(pred.data, 1)
        accu_num1 += torch.eq(pred_classes, labels.to(device)).sum()


        loss = loss_function(pred, labels.to(device))

        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc1: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num1.item() / sample_num,
                                                                              )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        #optimizer.zero_grad()



    return accu_loss.item() / (step + 1), accu_num1.item() / sample_num

@torch.no_grad()
def test(model, dataset, batch_size, loss_function, device, epoch):
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=0)

    model.eval()


    accu_num1 = torch.zeros(1).to(device)   # 累计预测正确的样本数

    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        _,pred_classes = torch.max(pred.data, 1)
        accu_num1 += torch.eq(pred_classes, labels.to(device)).sum()


        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[test epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num1.item() / sample_num,
                                                                               )


    return accu_loss.item() / (step + 1), accu_num1.item() / sample_num