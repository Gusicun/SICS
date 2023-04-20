import datetime
import json
import random
import math
from math import pow, acos, sqrt
from itertools import chain
from collections import Counter
import torch
import glob
from mydataset import Mydataset,Mydatasetpro
from torch import nn
from net import CNet
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
import argparse
from train_test import train_one_epoch,test
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Subset
from resnet import ResNet18
'''
torch.cuda.empty_cache()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark=False


def set_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
'''
def set_seed(seed=1):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)

def main(args):
    # 设备情况
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 数据转换模式
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # 加载为loader
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size

    # 定义模型及其优化参数


    #加载自定义数据集
    mnist = datasets.MNIST(root='./MNIST/raw', train=True, download=True,transform=transform)
    # 测试数据集
    test_dataset = datasets.MNIST(root='./MNIST/raw', train=False, download=True,
                                  transform=transform)

    set_seed()
    path = './mix/mix_5sim_5unsim_10%.pkl'
    f = open(path, 'rb')
    indices = pickle.load(f)
    f.close()
    train_dataset = Subset(mnist,indices)

    ##模型
    model = CNet().to(device)
    ##优化器（SGD）
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.5)
    loss_function = torch.nn.CrossEntropyLoss()
    '''

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=train_batch_size,
                                              shuffle=True,
                                              num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=test_batch_size,
                                              shuffle=False,
                                              num_workers=0)
    '''


    # 开始训练
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                dataset=train_dataset,
                                                batch_size=train_batch_size,
                                                loss_function = loss_function,
                                                optimizer=optimizer,
                                                device=device,
                                                epoch=epoch)

        # test
        test_loss, test_acc = test(model=model,
                                   dataset=test_dataset,
                                   batch_size=test_batch_size,
                                   loss_function=loss_function,
                                   device=device,
                                   epoch=epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 设置基础超参数
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    # 设置转折点参S数
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--threshold_epoch', type=int, default=None)
    parser.add_argument('--window', type=int, default=3)
    parser.add_argument('--stable_acc', type=float, default=1e-3)
    parser.add_argument('--stable_loss', type=float, default=1e-3)
    parser.add_argument('--threshold_acc', type=float, default=0.85)
    # 模型其他参数
    parser.add_argument('--model-name', default='', help='create model name')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args(args=[])

    main(opt)