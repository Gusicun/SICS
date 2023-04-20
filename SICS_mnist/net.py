import argparse
from torch import nn as nn
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
import numpy.random as npr
import numpy as np
import sys
from itertools import chain
import copy
import math
from torch.utils.data import Dataset

device = torch.device("cuda")


class CNet(nn.Module):
  def __init__(self):
    super(CNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.conv2_drop = nn.Dropout2d()
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))

    x = F.relu(F.max_pool2d(self.conv2(x), 2))

    x = x.view(-1, 320)
    x = F.relu(self.fc1(x))

    x = self.fc2(x)
    return F.log_softmax(x, dim=1)

class CNetFeatures(nn.Module):
    def __init__(self):
      super(CNetFeatures, self).__init__()
      self.cnet = CNet()
      self.cnet.fc1 = nn.Identity()
      self.cnet.fc2 = nn.Identity()

    def forward(self, x):
      x = self.cnet.conv1(x)
      x = F.relu(F.max_pool2d(x, 2))
      x = self.cnet.conv2(x)
      x = F.relu(F.max_pool2d(x, 2))
      x = x.view(-1,320)
      x = F.relu(self.cnet.fc1(x))
      x = self.cnet.fc2(x)
      return x

