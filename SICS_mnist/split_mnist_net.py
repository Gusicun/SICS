import torchvision.datasets as datasets
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import os.path
from net import CNetFeatures, CNet
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
import pickle
import torch.nn.functional as F
import sys
from tqdm import tqdm
from collections import Counter
import argparse
import glob

def set_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def data_concat(path, ratio):
    pkl_files = glob.glob(path + "*.pkl")
    data = []
    for file in pkl_files:
        with open(file, 'rb') as f:
            pkl = pickle.load(f)
            data.extend(pkl)

    with open(str(ratio)+"%"+'.pkl', 'wb') as f:
        pickle.dump(data, f)
    print(data)

def get_ratio_data(dataList, ratio):
    index = 0
    result = []
    print('we now choose {} sample from this class'.format(ratio))
    for i in range(len(dataList)):
        if index == ratio:
            break
        result.append(dataList[i][0])
        index += 1

    return result

def get_unsim(dataList):
    transitionList1 = []
    for i in range(0, int(len(dataList) / 2), 1):
        transitionList1.append(dataList[i][0])  # 标签1
        transitionList1.append(dataList[i][1])  # 标签2
    unsimList = sorted(dict(Counter(transitionList1)).items(), key=lambda x: x[1], reverse=True)

    return unsimList

def calculate_sim(targetList):
    resultList = []
    for i in range(0, len(targetList) - 1, 1):
        for j in range(i + 1, len(targetList), 1):
            transitionList = [targetList[i][1], targetList[j][1]]
            x = torch.tensor(targetList[i][0])
            x = F.normalize(x, p=2, dim=0)
            y = torch.tensor(targetList[j][0])
            y = F.normalize(y, p=2, dim=0)
            dists = torch.cosine_similarity(x, y, dim=0)
            #dists = torch.cdist(x.reshape(1, -1), y.reshape(1, -1), p=2)
            dists = dists.squeeze().item()
            transitionList.append(dists)
            resultList.append(transitionList)
    print('start sorting')
    resultList.sort(key=lambda t: t[2])
    print('end sorting')
    return resultList

def feature_model():

    cnet_model = CNet().cuda()
    cnet_model.load_state_dict(torch.load("net_epoch10.pth"))
    cnet_model.eval()
    model = CNetFeatures().cuda()
    model.cnet.load_state_dict(cnet_model.state_dict(), strict=False)

    return model

def resnet_feature(class_dataloader):
    sample_num = 0
    result = []
    class_dataloader = tqdm(class_dataloader, file=sys.stdout)
    for step, (data, indice) in enumerate(class_dataloader):
        image, label =data
        #image = image.cuda()
        sample_num += image.shape[0]
        model = feature_model()
        model.eval()
        with torch.no_grad():
            y = model(image.cuda())
        r = list(zip(y.cpu().data.numpy() , indice.cpu().data.numpy()))
        result.append([list(x) for x in r])
        class_dataloader.desc = "[sample {}] done".format(sample_num)
    flat_result = [element for sublist in result for element in sublist]
    return flat_result

def main(args):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load CIFAR-100 dataset
    train_dataset = datasets.MNIST(root='./MNIST', train=True, download=True, transform = transform)
    test_dataset = datasets.MNIST(root='./MNIST', train=False, download=True, transform = transform)
    # Define class labels
    class_labels = train_dataset.classes

    # Split CIFAR-100 dataset by classes
    class_datasets = {}
    class_result = {}
    ratio = [1,2,5,10,20]
    #weight =[4, 10, 18, 7, 12, 7, 7, 13, 11, 17]#这个是epoch50，一共89 #[5,6,12,6,11,7,6,8,23,6]#这个是训练前10个epoch得到的,一共93
    for r in ratio:
        path = './weight_net/'
        folder = str(int(r)) + "%/"
        for i, label in enumerate(class_labels):
            # Get indices of samples with the current label
            indices = [j for j, (data, target) in enumerate(train_dataset) if target == i]
            # Create a subset of CIFAR-100 with samples with the current label
            class_datasets[label] = Subset(train_dataset, indices)
            print(label)
            class_dataloader = DataLoader(list(zip(class_datasets[label], indices)), batch_size=50, shuffle=False,
                                          num_workers=2)
            class_result = resnet_feature(class_dataloader)
            result_list = calculate_sim(class_result)
            print(result_list)
            unsim_list = get_unsim(result_list)
            num = r * 600
            ratio_list = get_ratio_data(unsim_list, int( num * weight[i] / 89))
            #path = './whole_net/'
            #folder = str(int(r)) + "%/"
            with open(path + folder + str(i) + '.pkl', 'wb') as f:
                test_data = pickle.dump(ratio_list, f)
            f.close()
            print('class {} is done'.format(label))
        data_concat(path + folder, r)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 设置基础超参数
    parser.add_argument('--ratio', type=float, default=0.05)
    parser.add_argument('--percent', type=float, default=5)
    # 模型其他参数
    parser.add_argument('--model-name', default='ResNet18', help='create model name')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args(args=[])

    main(opt)


