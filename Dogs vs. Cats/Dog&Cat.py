# -*- coding: utf-8 -*-
# @Time    : 2021/1/25 15:51
# @Author  : Yike Cheng
# @FileName: Dog&Cat.py
# @Software: PyCharm
# coding=utf-8
import shutil
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms, datasets, models
import pandas as pd

# 随机种子设置
random_state = 42
np.random.seed(random_state)
original_dataset_dir = 'F:\\kk\\Dogs vs. Cats\\data_2\\train'
base_dir = 'F:\\kk\\Dogs vs. Cats\\data_2\\train'
total_num = int(len(os.listdir(original_dataset_dir)) / 2)

# 返回一个打乱的序号列表
def shuffle():
    random_idx = np.array(range(total_num))
    np.random.shuffle(random_idx)
    return random_idx

# 创建待处理的数据集目录
def mkdir():
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

# 训练集、测试集的划分
def data_partitioning(random_idx):
    sub_dirs = ['train', 'test']
    animals = ['cats', 'dogs']
    train_idx = random_idx[:int(total_num * 0.9)]
    test_idx = random_idx[int(total_num * 0.9):]
    numbers = [train_idx, test_idx]
    for idx, sub_dir in enumerate(sub_dirs):
        dir = os.path.join(base_dir, sub_dir)
        if not os.path.exists(dir):
            os.mkdir(dir)
        for animal in animals:
            animal_dir = os.path.join(dir, animal)  #
            if not os.path.exists(animal_dir):
                os.mkdir(animal_dir)
            fnames = [animal[:-1] + '.{}.jpg'.format(i) for i in numbers[idx]]
            for fname in fnames:
                src = os.path.join(original_dataset_dir, fname)
                dst = os.path.join(animal_dir, fname)
                shutil.copyfile(src, dst)

            # 验证训练集、验证集、测试集的划分的照片数目
            print(animal_dir + ' total images : %d' % (len(os.listdir(animal_dir))))
        # coding=utf-8

# 配置参数
epochs = 10 # 训练次数
batch_size = 4  # 批处理大小
num_workers = 0  # 多线程的数目
use_gpu = torch.cuda.is_available()
def set():
    random_state = 1
    torch.manual_seed(random_state)  # 设置随机数种子，确保结果可重复
    torch.cuda.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    np.random.seed(random_state)
    # random.seed(random_state)

    # PATH='E:\python ese\c_d\model.pt'
    # 对加载的图像作归一化处理， 并裁剪为[224x224x3]大小的图像
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root='F:\\kk\\Dogs vs. Cats\\data_2\\train',
                                         transform=data_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers)

    test_dataset = datasets.ImageFolder(root='F:\\kk\\Dogs vs. Cats\\data_2\\test', transform=data_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader,test_loader


# 创建模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 53 * 53, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 16 * 53 * 53)
        x = self.fc(x)
        return x

def train(train_loader,test_loader):
    net = Net()
    # if (os.path.exists('model.pt')):
    #     net = torch.load('model.pt')

    if torch.cuda.is_available():
        net = net.cuda()
    print(net)

    # 定义loss和optimizer
    cirterion = nn.CrossEntropyLoss() #该交叉熵自带softmax效果
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9) #weight_decay参数用来正则化

    for epoch in range(epochs):
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, train_labels = data
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(train_labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(train_labels)
            optimizer.zero_grad()
            outputs = net(inputs)
            _, train_predicted = torch.max(outputs.data, 1) # 这里返回每一行的最大值以及它的索引
            train_correct += (train_predicted == labels.data).sum()
            loss = cirterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_total += train_labels.size(0)

        print('train %d epoch loss: %.3f  acc: %.3f ' % (
            epoch + 1, running_loss / train_total, 100 * train_correct / train_total))
        # 模型测试
        correct = 0
        test_loss = 0.0
        test_total = 0
        test_total = 0
        net.eval()
        for data in test_loader:
            images, labels = data
            if use_gpu:
                images, labels = Variable(images.cuda()), Variable(labels.cuda())
            else:
                images, labels = Variable(images), Variable(labels)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = cirterion(outputs, labels)
            test_loss += loss.item()
            test_total += labels.size(0)
            correct += (predicted == labels.data).sum()

        print('test  %d epoch loss: %.3f  acc: %.3f ' % (epoch + 1, test_loss / test_total, 100 * correct / test_total))

    torch.save(net, 'model.pt')

def test_data():
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(root='F:\\kk\\Dogs vs. Cats\\test',
                                         transform=data_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset)
    net = torch.load('model.pt')
    net = net.cuda()
    output = []
    for data in test_loader:
        images = data[0]
        if use_gpu:
            images = Variable(images.cuda())
        else:
            images = Variable(images)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        output.append(predicted.item())
    dataframe = pd.DataFrame({'id': [i for i in range(1, 12501)],'label': output})
    dataframe.to_csv("submission.csv", index=False, sep=',')
# data_partitioning(shuffle())
train_loader, validate_loader = set()
train(train_loader, validate_loader)
test_data()