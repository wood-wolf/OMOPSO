# -*- 编码格式：utf-8 -*-
# 作者：常冥
# 创建时间：2022/5/24
import torchvision
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from thop import profile
import numpy as np


def mocnn(growthRate, blockConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 模型导入blockConfig=(6,12,24,16);growthRate=[32,32,32,32]
    densenet121 = torchvision.models.DenseNet(growth_rate=growthRate, num_classes=10, block_config=blockConfig)
    # 数据集获取
    train_data = torchvision.datasets.CIFAR10("./CIFAR_Dataset", train=True, download=False,
                                              transform=torchvision.transforms.ToTensor())
    test_data = torchvision.datasets.CIFAR10("./CIFAR_Dataset", train=False, download=False,
                                             transform=torchvision.transforms.ToTensor())
    test_data_size = len(test_data)
    # 训练集和测试集转换格式
    train_loader = DataLoader(dataset=train_data, batch_size=128)
    test_loader = DataLoader(dataset=test_data, batch_size=128)

    densenet121 = densenet121.to(device)  # 模型调用GPU

    optimizer = torch.optim.Adam(densenet121.parameters())  # 优化参数均采用默认值
    # 损失函数调用GPU
    loss_fn = nn.CrossEntropyLoss().to(device)
    ave_acc = 0  # 测试集平均精度
    # 设置网络参数
    # 训练的轮数
    epoch = 1
    for i in range(epoch):
        print("------Desnet121-CIFAR10,第{}轮训练开始!------".format(i + 1))

        # 训练步骤开始
        densenet121.train()
        for data in tqdm(train_loader, desc='Desnet-121 模型训练中：'):
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.to(device)
                targets = targets.to(device)
            output1 = densenet121(imgs)
            loss = loss_fn(output1, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 测试步骤开始
        densenet121.eval()
        total_test_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for data in tqdm(test_loader, desc='Desnet-121 模型测试中：'):
                imgs, targets = data

                imgs = imgs.to(device)  # 图片调用GPU
                targets = targets.to(device)  # 图片调用GPU
                output1 = densenet121(imgs)
                loss = loss_fn(output1, targets)
                total_test_loss = total_test_loss + loss.item()
                # 求出预测准确的数量
                accuracy = (output1.argmax(1) == targets).sum()
                total_accuracy = total_accuracy + accuracy
        ave_acc_curr = total_accuracy / test_data_size  # 当前代的精确度
        # 确定最高精度的模型
        if ave_acc < ave_acc_curr:
            ave_acc = ave_acc_curr
    # FLOPs计算
    input1 = torch.randn(128, 3, 32, 32).to(device)  # 128的batch size
    flops, params = profile(densenet121, inputs=(input1,))
    return np.float64(ave_acc), np.float64(flops)
