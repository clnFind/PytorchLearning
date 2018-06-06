# -*- coding: utf-8 -*-

from __future__ import print_function, division
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import os
import torch
import torchvision

__author__ = 'clnFind'

# 图片处理
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),      # 从原图随机切割一张224 * 224的图像
        transforms.RandomHorizontalFlip(),      # 以0.5的概率水平翻转
        transforms.ToTensor(),                  # 图像转为tensor，在[0，1]范围，主要是除以255
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 数据标准化
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),                 # 重设输入图片的大小：if high > wide, 图像缩小为 (256 * high / wide, 256)
        transforms.CenterCrop(224),             # 从中间切割一张224*224像素的图像
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 加载图片，并转换成tensor数据类型
data_dir = './animal_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
print(image_datasets)

# 多线程迭代器，打乱数据，每批处理4个，四个线程
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=4) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
print("Length........:", dataset_sizes)

# 解析目录获取图片的父目录名（即类型）
class_names = image_datasets['train'].classes
print("Labels........:", class_names)

i = 1
for inputs, labels in dataloaders['train']:

    # wrap time in Variable
    inputs, labels = Variable(inputs), Variable(labels)
    print(inputs, len(inputs))
    print(labels, len(labels))

    if i == 1:
        break


# 设备选择：if GPU可用，选GPU，反之，选CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dd = torch.randn(2, 3)

print(dd)
print(dd.to(device))
