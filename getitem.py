from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy
import os
import numpy as np
import random
import torch
import argument as arg

class Cifar10_train(Dataset):
    def __init__(self, path,download,trans):
        self.cifar10 = datasets.CIFAR10(path, train=True, download=download, transform=trans)

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)

        data, target = self.cifar10[index]

        return data,target,index

    def __len__(self):
        return len(self.cifar10)

class Cifar10_test(Dataset):
    def __init__(self, path,download,trans):
        self.cifar10 = datasets.CIFAR10(path, train=False, download=download, transform=trans)

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)

        data, target = self.cifar10[index]

        return data,target,index

    def __len__(self):
        return len(self.cifar10)



class Cifar100_train(Dataset):
    def __init__(self, path,download,trans):
        self.cifar100 = datasets.CIFAR100(path, train=True, download=download, transform=trans)

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)

        data, target = self.cifar100[index]

        return data,target,index

    def __len__(self):
        return len(self.cifar100)

class Cifar100_test(Dataset):
    def __init__(self, path,download,trans):
        self.cifar100 = datasets.CIFAR100(path, train=False, download=download, transform=trans)

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)

        data, target = self.cifar100[index]

        return data,target,index

    def __len__(self):
        return len(self.cifar100)
