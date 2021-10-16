from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy
import random


import os
import numpy as np
import random
import torch
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']= str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_torch()


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


def cifar10_transformer(x):
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(x)

def cifar10_test_transformer(x):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(x)

def compose_tensor(x):
    compose = transforms.Compose([
        transforms.ToTensor(),
    ])
    return compose(x)






class Cifar10(Dataset):
    def __init__(self, path):
        self.cifar10 = datasets.CIFAR10(root=path,
                                        download=True,
                                        train=True,
                                        transform=None)

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)



        data_original, target = self.cifar10[index]
        data_original_f = compose_tensor(data_original)
        data_trans = cifar10_transformer(data_original)

        return data_trans,target,index, data_original_f

    def __len__(self):
        return len(self.cifar10)


class Cifar10_test(Dataset):
    def __init__(self, path):
        self.cifar10_test = datasets.CIFAR10(root=path,
                                        download=True,
                                        train=False,
                                        transform=None)

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)

        data_original, target = self.cifar10_test[index]
        data_original_f = compose_tensor(data_original)
        data_trans = cifar10_test_transformer(data_original)

        return data_trans, target, index, data_original_f

    def __len__(self):
        return len(self.cifar10_test)



class Cifar100(Dataset):
    def __init__(self, path):
        self.cifar100 = datasets.CIFAR100(root=path,
                                        download=True,
                                        train=True,
                                        transform=None)

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)



        data_original, target = self.cifar100[index]
        data_original_f = compose_tensor(data_original)
        data_trans = cifar10_transformer(data_original)

        return data_trans,target,index, data_original_f

    def __len__(self):
        return len(self.cifar100)


class Cifar100_test(Dataset):
    def __init__(self, path):
        self.cifar100_test = datasets.CIFAR100(root=path,
                                        download=True,
                                        train=False,
                                        transform=None)

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)

        data_original, target = self.cifar100_test[index]
        data_original_f = compose_tensor(data_original)
        data_trans = cifar10_test_transformer(data_original)

        return data_trans, target, index, data_original_f

    def __len__(self):
        return len(self.cifar100_test)




def get_labels(labels,args):
    labels_new = torch.ones_like(labels)
    for i in range(len(labels)):
        for k in range(len(args.target_list)):
            if labels[i] == args.target_list[k]:
                labels_new[i] = k

    return labels_new