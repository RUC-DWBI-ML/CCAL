import os
import numpy as np
import torch
import math
import random
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
from utils.utils import set_random_seed
from getitem import Cifar10_train, Cifar10_test,Cifar100_train, Cifar100_test


DATA_PATH = '~/data/'
IMAGENET_PATH = '~/data/ImageNet'


CIFAR10_SUPERCLASS = list(range(10))  # one class
IMAGENET_SUPERCLASS = list(range(30))  # one class

CIFAR100_SUPERCLASS = [
    [4, 31, 55, 72, 95],#1
    [1, 33, 67, 73, 91],#2
    [54, 62, 70, 82, 92],#3
    [9, 10, 16, 29, 61],#4
    [0, 51, 53, 57, 83],#5
    [22, 25, 40, 86, 87],#6
    [5, 20, 26, 84, 94],#7
    [6, 7, 14, 18, 24],#8
    [3, 42, 43, 88, 97],#9
    [12, 17, 38, 68, 76],#10
    [23, 34, 49, 60, 71],#11
    [15, 19, 21, 32, 39],#12
    [35, 63, 64, 66, 75],#13
    [27, 45, 77, 79, 99],#14
    [2, 11, 36, 46, 98],#15
    [28, 30, 44, 78, 93],#16
    [37, 50, 65, 74, 80],#17
    [47, 52, 56, 59, 96],#18
    [8, 13, 48, 58, 90],#19
    [41, 69, 81, 85, 89],#20
]


class MultiDataTransform(object):
    def __init__(self, transform):
        self.transform1 = transform
        self.transform2 = transform

    def __call__(self, sample):
        x1 = self.transform1(sample)
        x2 = self.transform2(sample)
        return x1, x2


class MultiDataTransformList(object):
    def __init__(self, transform, clean_trasform, sample_num):
        self.transform = transform
        self.clean_transform = clean_trasform
        self.sample_num = sample_num

    def __call__(self, sample):
        set_random_seed(0)

        sample_list = []
        for i in range(self.sample_num):
            sample_list.append(self.transform(sample))

        return sample_list, self.clean_transform(sample)


def get_transform(image_size=None):
    # Note: data augmentation is implemented in the layers
    # Hence, we only define the identity transformation here
    if image_size:  # use pre-specified image size
        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
    else:  # use default image size
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = transforms.ToTensor()

    return train_transform, test_transform


def get_subset_with_len(dataset, length, shuffle=False):
    set_random_seed(0)
    dataset_size = len(dataset)

    index = np.arange(dataset_size)
    if shuffle:
        np.random.shuffle(index)

    index = torch.from_numpy(index[0:length])
    subset = Subset(dataset, index)

    assert len(subset) == length

    return subset


def get_transform_imagenet():

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_transform = MultiDataTransform(train_transform)

    return train_transform, test_transform


def get_dataset(P, dataset, test_only=False, image_size=None, download=True, eval=False):
    train_transform, test_transform = get_transform(image_size=image_size)
    if dataset == 'cifar10':
        image_size = (32, 32, 3)
        n_classes = 10
        train_set = Cifar10_train(DATA_PATH,download,train_transform)
        test_set = Cifar10_test(DATA_PATH, download, test_transform)

    elif dataset == 'cifar100':
        image_size = (32, 32, 3)
        n_classes = 100
        train_set = Cifar100_train(DATA_PATH, download, train_transform)
        test_set = Cifar100_test(DATA_PATH, download, test_transform)
    else:
        raise NotImplementedError()

    if test_only:
        return test_set
    else:
        return train_set, test_set, image_size, n_classes


def get_superclass_list(dataset):
    if dataset == 'cifar10':
        return CIFAR10_SUPERCLASS
    elif dataset == 'cifar100':
        return CIFAR100_SUPERCLASS
    elif dataset == 'imagenet':
        return IMAGENET_SUPERCLASS
    else:
        raise NotImplementedError()


def get_subclass_dataset(dataset, classes):
    if not isinstance(classes, list):
        classes = [classes]

    indices = []
    for idx, tgt in enumerate(dataset.targets):
        if tgt in classes:
            indices.append(idx)

    dataset = Subset(dataset, indices)
    return dataset

def get_sub_labeled_dataset(dataset, classes,select_L_index,select_O_index,query_index,query_label,budget, initial= False):
    if initial:
        labeled_index = [dataset[i][2] for i in range(len(dataset)) if dataset[i][1] in classes]
        set_random_seed(0)
        initial_indices = random.sample(labeled_index, budget)
        dataset_L = Subset(dataset, initial_indices)
        return dataset_L, initial_indices

    else:
        labeled_index, after_label = [], []
        others_index, others_label = [], []
        query_index, query_label = list(query_index), list(query_label)
        for i in list(query_label):
            if i in classes:
                labeled_index.append(query_index[i])
                after_label.append(i)
            else:
                others_index.append(query_index[i])
                others_label.append(i)

        select_L_index = select_L_index + labeled_index
        select_O_index = select_O_index + others_index

        dataset_L = Subset(dataset, select_L_index)
        dataset_O = Subset(dataset, select_O_index)

        return dataset_L, dataset_O, select_L_index, select_O_index


def get_sub_test_dataset(dataset, classes):
    labeled_index = [dataset[i][2] for i in range(len(dataset)) if dataset[i][1] in classes]
    random.shuffle(labeled_index)
    dataset_test = Subset(dataset, labeled_index)
    return dataset_test, labeled_index


def get_sub_unlabeled_dataset(dataset, select_L_index,select_O_index, target_list, num_images):
    all_index = set(np.arange(num_images))
    select_index = select_L_index + select_O_index
    unlabeled_indices = list(np.setdiff1d(list(all_index),select_index))  # find indices which is in all_indices but not in current_indices

    unlabeled_L_index = []
    unlabeled_O_index = []
    for i in unlabeled_indices:
        if dataset[i][1] in target_list:
            unlabeled_L_index.append(i)
        else:
            unlabeled_O_index.append(i)
    datasey_UL = Subset(dataset, unlabeled_L_index)
    datasey_UO = Subset(dataset, unlabeled_O_index)
    dataset_U = Subset(dataset, unlabeled_indices)

    return dataset_U, datasey_UL, datasey_UO, unlabeled_indices, unlabeled_L_index, unlabeled_O_index



def get_mismatch_unlabeled_dataset(dataset, select_L_index, target_list,mismatch, num_images):

    all_index = set(np.arange(num_images))
    unlabeled_indices = list(np.setdiff1d(list(all_index),select_L_index))  # find indices which is in all_indices but not in current_indices

    unlabeled_L_index = []
    unlabeled_O_index = []
    for i in unlabeled_indices:
        if dataset[i][1] in target_list:
            unlabeled_L_index.append(i)
        else:
            unlabeled_O_index.append(i)

    target_number = len(unlabeled_L_index)
    others_number = math.ceil((mismatch*target_number)/(1-mismatch))

    set_random_seed(0)
    select_O_index = random.sample(unlabeled_O_index, others_number)
    unlabeled_index = unlabeled_L_index + select_O_index
    dataset_U = Subset(dataset, unlabeled_index)

    return dataset_U,unlabeled_index



def get_mismatch_contrast_dataset(dataset, select_L_index, target_list,mismatch, num_images):
    all_index = set(np.arange(num_images))
    unlabeled_indices = list(np.setdiff1d(list(all_index),select_L_index))  # find indices which is in all_indices but not in current_indices

    unlabeled_L_index = []
    unlabeled_O_index = []
    for i in unlabeled_indices:
        if dataset[i][1] in target_list:
            unlabeled_L_index.append(i)
        else:
            unlabeled_O_index.append(i)

    target_number = len(unlabeled_L_index)
    others_number = math.ceil((mismatch*target_number)/(1-mismatch))

    set_random_seed(0)
    select_O_index = random.sample(unlabeled_O_index, others_number)
    unlabeled_index = unlabeled_L_index + select_O_index
    contrast_index = unlabeled_index + select_L_index

    set_random_seed(0)
    random.shuffle(contrast_index)
    dataset_contrast = Subset(dataset, contrast_index)

    return dataset_contrast,contrast_index




def get_simclr_eval_transform_imagenet(sample_num, resize_factor, resize_fix):

    resize_scale = (resize_factor, 1.0)  # resize scaling factor
    if resize_fix:  # if resize_fix is True, use same scale
        resize_scale = (resize_factor, resize_factor)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=resize_scale),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    clean_trasform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    transform = MultiDataTransformList(transform, clean_trasform, sample_num)

    return transform, transform


def get_label_index(dataset, L_index,args):
    label_i_index = [[] for i in range(len(args.target_list))]
    for i in L_index:
        for k in range(len(args.target_list)):
            if dataset[i][1] == args.target_list[k]:
                label_i_index[k].append(i)
    return label_i_index