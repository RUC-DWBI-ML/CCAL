

# number and network

from sklearn.metrics import accuracy_score
import os, sys

sys.path.append(os.getcwd())

import matplotlib

matplotlib.use('Agg')


from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import os
import numpy as np
import random
import torch
from utils import *
from custom_datasets import get_labels




def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']= str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class Solver:
    def __init__(self, args, dataloader):
        self.args = args
        self.dataloader = dataloader

        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def read_data(self, dataloader, labels=True):
        if labels:
            while True:
                for img, label, index, original in dataloader:
                    yield img, label, index, original
        else:
            while True:
                for img, _, _, original in dataloader:
                    yield img, original


    def train_classifiers(self, classifiers_ours, labeled_dataloader, test_dataloader_labeled,
                          select='ours'):

        seed_torch()
        print("--------------------{}-------------".format(select))
        train_epochs = 100
        train_iterations = len(labeled_dataloader) * train_epochs
        labeled_data = self.read_data(labeled_dataloader)
        accuracy_best = 0
        accuracy_epochs = 0

        lr = 5e-4
        optim_classifier_ours = optim.Adam(classifiers_ours.parameters(), lr=lr)  # , lr= 5e-5

        classifiers_ours.train()

        if self.args.cuda:
            classifiers_ours = classifiers_ours.cuda()

        # change_lr_iter = train_iterations // 25
        # change_lr_iter = 100

        # for iter_count in range(train_iterations):#train_iterations
        for idx, iter_count in enumerate(tqdm(range(train_iterations))):  # train_iterations

            labeled_imgs, labels, index, trans_imgs = next(labeled_data)
            labeled_imgs = labeled_imgs.type(torch.FloatTensor)
            labels_new = get_labels(labels, self.args)

            if labeled_imgs.size(0) < 2:
                labeled_imgs, labels, index, trainfor_imgs = next(labeled_data)
                labeled_imgs = labeled_imgs.type(torch.FloatTensor)
                labels_new = get_labels(labels, self.args)

            if self.args.cuda:
                labeled_imgs = labeled_imgs.cuda()
                labels = labels.cuda()
                labels_new = labels_new.cuda()

            # classifier step

            labeled_preds_logits = classifiers_ours(labeled_imgs)
            labeled_preds_softmax = F.softmax(labeled_preds_logits, dim=1)

            task_loss = self.ce_loss(labeled_preds_logits, labels_new)
            optim_classifier_ours.zero_grad()
            task_loss.backward()
            optim_classifier_ours.step()

            if (iter_count + 1) % len(labeled_dataloader) == 0:

                accuracy_test = self.test_C_extra(classifiers_ours, test_dataloader_labeled, what='train_C',
                                                  select=select)
                if accuracy_test > accuracy_best:
                    accuracy_best = accuracy_test
                    accuracy_epochs = (iter_count + 1) / len(labeled_dataloader)

            if iter_count % 100 == 0:
                print('Current training iteration: {}'.format(iter_count))
                print('Current task model loss: {:.4f}'.format(task_loss.item()))

            if (iter_count < 5) or (iter_count % 50 == 49):
                print("classifier loss is:",task_loss.item())

        accuracy_train = self.test_C_extra(classifiers_ours, labeled_dataloader, what='train_C', select=select)
        accuracy_test = self.test_C_extra(classifiers_ours, test_dataloader_labeled, what='test_C', select=select)
        return classifiers_ours, accuracy_test, accuracy_best, accuracy_epochs

    def test_C_extra(self, classifier, test_dataloader_labeled, what='train_C', select='ours'):

        classifier.eval()

        # ************ test the performance of the classifier-------------------------------
        if what == 'train_C' or what == 'test_C':

            total_C, correct_C = 0, 0
            for id_l, (imgs, labels, indices, trans) in enumerate(tqdm(test_dataloader_labeled)):
                imgs = imgs.type(torch.FloatTensor)
                if self.args.cuda:
                    imgs = imgs.cuda()

                class_preds = classifier(imgs)
                class_preds_s = F.softmax(class_preds, dim=1)
                labels_new = get_labels(labels, self.args)

                preds = torch.argmax(class_preds_s, dim=1).cpu().numpy()
                correct_C += accuracy_score(labels_new, preds, normalize=False)
                total_C += imgs.size(0)
            accuracy = correct_C / total_C * 100

            if what == 'test_C':

                if select == 'ours':
                    print("ours test correct_C", correct_C)
                    print("ours test total_C", total_C)
                    print("ours test class accuracy", correct_C / total_C * 100)

                elif select == 'random':
                    print("random_test correct_C", correct_C)
                    print("random_test total_C", total_C)
                    print("random_test class accuracy", correct_C / total_C * 100)

                elif select == 'shann':
                    print("shann_test correct_C", correct_C)
                    print("shann_test total_C", total_C)
                    print("shann_test class accuracy", correct_C / total_C * 100)



            elif what == 'train_C':

                if select == 'ours':
                    print("ours train correct_C", correct_C)
                    print("ours train total_C", total_C)
                    print("ours train class accuracy", correct_C / total_C * 100)

                elif select == 'random':


                    print("random_train correct_C", correct_C)
                    print("random_train total_C", total_C)
                    print("random_train class accuracy", correct_C / total_C * 100)

                elif select == 'shann':
                    print("shann_train correct_C", correct_C)
                    print("shann_train total_C", total_C)
                    print("shann_train class accuracy", correct_C / total_C * 100)

        return accuracy



