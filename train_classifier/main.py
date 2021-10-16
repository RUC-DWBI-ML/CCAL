import torch
import torch.utils.data as data
from custom_datasets import *

from train import Solver
import arguments
from torchvision import datasets, transforms
from ResNet import resnet50,resnet18
from ResNet_1 import resnet18_1
from ResNet_2 import resnet18_2
from ResNet_3 import resnet18_3
from ResNet_4 import resnet18_4
import get_index_20, get_index_40, get_index_60, get_index_80
import numpy as np




def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']= str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def main(args):

    if args.dataset == 'cifar10':

        train_dataset = Cifar10(args.data_path)
        test_dataset = Cifar10_test(args.data_path)

        args.num_images = 50000
        args.budget = 1500
        args.initial_budget = 800
        args.num_classes = 2
        args.input_size = 32 * 32 * 3
        args.batch_size_classifier = 32
        args.target_list = [0,1]
        args.untarget_list = [2,3,4,5,6,7,8,9]
        args.target_number = 2


    elif args.dataset == 'cifar100':

        train_dataset = Cifar100(args.data_path)
        test_dataset = Cifar100_test(args.data_path)

        args.num_images = 50000
        args.budget = 1500
        args.initial_budget = 800
        args.num_classes = 20
        args.input_size = 32 * 32 * 3
        args.batch_size_classifier = 32
        args.target_list = [3, 42, 43, 88, 97, 15, 19, 21, 32, 39, 35, 63, 64, 66, 75, 37, 50, 65, 74, 80]
        args.untarget_list = list(np.setdiff1d(list(range(0, 100)), list(args.target_list)))
        args.target_number = 20
    else:
        raise NotImplementedError



    all_indices = list(set(np.arange(args.num_images)))
    target_list = args.target_list
    labeled_indices = [train_dataset[i][2] for i in range(args.num_images) if train_dataset[i][1] in target_list]


    #-----------get labeled index----------------------
    if args.number == 0:
        seed_torch()
        initial_indices = random.sample(labeled_indices, args.initial_budget)
    else:
        if args.mismatch == 0.2:
            initial_indices = get_index_20.get_indice_labeled(args.number)
        elif args.mismatch == 0.4:
            initial_indices = get_index_40.get_indice_labeled(args.number)
        elif args.mismatch == 0.6:
            initial_indices = get_index_60.get_indice_labeled(args.number)
        elif args.mismatch == 0.8:
            initial_indices = get_index_80.get_indice_labeled(args.number)

    initial_indices = list(set(initial_indices))

    print("initial_indices",initial_indices)


    #-------------------------get labeled dataloader----------------
    seed_torch()
    random.shuffle(initial_indices)
    sampler_labeled = data.sampler.SubsetRandomSampler(initial_indices)  # make indices initial to the samples
    labeled_dataloader_classifier = data.DataLoader(train_dataset, sampler=sampler_labeled,
                                         batch_size=args.batch_size_classifier, drop_last=False)


    args.cuda = args.cuda and torch.cuda.is_available()  # use gpu




    #---------------get test labeled index--------------------------------
    test_labeled_indices = [i for i in range(len(test_dataset)) if test_dataset[i][1] in target_list]
    test_sampler_labeled = data.sampler.SubsetRandomSampler(test_labeled_indices)  # make indices initial to the samples
    test_dataloader_labeled = data.DataLoader(test_dataset, sampler=test_sampler_labeled,
                                         batch_size=args.batch_size, drop_last=False)


    #---------------get all dataloader-------------------------
    train_indices = range(len(train_dataset))
    train_sampler = data.sampler.SubsetRandomSampler(train_indices)  # make indices initial to the samples
    train_dataloader = data.DataLoader(train_dataset, sampler=train_sampler,
                                              batch_size=args.batch_size, drop_last=False)


    solver = Solver(args, train_dataloader)



    accuracy_ours_best_epoch = []
    accuracy_all_ours = []
    accuracy_best = []
    accuracy_best_all = []




    accuracys_ours = []
    accuracy_ours_best = []
    accuracy_i_best = 0
    for i in range(5):
        if i == 0:
            classifiers_ours = resnet18(num_classes=args.num_classes)#_0
        elif i == 1:
            classifiers_ours = resnet18_1(num_classes=args.num_classes)#_1
        elif i == 2:
            classifiers_ours = resnet18_2(num_classes=args.num_classes)#_2
        elif i == 3:
            classifiers_ours = resnet18_3(num_classes=args.num_classes)#_3
        elif i == 4:
            classifiers_ours = resnet18_4(num_classes=args.num_classes)#_4
        classifiers_ours,accuracy_ours,accuracys_ours_best,accuracys_ours_best_epoch = solver.train_classifiers(classifiers_ours,labeled_dataloader_classifier,test_dataloader_labeled,select='ours')
        accuracys_ours.append(accuracy_ours)
        if accuracy_ours > accuracy_i_best:
            torch.save(classifiers_ours, 'model_classifier/model_best.pkl')
            accuracy_i_best = accuracy_ours

        print("accuracys_ccal:",accuracys_ours)

        accuracy_ours_best.append(accuracys_ours_best)
        accuracy_ours_best_epoch.append(accuracys_ours_best_epoch)

    accuracy_all_ours.append(np.mean(accuracys_ours))
    accuracy_best.append(np.mean(accuracy_ours_best))
    accuracy_best_all.append(max(accuracy_ours_best))
    print("accuracy_all_ccal", accuracy_all_ours)


    print("ccal accuracy---mean:", np.mean(accuracys_ours))
    print("ccal accuracy---std:", np.std(accuracys_ours,ddof=1))


    print('-----ccal is finish------')


    torch.save(classifiers_ours, os.path.join(args.out_path, args.classifier_name))


if __name__ == '__main__':
    args = arguments.get_args()
    seed_torch()
    main(args)