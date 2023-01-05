import torch
import torch.utils.data as data
import argument
from torchvision import datasets, transforms
from datasets import get_dataset, get_superclass_list, get_subclass_dataset
from datasets import get_sub_labeled_dataset, get_sub_test_dataset, get_sub_unlabeled_dataset,get_mismatch_unlabeled_dataset
import models.classifier as C
import torch.optim.lr_scheduler as lr_scheduler
from utils.utils import load_checkpoint
import torch.nn as nn
import torch.optim as optim
import random
from utils.utils import Logger
from utils.utils import save_checkpoint
from utils.utils import save_linear_checkpoint
from evals.eval import test_classifier
import os
import numpy as np
from utils.utils import set_random_seed
from datasets import get_label_index
from evals.eval import eval_unlabeled_detection



def main(args):
    # --------------Set torch device --------------------------------------------------------

    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    args.n_gpus = torch.cuda.device_count()

    if args.n_gpus > 1:
        import apex
        import torch.distributed as dist
        from torch.utils.data.distributed import DistributedSampler

        args.multi_gpu = True
        torch.distributed.init_process_group(
            'nccl',
            init_method='env://',
            world_size=args.n_gpus,
            rank=args.local_rank,
        )
    else:
        args.multi_gpu = False

    args.ood_layer = args.ood_layer[0]### only use one ood_layer while training
    kwargs = {'pin_memory': False, 'num_workers': 4}
    #-----------------end-------------------------------------------------------------



    # --------------Set dataset--------------------------------------------------------

    if args.dataset == 'cifar10':

        train_set, test_set, image_size, n_classes = get_dataset(args, dataset=args.dataset)

        args.image_size = image_size
        args.n_classes = n_classes
        args.num_images = 50000
        args.budget = 1500
        args.initial_budget = 800
        args.num_classes = 10
        args.input_size = 32 * 32 * 3
        args.batch_size_classifier = 32
        args.target_list = [0,1]
        args.untarget_list = [2,3,4,5,6,7,8,9]
        args.target_number = 2


    elif args.dataset == 'cifar100':

        train_set, test_set, image_size, n_classes = get_dataset(args, dataset=args.dataset)

        args.image_size = image_size
        args.n_classes = n_classes
        args.num_images = 50000
        args.budget = 1500
        args.initial_budget = 800
        args.num_classes = 100
        args.input_size = 32 * 32 * 3
        args.batch_size_classifier = 32
        args.target_list = [3, 42, 43, 88, 97,15, 19, 21, 32, 39,35, 63, 64, 66, 75,37, 50, 65, 74, 80]
        args.untarget_list = list(np.setdiff1d(list(range(0,100)), list(args.target_list)))
        args.target_number = 20

    else:
        raise NotImplementedError

    #---------------end---------------------------------------------------------


    #----------------set active learning dataset----------------------------------------

    select_L_index, select_O_index, query_index, query_label = [], [], [], []

    initial_dataset, select_L_index = get_sub_labeled_dataset(train_set, args.target_list, select_L_index,
                                                              select_O_index, query_index, query_label,
                                                              args.initial_budget,
                                                              initial=True)
    initial_test_dataset, test_index = get_sub_test_dataset(test_set, args.target_list)

    #mismatch
    unlabeled_dataset, unlabeled_index = get_mismatch_unlabeled_dataset(train_set, select_L_index, args.target_list, args.mismatch,args.num_images)

    label_i_index = get_label_index(train_set,select_L_index, args)

    sampler_unlabeled = data.sampler.SubsetRandomSampler(unlabeled_index)  # make indices initial to the samples
    U_loader = data.DataLoader(train_set, sampler=sampler_unlabeled,
                                   batch_size=args.test_batch_size, **kwargs)

    sampler_labeled = data.sampler.SubsetRandomSampler(select_L_index)  # make indices initial to the samples
    train_loader = data.DataLoader(train_set, sampler=sampler_labeled,
                                   batch_size=args.test_batch_size, **kwargs)

    label_i_loader = []
    for i in range(len(label_i_index)):
        sampler_label_i = data.sampler.SubsetRandomSampler(label_i_index[i])  # make indices initial to the samples
        label_loader_i = data.DataLoader(train_set, sampler=sampler_label_i,
                                         batch_size=args.test_batch_size, **kwargs)
        label_i_loader.append(label_loader_i)


    #-------------Initialize model------------------------------------------------------

    # anchor transform
    simclr_aug = C.get_simclr_augmentation(args, image_size=args.image_size).to(device)
    # shift transform---------rotation
    args.shift_trans, args.K_shift = C.get_shift_module(args, eval=True)
    args.shift_trans = args.shift_trans.to(device)

    # select feature contrast model
    model_feature = C.get_classifier(args.model, n_classes=args.n_classes).to(device)
    # get linear predict shift
    model_feature = C.get_shift_classifer(model_feature, args.K_shift).to(device)

    # select senmatic contrast model
    model_senmatic = C.get_classifier(args.model, n_classes=args.n_classes).to(device)
    # get linear predict shift
    model_senmatic = C.get_shift_classifer(model_senmatic, 1).to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    # get optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model_feature.parameters(), lr=args.lr_init, momentum=0.9, weight_decay=args.weight_decay)
        lr_decay_gamma = 0.1
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model_feature.parameters(), lr=args.lr_init, betas=(.9, .999), weight_decay=args.weight_decay)
        lr_decay_gamma = 0.3
    else:
        raise NotImplementedError()

    if args.lr_scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif args.lr_scheduler == 'step_decay':
        milestones = [int(0.5 * args.epochs), int(0.75 * args.epochs)]
        scheduler = lr_scheduler.MultiStepLR(optimizer, gamma=lr_decay_gamma, milestones=milestones)
    else:
        raise NotImplementedError()

    if args.resume_path is not None:
        resume = True
        model_state, optim_state, config = load_checkpoint(args.resume_path, mode='last')
        model_feature.load_state_dict(model_state, strict=not args.no_strict)
        optimizer.load_state_dict(optim_state)
        start_epoch = config['epoch']
        best = config['best']
        error = 100.0
    else:
        resume = False
        start_epoch = 1
        best = 100.0
        error = 100.0

    if args.mode == 'eval':
        #----feature contrast model
        assert args.load_feature_path is not None
        checkpoint_feature = torch.load(args.load_feature_path)
        args.no_strict = False
        model_feature.load_state_dict(checkpoint_feature, strict=not args.no_strict)

        # ----senmatic contrast model
        assert args.load_senmatic_path is not None
        checkpoint_senmatic = torch.load(args.load_senmatic_path)
        args.no_strict = False
        model_senmatic.load_state_dict(checkpoint_senmatic, strict=not args.no_strict)


    if args.multi_gpu:
        simclr_aug = apex.parallel.DistributedDataParallel(simclr_aug, delay_allreduce=True)
        model_feature = apex.parallel.convert_syncbn_model(model_feature)
        model_feature = apex.parallel.DistributedDataParallel(model_feature, delay_allreduce=True)
        model_senmatic = apex.parallel.convert_syncbn_model(model_senmatic)
        model_senmatic = apex.parallel.DistributedDataParallel(model_senmatic, delay_allreduce=True)

    #-------------------------------end---------------------------------------------------


    #-------------------------------eval---------------------------------------------------
    print("-----------------eval---------------------")

    args.shift_trans_type = 'rotation'
    args.print_score = True
    args.ood_samples = 10
    args.resize_factor = 0.54
    args.resize_fix = True
    simclr_aug = C.get_simclr_augmentation(args, image_size=args.image_size).to(device)


    split = args.split #query times

    for i in range(split):

        with torch.no_grad():
            query_index, query_label = eval_unlabeled_detection(args, model_feature, model_senmatic, U_loader, train_loader,label_i_loader,simclr_aug=simclr_aug)

        unlabeled_index = list(np.setdiff1d(list(unlabeled_index), list(query_index)))  # find indices which is in all_indices but not in current_indices

        query_L = []
        query_O = []
        label_i_add_index = [[] for i in range(len(args.target_list))]
        query_index = list(query_index)
        query_label = list(query_label)
        for j in range(len(query_label)):
            if query_label[j] in args.target_list:
                query_L.append(query_index[j])
                for k in range(len(args.target_list)):
                    if query_label[j] == args.target_list[k]:
                        label_i_add_index[k].append(query_index[j])
            else:
                query_O.append(query_index[j])


        print("target number is {}, unseen number is{}".format(len(query_L), len(query_O)))
        select_L_index = select_L_index + query_L
        select_O_index = select_O_index + query_O

        select_all_index = select_L_index + select_O_index

        print("labeled index:",select_L_index)
        print("unseen samples index:", select_O_index)

        print("number of labeled samples:{}, number of unseen samples:{}".format(len(select_L_index), len(select_O_index)))
        print("number of unlabeled samples:", len(unlabeled_index))


        for k in range(len(args.target_list)):
            label_i_index[k] += label_i_add_index[k]

        label_i_loader = []
        for i in range(len(label_i_index)):
            sampler_label_i = data.sampler.SubsetRandomSampler(label_i_index[i])  # make indices initial to the samples
            label_loader_i = data.DataLoader(train_set, sampler=sampler_label_i,
                                             batch_size=args.test_batch_size, **kwargs)
            label_i_loader.append(label_loader_i)

        sampler_unlabeled = data.sampler.SubsetRandomSampler(unlabeled_index)  # make indices initial to the samples
        U_loader = data.DataLoader(train_set, sampler=sampler_unlabeled,
                                   batch_size=args.test_batch_size, **kwargs)

        sampler_labeled = data.sampler.SubsetRandomSampler(select_L_index)  # make indices initial to the samples
        train_loader = data.DataLoader(train_set, sampler=sampler_labeled,
                                       batch_size=args.test_batch_size, **kwargs)


    print("***********************************************")


if __name__ == '__main__':
    args = argument.parse_args()
    set_random_seed(0)
    main(args)
