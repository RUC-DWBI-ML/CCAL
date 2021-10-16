import argparse
import os
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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', help='If training is to be done on a GPU')
    parser.add_argument('--dataset', type=str, default='mnist', help='Name of the dataset used.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size used for training and testing')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to where the data is')
    parser.add_argument('--beta', type=float, default=1, help='Hyperparameter for training. The parameter for VAE')
    parser.add_argument('--out_path', type=str, default='./results', help='Path to where the output log will be')
    parser.add_argument('--log_name', type=str, default='accuracies.log',
                        help='Final performance of the models will be saved with this name')
    parser.add_argument('--generator_name', type=str, default='generator',
                        help='Final performance of the models will be saved with this name')
    parser.add_argument('--selector_name', type=str, default='selector',
                        help='Final performance of the models will be saved with this name')
    parser.add_argument('--discriminator_two_name', type=str, default='discriminator_two',
                        help='Final performance of the models will be saved with this name')
    parser.add_argument('--classifier_name', type=str, default='classifier',
                        help='Final performance of the models will be saved with this name')

    parser.add_argument('--ED_name', type=str, default='ED',
                        help='Final performance of the models will be saved with this name')
    parser.add_argument('--ED_Three_name', type=str, default='ED_Three',
                        help='Final performance of the models will be saved with this name')
    parser.add_argument('--mismatch', type=float, default=0.2,
                        help='mismatch')
    parser.add_argument('--number', type=int, default=0,
                        help='0 is initial, others is query times')

    # parser.add_argument('--decoder_name', type=str, default='decoder',
    #                     help='Final performance of the models will be saved with this name')
    #


    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    return args
