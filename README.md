# Contrastive Coding for Active Learning under Class Distribution Mismatch

Official PyTorch implementation of
["**Contrastive Coding for Active Learning under Class Distribution Mismatch**"](
ICCV2021）


## 1. Requirements
### Environments
Currently, requires following packages.

- CUDA 10.1+
- python == 3.7.9
- pytorch == 1.7.1
- torchvision == 0.8.2
- scikit-learn == 0.24.0
- tensorboardx == 2.1
- matplotlib  == 3.3.3
- numpy == 1.19.2
- scipy == 1.5.3
- [apex](https://github.com/NVIDIA/apex) == 0.1
- [diffdist](https://github.com/ag14774/diffdist) == 0.1 
- [pytorch-gradual-warmup-lr](https://github.com/ildoonet/pytorch-gradual-warmup-lr) packages 



### Datasets 
For CIFAR10 and CIFAR100, we provide a function to automatically download and preprocess the data, you can also download the datasets from the link, and please download it to `~/data`.
* [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
* [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)


## 2. Training
Currently, all code examples are assuming distributed launch with 4 multi GPUs.
To run the code with single GPU, remove `-m torch.distributed.launch --nproc_per_node=4`.

### 



### Semantic feature extraction
To train semantic feature extraction in the paper, run this command:

```train
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 contrast_main.py --mismatch 0.8 --dataset <DATASET> --model <NETWORK> --mode senmatic --shift_trans_type none --batch_size 32 --epoch <EPOCH> --logdir './model/semantic'
```

* **Option** 
* For CIFAR10, set --datatset cifar10, else set --datatset cifar100.
* In our experiment, we set --epoch 700 in cfar10 and --epoch 2000 in cifar100 .
* And we set mismatch = 0.2, 0.4, 0.6, 0.8.


### Distinctive feature extraction
To train distinctive feature extraction in the paper, run this command:

```train
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 contrast_main.py --mismatch 0.8 --dataset <DATASET> --model <NETWORK> --mode feature --shift_trans_type rotation --batch_size 32 --epoch 700 --logdir './model/distinctive'
```

* **Option** 
* For CIFAR10, set --datatset cifar10, else set --datatset cifar100.
* In our experiment, we set --epoch 700 in cifar10 and cifar100 .
* And we set mismatch = 0.2, 0.4, 0.6, 0.8.


### Joint query strategy

To select samples from unlabeled dataset in the paper, run this command:

```train
CUDA_VISIBLE_DEVICES=0 python active_main.py --mode eval --k 100.0 --t 0.9 --dataset <DATASET> --model <NETWORK> --mismatch <MISMATCH> --target <INT> --shift_trans_type rotation --print_score --ood_samples 10 --resize_factor 0.54 --resize_fix --load_feature_path './model/distinctive/last.model' --load_senmatic_path './model/semantic/last.model'  --load_path './model'
```
* **Option** 
* For CIFAR10, set --datatset cifar10, else set --datatset cifar100.
* The value of mismatch is between 0 and 1. In our experiment, we set mismatch = 0.2, 0.4, 0.6, 0.8.
* --target represents the number of queried samples in each category in each AL cycle.

Then, we can get the index of the samples be queried in each active learning cycle. Take mismatch=0.8 for example，the index of the samples should be added in to `CCAL_master/train_classifier/get_index_80`.


## 3. Evaluation
To evaluate the proformance of CCAL, we provide a script to train a classifier, as shown in `CCAL_master/train_classifier`.
, run this command to train the classifier:

```train
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --split <CYCLES> --dataset <DATASET> --mismatch <MISMATCH> --number <NUMBER> --epoch 100
```
* **Option** 
* For CIFAR10, set --datatset cifar10, else set --datatset cifar100.
* The value of mismatch is between 0 and 1. In our experiment, we set mismatch = 0.2, 0.4, 0.6, 0.8. The value of mismatch should be the same as before.
* --number indicates the cycle of active learning.
* --epoch indicates the epochs that training continues in each active learning cycle. In our experiment, we set --epoch 100.
* --split represents the cycles of active learning.

Then, we can get the average of the accuracies over 5 runs(random seed = 0,1,2,3,4,5).

## 4. Citation
```
@InProceedings{Du_2021_ICCV,
    author    = {Du, Pan and Zhao, Suyun and Chen, Hui and Chai, Shuwen and Chen, Hong and Li, Cuiping},
    title     = {Contrastive Coding for Active Learning Under Class Distribution Mismatch},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {8927-8936}
}
```

## 5. Reference
```
@inproceedings{tack2020csi,
  title={CSI: Novelty Detection via Contrastive Learning on Distributionally Shifted Instances},
  author={Jihoon Tack and Sangwoo Mo and Jongheon Jeong and Jinwoo Shin},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```
