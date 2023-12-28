# On the Importance of Feature Separability in Predicting Out-Of-Distribution Error
This repository is the official implementation of "[On the Importance of Feature Separability in Predicting Out-Of-Distribution Error](https://arxiv.org/abs/2303.15488)" published in NeurIPS 2023. 

## Datasets
### Pre-training process
1. CIFAR-10 and CIFAR-100 can be downloaded in the code.

2. Download TinyImageNet

`wget http://cs231n.stanford.edu/tiny-imagenet-200.zip`
`unzip tiny-imagenet-200.zip`
`rm tiny-imagenet-200.zip`

### Evaluation process
1. Download CIFAR-10C

`mkdir -p ./data/cifar`
`curl -O https://zenodo.org/record/2535967/files/CIFAR-10-C.tar`
`tar -xvf CIFAR-10-C.tar -C data/cifar10/`

2. Download CIFAR-100C

`mkdir -p ./data/cifar`
`curl -O https://zenodo.org/record/3555552/files/CIFAR-100-C.tar`
`tar -xvf CIFAR-100-C.tar -C data/cifar100/`

3. Download TinyImageNet-C

`https://github.com/hendrycks/robustness`

## Pre-training and evaluation

Step 1: Pre-train models on CIFAR-10, CIFAR-100 and TinyImageNet using commands in `./bash/init_base_model.sh`.

Step 2: Estimate OOD error on CIFAR-10C, CIFAR-100C and TinyImageNet-C using commands in `./bash/dispersion.sh`.

## Reference

If you find it is useful for your work, please consider citing 
```
@article{xie2023dispersion,
  title={On the Importance of Feature Separability in Predicting Out-Of-Distribution Error},
  author={Xie, Renchunzi and Wei, Hongxin and Feng, Lei and An, Bo},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```