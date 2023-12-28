#!/usr/bin/env bash

severity=$1
gpu=$2

for arch in resnet18 resnet50 wrn_50_2
do
  python main.py --alg dispersion --arch ${arch} --severity ${severity} --dataname cifar10 --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path ../datasets/Cifar10 --cifar_corruption_path ../datasets/Cifar10/CIFAR-10-C --score dispersion
  python main.py --alg dispersion --arch ${arch} --severity ${severity} --dataname cifar100 --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path ../datasets/Cifar100 --cifar_corruption_path ../datasets/Cifar100/CIFAR-100-C --score dispersion
  python main.py --alg dispersion --arch ${arch} --severity ${severity} --dataname tinyimagenet --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path ../datasets/Tiny-ImageNet/tiny-imagenet-200 --cifar_corruption_path ../datasets/Tiny-ImageNet/Tiny-ImageNet-C --score dispersion
done

