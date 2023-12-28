#!/usr/bin/env bash

gpu=$1

for ARCH in resnet18 resnet50 wrn_50_2
do
    python init_base_model.py --arch ${ARCH} --train_epoch 20 --train_data_name cifar10 --lr 0.001 --batch_size 128 --seed 123 --gpu ${gpu} --cifar_data_path ../datasets/Cifar10 --cifar_corruption_path ../datasets/Cifar10/CIFAR-10-C
    python init_base_model.py --arch ${ARCH} --train_epoch 50 --train_data_name cifar100 --lr 0.001 --batch_size 128 --seed 123 --gpu ${gpu}  --cifar_data_path ../datasets/Cifar100 --cifar_corruption_path ../datasets/Cifar100/CIFAR-100-C
    python init_base_model.py --arch ${ARCH} --train_epoch 50 --train_data_name tinyimagenet --lr 0.001 --batch_size 128 --seed 123 --gpu ${gpu} --cifar_data_path ../datasets/Tiny-ImageNet/tiny-imagenet-200 --cifar_corruption_path ../datasets/Tiny-ImageNet/Tiny-ImageNet-C
done

    python init_base_model.py --arch resnet18 --train_epoch 20 --train_data_name cifar10 --lr 0.001 --batch_size 128 --seed 123 --gpu 1 --cifar_data_path /data/home/czxie/datasets/Cifar10 --cifar_corruption_path /data/home/czxie/datasets/Cifar10/CIFAR-10-C

