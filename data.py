
import os
import sys
import re
import datetime

import numpy

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloader(args):

    # cifar10/cifar100: 32x32, stl10: 96x96, fmnist: 28x28
    if args.dset in ['cifar10', 'stl10']:
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2434, 0.2615])
        transform = transforms.Compose([
            # transforms.RandomCrop(96, padding=4), # for stl10
            transforms.ToTensor(),
            normalize
        ])
    elif args.dset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761])
        transform = transforms.Compose([
            # transforms.RandomCrop(96, padding=4), # for stl10
            transforms.ToTensor(),
            normalize
        ])
    elif args.dset == 'fmnist':
        fashion_mnist = torchvision.datasets.FashionMNIST(download=True, train=True, root="data").train_data.float()
        transform = transforms.Compose([transforms.Resize((32, 32)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((fashion_mnist.mean() / 255,), (fashion_mnist.std() / 255,))
                                        ])

    if args.dset == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=True, download=True, transform=transform),
            batch_size=args.batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=False, download=True, transform=transform),
            batch_size=args.batch_size, shuffle=False)
    elif args.dset == 'cifar100':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('data', train=True, download=True, transform=transform),
            batch_size=args.batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('data', train=False, download=True, transform=transform),
            batch_size=args.batch_size, shuffle=False)

    elif args.dset == 'stl10':
        # print('STL10 is on the test')
        train_loader = torch.utils.data.DataLoader(
            datasets.STL10('data', split='train', download=True, transform=transform),
            batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.STL10('data', split='test', download=True, transform=transform),
            batch_size=args.batch_size, shuffle=False)

    elif args.dset == 'fmnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST("data", download=True, train=True, transform=transform),
            batch_size=args.batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST("data", download=True, train=False, transform=transform),
            batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader