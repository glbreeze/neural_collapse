
import os
import sys
import re
import datetime

import numpy

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


data_folder = '../dataset'

def get_dataloader(args):

    # cifar10/cifar100: 32x32, stl10: 96x96, fmnist: 28x28
    if args.dset == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2434, 0.2615])
        transform = transforms.Compose([
            # transforms.RandomCrop(96, padding=4), # for stl10
            transforms.ToTensor(),
            normalize
        ])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=True, download=True, transform=transform),
            batch_size=args.batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=False, download=True, transform=transform),
            batch_size=args.batch_size, shuffle=False)

    if args.dset == 'stl10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2434, 0.2615])
        transform = transforms.Compose([
            # transforms.RandomCrop(96, padding=4), # for stl10
            transforms.ToTensor(),
            normalize
        ])
        train_loader = torch.utils.data.DataLoader(
            datasets.STL10('data', split='train', download=True, transform=transform),
            batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.STL10('data', split='test', download=True, transform=transform),
            batch_size=args.batch_size, shuffle=False)

    elif args.dset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761])
        transform = transforms.Compose([
            # transforms.RandomCrop(96, padding=4), # for stl10
            transforms.ToTensor(),
            normalize
        ])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('data', train=True, download=True, transform=transform),
            batch_size=args.batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('data', train=False, download=True, transform=transform),
            batch_size=args.batch_size, shuffle=False)

    elif args.dset == 'fmnist':
        fashion_mnist = torchvision.datasets.FashionMNIST(download=True, train=True, root="data").train_data.float()
        transform = transforms.Compose([transforms.Resize((32, 32)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((fashion_mnist.mean() / 255,), (fashion_mnist.std() / 255,))
                                        ])
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST("data", download=True, train=True, transform=transform),
            batch_size=args.batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST("data", download=True, train=False, transform=transform),
            batch_size=args.batch_size, shuffle=False)

    elif args.dset == 'tinyi': # image_size:64 x 64
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        transform = transforms.Compose([transforms.ToTensor(),
                                        normalize,
                                        ])
        train_dataset = datasets.ImageFolder(os.path.join(data_folder, 'tiny-imagenet-200', 'train'), transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        test_dataset = datasets.ImageFolder(os.path.join(data_folder, 'tiny-imagenet-200', 'val'), transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader