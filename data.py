
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

    if args.dset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2434, 0.2615]
    if args.dset == 'cifar100':
        mean = [0.5071, 0.4865, 0.4409]
        std = [0.2673, 0.2564, 0.2761]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
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

    return train_loader, test_loader