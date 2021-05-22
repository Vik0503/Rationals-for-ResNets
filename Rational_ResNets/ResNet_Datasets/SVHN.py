import random

import torch
torch.cuda.manual_seed_all(42)

import torchvision
from torchvision import transforms
import numpy as np
np.random.seed(42)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])
])

aug_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])
])

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def random_seed_for_worker(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def get_classes() -> list:
    """
    Return all the different classes in the SVHN dataset.

    Returns
    -------
    classes: list
    """
    return classes


def get_num_classes() -> int:
    """
     Return number of different classes in the CIFAR10 dataset.

     Returns
     -------
     int
         Number of different classes.

     """
    return len(classes)


def get_train_data(aug: bool = False, bs: int = 128):  # TODO: is split a problem???
    """
    Parameters
    ----------
    aug: bool
         data augmentation
    bs: int
        Batch size for the train dataloader.

    Returns
    -------
    trainset:
              Dataset containing the training data.
    trainloader: DataLoader
    """
    if aug:
        data_transform = aug_transform
    else:
        data_transform = transform

    train_val_set = torchvision.datasets.SVHN(root='../data/SVHN', split='train', download=True, transform=data_transform)
    trainset, _ = torch.utils.data.random_split(train_val_set, [54943, 18314])
    trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=16, batch_size=bs, drop_last=True, worker_init_fn=random_seed_for_worker)
    it_per_ep = np.ceil(len(trainset) / bs).astype(int)
    return trainset, trainloader, it_per_ep


def get_validation_data(aug: bool = False, bs: int = 128):
    """
    Return dataloader with the validation set and batch size.

    Parameters
    ----------
    aug: bool
         data augmentation
    bs: int
        Batch size for the validation dataloader.

    Returns
    -------
    valset:
            Dataset containing the validation data.
    valloader: DataLoader
    """
    if aug:
        data_transform = aug_transform
    else:
        data_transform = transform

    train_val_set = torchvision.datasets.SVHN(root='../data/SVHN', split='train', download=True, transform=data_transform)
    _, valset = torch.utils.data.random_split(train_val_set, [54943, 18314])
    valloader = torch.utils.data.DataLoader(valset, shuffle=True, num_workers=16, batch_size=bs, drop_last=True, worker_init_fn=random_seed_for_worker)
    return valset, valloader


def get_test_data(aug: bool = False, bs: int = 128):
    """
    Return dataloader with the test set and batch size.

    Parameters
    ----------
    aug: bool
         data augmentation
    bs: int
        Batch size for the test dataloader.

    Returns
    -------
    testset:
            Dataset containing the test data.
    testloader: DataLoader
    """
    if aug:
        data_transform = aug_transform
    else:
        data_transform = transform
    testset = torchvision.datasets.SVHN(root='../data/SVHN', split='test', download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, shuffle=False, num_workers=16, batch_size=bs, drop_last=True, worker_init_fn=random_seed_for_worker)
    return testset, testloader
