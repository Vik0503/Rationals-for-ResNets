import torch
import torchvision
from torchvision import transforms
import numpy as np

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])
])

train_val_set = torchvision.datasets.SVHN(root='/home/viktoria/Git/thesis_stuff/data/SVHN', split='train', download=True, transform=train_transform)
testset = torchvision.datasets.SVHN(root='/home/viktoria/Git/thesis_stuff/data/SVHN', split='test', download=True, transform=test_transform)
trainset, valset = torch.utils.data.random_split(train_val_set, [54943, 18314])  # 3/4 of train set for training 1/4 for validation

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


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


def get_trainset():
    """Return part of the dataset, that is used as training set."""
    return trainset


def get_validationset():
    """Return part of the dataset, that is used as validation set."""
    return valset


def get_testset():
    """Return part of the dataset, that is used as test set."""
    return testset


def get_trainloader(bs: int):
    """
    Return dataloader with the train set and batch size.
    Parameters
    ----------
    bs: int
        Batch size for the train dataloader.
    Returns
    -------
    trainloader: DataLoader
    """
    trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=16, batch_size=bs, drop_last=True)
    return trainloader


def get_valloader(bs: int):
    """
    Return dataloader with the validation set and batch size.
    Parameters
    ----------
    bs: int
        Batch size for the validation dataloader.
    Returns
    -------
    valloader: DataLoader
    """
    valloader = torch.utils.data.DataLoader(valset, shuffle=True, num_workers=16, batch_size=bs, drop_last=True)
    return valloader


def get_testloader(bs: int):
    """
    Return dataloader with the test set and batch size.
    Parameters
    ----------
    bs: int
        Batch size for the test dataloader.
    Returns
    -------
    testloader: DataLoader
    """
    testloader = torch.utils.data.DataLoader(testset, shuffle=False, num_workers=16, batch_size=bs, drop_last=True)
    return testloader

