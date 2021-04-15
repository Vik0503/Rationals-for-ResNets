import torch
import torchvision
from torchvision import transforms
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

aug_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

classes = ['bird', 'car', 'cat', 'deer', 'dog', 'frog', 'horse', 'plane', 'ship', 'truck']


def get_classes() -> list:
    """
    Return all the different classes in the CIFAR10 dataset.

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


def get_train_data(aug: bool = False, bs: int = 128):
    """
    Return train set and train set's dataloader with specific batch size.

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

    train_val_set = torchvision.datasets.CIFAR10(root='../data/cifar10', train=True, download=True, transform=data_transform)
    trainset, _ = torch.utils.data.random_split(train_val_set, [45000, 5000])
    trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=16, batch_size=bs, drop_last=True)
    it_per_ep = np.ceil(len(trainset) / bs).astype(int)
    return trainset, trainloader, it_per_ep


def get_validation_data(aug: bool = False, bs: int = 128):
    """
    Return validation set and dataloader with the validation set and batch size.

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

    train_val_set = torchvision.datasets.CIFAR10(root='../data/cifar10', train=True, download=True, transform=data_transform)
    _, valset = torch.utils.data.random_split(train_val_set, [45000, 5000])
    valloader = torch.utils.data.DataLoader(valset, shuffle=True, num_workers=16, batch_size=bs, drop_last=True)
    return valset, valloader


def get_test_data(aug: bool = False, bs: int = 128):
    """
    Return test set and dataloader with the test set and batch size.

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
    testset = torchvision.datasets.CIFAR10(root='../data/cifar10', train=False, download=True, transform=data_transform)
    testloader = torch.utils.data.DataLoader(testset, shuffle=False, num_workers=16, batch_size=bs, drop_last=True)
    return testset, testloader
