import torch
import torchvision
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_val_set = torchvision.datasets.CIFAR10(root='../data/cifar10', train=True, download=True, transform=train_transform)
testset = torchvision.datasets.CIFAR10(root='../data/cifar10', train=False, download=True, transform=train_transform)
trainset, valset = torch.utils.data.random_split(train_val_set, [45000, 5000])


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


def get_it_per_epoch(bs: int = 128) -> int:
    return np.ceil(len(trainset) / bs).astype(int)
