from typing import List

import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from LTH_for_Rational_ResNets import argparser
from LTH_for_Rational_ResNets.Datasets import CIFAR10 as cifar10
from LTH_for_Rational_ResNets.Datasets import SVHN

LTH_args = argparser.get_arguments()
prune_shortcuts = LTH_args.prune_shortcuts

if LTH_args.dataset == 'cifar10':
    it_per_ep = cifar10.get_it_per_epoch(bs=LTH_args.batch_size)

elif LTH_args.dataset == 'SVHN':
    it_per_ep = SVHN.get_it_per_epoch(bs=LTH_args.batch_size)


def reinit(model, mask, initial_state_model):
    """
    Reset pruned model's weights to the initial initialization.

    Parameter
    ---------
    model: RationalResNet
    mask: Mask
          A mask with pruned weights.
    initial_state_model: dict
                         Initially saved state, before the model is trained.
    """
    for name, param in model.named_parameters():
        if prune_shortcuts:
            if 'weight' in name and 'batch_norm' not in name and 'shortcut.1.' not in name and 'fc' not in name:
                param.data = initial_state_model[name].clone().detach() * mask[name].clone().detach()
            else:
                param.data = initial_state_model[name].clone().detach()
        else:
            if 'weight' in name and 'batch_norm' not in name and 'shortcut' not in name and 'fc' not in name:
                param.data = initial_state_model[name].clone().detach() * mask[name].clone().detach()
            else:
                param.data = initial_state_model[name].clone().detach()


def prunable_layers(model) -> List:
    """
    Return all layers that are prunable.

    Returns
    -------
    prunable_layer_list: List
                        A list with all layers that are prunable.
    """
    prunable_layer_list = []
    for n, module in model.named_modules():
        if not prune_shortcuts and 'shortcut' in n:
            continue
        elif isinstance(module, nn.Conv2d):
            prunable_layer_list.append(n + '.weight')
    return prunable_layer_list


def initial_state(model) -> dict:
    initial_state_model = {}
    for name, param in model.named_parameters():
        initial_state_model[name] = param.data.clone().detach()

    return initial_state_model


def initialize_alpha(b: int = 4) -> torch.Tensor:
    """Initialize the vector alpha.

    Parameters
    ----------
    b : int
        The length of the vector alpha.

    Returns
    -------
    alpha : torch.Tensor
            The tensor with initial values for alpha.
    """
    alpha = torch.rand(b, requires_grad=True)
    alpha = alpha / alpha.sum()
    return alpha


def get_scheduler_optimizer(model):
    """
    Return scheduler with custom milestones and optimizer

    Parameters
    ----------
    model

    Returns
    -------
    torch.optim.lr_scheduler.LambdaLR
    optimizer: torch.optim.SGD
    """
    optimizer = optim.SGD(model.parameters(), lr=LTH_args.learning_rate, momentum=0.9, weight_decay=0.0001)
    milestones = LTH_args.milestones
    milestones = list(map(int, milestones))
    milestones.sort()
    print(milestones)

    def lr_lambda(it):
        if it < LTH_args.warmup_iterations:
            if it % 430 == 0:
                print('Warmup')
            return min(1.0, it / LTH_args.warmup_iterations)
        else:
            for m in range(len(milestones)):
                if it < milestones[m] * it_per_ep:
                    if it % 430 == 0:
                        print('Milestone {}: {}'.format(m, 1 * 10 ** -m))
                    return 1 * 10 ** -m
            if it % 430 == 0:
                print('Milestone {}: {}'.format(len(milestones), 1 * 10 ** -(len(milestones))))
            return 1 * 10 ** -(len(milestones))

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda), optimizer
