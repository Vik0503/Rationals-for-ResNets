from typing import List

import torch
from torch import nn

from LTH_for_Rational_ResNets import argparser

args = argparser.get_arguments()
prune_shortcuts = args.prune_shortcuts


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
