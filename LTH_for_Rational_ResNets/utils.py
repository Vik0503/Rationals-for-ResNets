from typing import List

from torch import nn


def reinit(model, mask, initial_state_model):  # TODO: Add shortcut option
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
        if 'rational' in name or 'alpha' in name:
            param.data = initial_state_model[name].clone().detach()
            continue
        if 'weight' not in name or 'batch_norm' in name or 'shortcut' in name or 'fc' in name:
            continue
        else:
            param.data = initial_state_model[name].clone().detach() * mask[name].clone().detach()


def prunable_layers(model) -> List:  # TODO: Add shortcut option
    """
    Return all layers that are prunable.

    Returns
    -------
    prunable_layer_list: List
                        A list with all layers that are prunable.
    """
    prunable_layer_list = []
    for n, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and 'shortcut' not in n:
            prunable_layer_list.append(n + '.weight')

    return prunable_layer_list


def initial_state(model) -> dict:
    initial_state_model = {}
    for name, param in model.named_parameters():
        initial_state_model[name] = param.data.clone().detach()

    return initial_state_model
