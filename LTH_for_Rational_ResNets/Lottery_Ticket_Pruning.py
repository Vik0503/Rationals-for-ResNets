import random
from typing import List, Dict

import numpy as np
import torch

from LTH_for_Rational_ResNets import utils
from LTH_for_Rational_ResNets.Mask import Mask, make_new_mask, get_number_of_unpruned_weights, get_number_of_weights


def make_pruning_global(mask) -> Mask:
    """
    Shuffle mask to make random, global pruning possible.

    Parameters
    ----------
    mask: Mask
    Returns
    -------
    Mask
         Shuffled mask for random, global pruning
    """
    weights = dict_to_list(mask)
    random.shuffle(weights)
    weights_tensor = torch.Tensor(weights)
    mask_new = tensor_to_dict(weights_tensor, mask)

    return Mask(mask_new)


def prune_layerwise(pruning_frac: float, model_prune, mask: Mask = None):  # needed???
    """
    Prune every layer with the same percentage.

    Parameters
    ----------
    pruning_frac: float
                  Fracture of the model's weights that will be pruned.
    model_prune:
                 Model whose weights will be pruned.
    mask: Mask

    Returns
    -------
    model_prune:
                 Model with pruned weights.
    mask: Mask
    """
    for name, param in model_prune.prunable_layer_dict():
        data = param.data.cpu().numpy()[mask[name].cpu().numpy() > 0.5]
        threshold = np.percentile(np.abs(data), pruning_frac)
        mask[name][torch.abs(param.data) < threshold] = 0
        model_prune.apply_mask(mask)

    return model_prune, mask


def dict_to_list(dictionary) -> List:
    """
    Transform a dictionary into a list.

    Parameters
    ----------
    dictionary:
                Dictionary to be transformed into a list.

    Returns
    -------
    weight_list: List
                 A list containing the values of the dictionary.
    """
    tensor_list = []
    for key in sorted(dictionary.keys()):
        tensor_dict = torch.cat([dictionary[key].reshape(-1)])
        tensor_list.append(tensor_dict)

    weight_list = []
    for i in range(len(tensor_list)):
        k = tensor_list[i].numpy()
        for j in range(len(k)):
            weight_list.append(k[j])
    return weight_list


def tensor_to_dict(tensor: torch.Tensor, ref_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Transform a tensor into a dict with a given shape.

    Parameters
    ----------
    tensor: torch.Tensor
    ref_dict: Dict[str, torch.Tensor]
              Dictionary that has the shape that the output dictionary should have.

    Returns
    -------
    tensor_dict: Dict[str, torch.Tensor]
    """
    tensor_dict = {}
    for key in sorted(ref_dict.keys()):
        shape = ref_dict[key].shape
        size = 1
        for e in range(len(shape)):
            size *= shape[e]
        element = tensor[:size]
        tensor_dict[key] = element.reshape(shape)

    return tensor_dict


def get_unpruned_weights(model_weights, mask: Mask):
    unpruned_weight_item = torch.cat([values[mask[key] == 1] for key, values in model_weights.items()])
    print(unpruned_weight_item.shape)
    return unpruned_weight_item


def prune(pruning_frac: float, model, mask: Mask):
    prunable_layers = set(utils.prunable_layers(model))

    model_weights = {}
    for item, values in model.state_dict().items():
        if item in prunable_layers:
            model_weights[item] = values.clone().detach()

    weights_unpruned = get_unpruned_weights(model_weights, mask)
    num_prunable_weights = get_number_of_unpruned_weights(mask)

    num_prune_weights = torch.ceil((pruning_frac / 100) * num_prunable_weights)
    print('number of weights to prune: ', num_prune_weights.item())
    sorted_weights, _ = torch.sort(torch.abs(weights_unpruned))
    upper_prune_limit = sorted_weights[int(num_prune_weights.item())]
    updated_mask = make_new_mask(upper_limit=upper_prune_limit, mask=mask, weights=model_weights)
    model.apply_mask(updated_mask)
    return model, updated_mask
