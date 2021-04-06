import random
from typing import List, Dict

import numpy as np
import torch

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


def prune(pruning_frac: float, model_prune, mask: Mask):  # eventuell alles auf GPU lassen?
    """
    Prune a percentage of the model's weights globally.

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
    updated_mask: Mask
    """
    np_array = False
    for item, value in mask.items():
        if isinstance(value, np.ndarray):
            np_array = True
            break
    if not np_array:
        np_mask = {key: values.cpu().numpy() for key, values in mask.items()}
    else:
        np_mask = mask

    prunable_layers = set(model_prune.prunable_layers())

    model_weights = {}
    for item, values in model_prune.state_dict().items():
        if item in prunable_layers:
            model_weights[item] = values.clone().detach().cpu().numpy()

    weights_unpruned = torch.Tensor(get_unpruned_weights(model_weights=model_weights, mask=np_mask))

    num_rem_weights = get_number_of_unpruned_weights(mask)
    num_prune_weights = np.ceil((pruning_frac / 100) * num_rem_weights.cpu().numpy())
    print('number of weights to prune: ', num_prune_weights)

    upper_prune_limit = np.sort(np.abs(weights_unpruned.numpy()))[num_prune_weights.astype(int)]

    updated_mask = make_new_mask(upper_limit=upper_prune_limit, mask=np_mask, weights=model_weights)

    # updated_mask = Mask.cuda(updated_mask)
    # model_prune.apply_mask(updated_mask)
    return model_prune, updated_mask


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


def get_unpruned_weights(model_weights, mask: Mask) -> List:  # TODO: unbedingt schöner machen!!!
    """
    Return a list containing all unpruned weights.

    Parameters
    ----------
    model_weights:
                    All weights of the model.
    mask: Mask

    Returns
    -------
    unpruned_weights: List
                      A list containing all unpruned weights.
    """
    unpruned_weights_items = []
    unpruned_weights = []
    for key, values in model_weights.items():
        unpruned_weights_items.append([values[mask[key] == 1]])

    for i in range(len(unpruned_weights_items)):
        k = unpruned_weights_items[i]
        for j in range(len(k)):
            l = k[j]
            for m in range(len(l)):
                unpruned_weights.append(l[m])

    return unpruned_weights


def get_unpruned_weights_2(model_weights, mask: Mask):
    unpruned_weight_item = []
    unpruned_weight_item = torch.cat([values[mask[key] == 1] for key, values in model_weights.items()])
    return unpruned_weight_item


def prune_2(model, mask: Mask, pruning_frac: float):
    prunable_layers = set(model.prunable_layers)

    model_weights = {}
    for item, values in model.state_dict().items():
        if item in prunable_layers:
            model_weights[item] = values.clone().detach()

    weights_unpruned = get_unpruned_weights_2(model_weights, mask)
    num_prunable_weights = get_number_of_unpruned_weights(mask)

    num_weights = get_number_of_weights(mask)

    threshold = torch.sum(torch.abs(torch.tensor(model_weights))) / num_weights

    num_rem_weights = get_number_of_unpruned_weights(mask)
    num_prune_weights = np.ceil((pruning_frac / 100) * num_rem_weights.cpu().numpy())