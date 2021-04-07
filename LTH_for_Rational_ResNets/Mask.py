import torch
import numpy as np


class Mask(dict):
    """Mask class for the Lottery Ticket Hypothesis."""

    def __init__(self, mask_dict=None):
        super(Mask, self).__init__()

        if mask_dict is not None:
            for key, values in mask_dict.items():
                self[key] = values

    def cuda(self):
        return Mask({k: v.cuda() for k, v in self.items()})


def make_initial_mask(model):
    """
    Make mask that spans over the whole model.

    Parameters
    ----------
    model:
          The model that the mask spans.

    Returns
    -------
    mask: Mask
    """
    mask = Mask()
    for module in model.prunable_layers():
        mask[module] = torch.ones(list(model.state_dict()[module].shape))
    return mask


def mask_sparsity(mask: Mask):
    """
    Return the sparsity of the mask's model.

    Parameters
    __________
    mask: Mask

    Returns
    _______
    Tensor
           sparsity of the mask's model.
    """
    return 1 - mask_density(mask)


def mask_density(mask):
    """
    Return the density of the mask's model.

    Parameters
    __________
    mask: Mask

    Returns
    _______
    Tensor
           Density of the mask's model.
    """
    return get_number_of_unpruned_weights(mask).float() / get_number_of_weights(mask).float()


def get_number_of_unpruned_weights(mask: Mask):
    """
    Return number of unpruned weights.

    Parameters
    __________
    mask: Mask

    Returns
    -------
    Tensor
           Number of unpruned weights.
    """
    return torch.sum(torch.Tensor([torch.sum(torch.Tensor(values.cpu())) for values in mask.values()]))


def get_number_of_weights(mask: Mask):
    """
    Return number of weights.

    Parameters
    __________
    mask : Mask

    Returns
    -------
    Tensor
           Number of weights.
    """
    return torch.sum(torch.tensor([torch.sum(torch.ones_like(torch.Tensor(values.cpu()))) for values in mask.values()]))


def make_new_mask(upper_limit, mask: Mask, weights) -> Mask:
    """
    Make updated mask.

    Parameters
    ----------
    upper_limit:
                 Threshold of weights with the lowest magnitude.
    mask: Mask
    weights:
             All weights of the net.

    Returns
    -------
    Mask
         Updated mask with all values set to zero, where the weights of the net are lower than the threshold.
    """
    new_mask = Mask({key: np.where(np.abs(values) > upper_limit, mask[key], np.zeros_like(values))
                     for key, values in weights.items()})

    new_mask = {}
    for key, values in weights.items():
        if torch.abs(values) > upper_limit:
            new_mask[key] = mask[key]
        else:
            new_mask[key] = torch.zeros_like(values)

    for key in mask:
        if key not in new_mask:
            new_mask[key] = mask[key]

    new_mask = Mask(new_mask)
    return Mask(new_mask.cuda())


def apply_mask(model, mask: Mask):
    """
    Apply a new mask to a net.

    Parameters
    ----------
    model:
           The model to which the mask is applied.
    mask: Mask
    """
    for name, param in model.named_parameters():
        if 'weight' not in name or 'batch_norm' in name or 'shortcut' in name or 'fc' in name:
            continue
        # param.data = param.data.cpu()
        param.data *= mask[name]
