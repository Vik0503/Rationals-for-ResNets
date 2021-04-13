import torch

import utils


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
    prunable_layers = utils.prunable_layers(model)
    for module in prunable_layers:
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
    for key, values in weights.items():
        mask[key][torch.abs(values) <= upper_limit] = 0

    return Mask(mask.cuda())


