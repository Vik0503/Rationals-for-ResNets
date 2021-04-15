import torch

from LTH_for_Rational_ResNets.LTH_Models import select_2_expert_groups_rational_resnet as sel
from LTH_for_Rational_ResNets.LTH_Models import rational_resnet20_cifar10 as rrn20
from LTH_for_Rational_ResNets import Mask
from LTH_for_Rational_ResNets import utils
from copy import deepcopy

model = rrn20.rational_resnet20()
mask: Mask
original_mask: Mask
original_mask = Mask.make_initial_mask(model)

PATH = '/home/viktoria/Git/Rationals-for-ResNets/LTH_for_Rational_ResNets/Saved_Models/no_shortcuts_14.4./2021-04-14 16:17:30.512244_ep23s99.41239_test0.90934.pth'
checkpoint = torch.load(PATH)

mask = checkpoint['mask']

sparse_mask = {}

for key, values in mask.items():
    sparse_mask[key] = values.to_sparse().requires_grad_(True)

sparse_original = {}

for key, values in original_mask.items():
    sparse_original[key] = values.to_sparse().requires_grad_(True)

print('original: ', sparse_original)
print('pruned', sparse_mask)
"""a = mask['conv_layer_1.weight'].to_sparse().requires_grad_(True)
print(a)"""

"""b = torch.randn(4)
# print(b)
c = torch.zeros_like(b)
e = torch.ones_like(b)
print(c)
print(e)
# a = torch.randn(2, 3).to_sparse().requires_grad_(True)
d = c.to_sparse().requires_grad_(True)
f = e.to_sparse().requires_grad_(True)
# print(a)
print(d)
print(f)"""