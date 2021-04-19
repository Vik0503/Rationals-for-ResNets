import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.random import randn
from IPython.display import display
from LTH_for_Rational_ResNets.LTH_Models import resnet20_cifar10 as rn20
from LTH_for_Rational_ResNets.LTH_Models import select_2_expert_groups_rational_resnet as sel
from LTH_for_Rational_ResNets.LTH_Models import rational_resnet20_cifar10 as rrn20
from LTH_for_Rational_ResNets import Mask
from LTH_for_Rational_ResNets import utils
from copy import deepcopy

models = [rrn20.rational_resnet20(), rn20.resnet20(), sel.select_2_expert_groups_rational_resnet20(['leaky_relu', 'gelu', 'swish', 'tanh', 'sigmoid'])]
model = rrn20.rational_resnet20()
mask: Mask

original_masks = []
for mod in range(len(models)):
    model = models[mod]
    original_mask: Mask
    original_mask = Mask.make_initial_mask(model)
    original_masks.append(original_mask)
univ_rat_PATH = './Saved_Models/no_shortcuts_14.4./2021-04-14 07:56:07.220106_ep25s99.62495_test0.90235.pth'
original_PATH = './Saved_Models/no_shortcuts_14.4./2021-04-14 09:17:20.710430_ep20s98.84982_test0.91353.pth'
multi_exp_PATH = './Saved_Models/no_shortcuts_14.4./2021-04-14 16:17:30.512244_ep23s99.41239_test0.90934.pth'

all_PATHS = [univ_rat_PATH, original_PATH, multi_exp_PATH]

masks = []
for p in range(len(all_PATHS)):
    PATH = all_PATHS[p]
    checkpoint = torch.load(PATH)
    mask = checkpoint['mask']
    masks.append(mask)
    masks.append(original_masks[p])

all_data = []
for m in range(len(masks)):
    mask = masks[m]
    data = []
    for key, values in mask.items():
        print(key)
        print(values.shape)
        x = torch.nonzero(values)

        x_indices = []
        x_counter = 0

        y_indices = []
        y_counter = 0
        for i in range(x.shape[0]):
            x_i = x[i][0]
            y_i = x[i][1]

            if x_i not in x_indices:
                x_indices.append(x_i)
                x_counter += 1

            if y_i not in y_indices:
                y_indices.append(y_i)
                y_counter += 1
        x_y_data = [x_counter, y_counter]
        data.append(x_y_data)
        print('x: ', x_counter)
        print('y: ', y_counter)
    all_data.append(data)

# all_data_tensor = torch.tensor(all_data)

array_1 = ['Layer 0'] + ['BasicBlock 0', 'BasicBlock 0', 'BasicBlock 1', 'BasicBlock 1', 'BasicBlock 2', 'BasicBlock 2'] * 3
array_2 = ['conv. 0'] + ['conv. 0', 'conv. 1'] * 9
arrays = [array_1, array_2]
tuples = list(zip(*arrays))
index = pd.MultiIndex.from_tuples(tuples)
df = pd.DataFrame(all_data, index=['pruned univ. rational ResNet20', 'original univ. rational ResNet20', 'pruned ResNet20 Original', 'original ResNet20 Original', 'pruned mix. exp. ResNet20', 'original mix. exp. ResNet20'], columns=index)

display(df)
compression_opts = dict(method='zip', archive_name='out.csv')
df.to_csv('out_2.csv', index=True)

# print(df)
