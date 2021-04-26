import os

import pandas as pd
import torch
from IPython.display import display

from LTH_for_Rational_ResNets import Mask, plots
from LTH_for_Rational_ResNets import argparser
from LTH_for_Rational_ResNets.LTH_Models import resnet20_cifar10 as rn20
from LTH_for_Rational_ResNets.LTH_write_read_csv import make_csv

args = argparser.get_arguments()
prune_shortcuts = args.prune_shortcuts

original_PATH = './Saved_Models/no_shortcuts_14.4./2021-04-14 09:17:20.710430_ep20s98.84982_test0.91353.pth'
univ_rat_PATH = './Saved_Models/no_shortcuts_14.4./2021-04-14 07:56:07.220106_ep25s99.62495_test0.90235.pth'
multi_exp_PATH = './Saved_Models/no_shortcuts_14.4./2021-04-14 16:17:30.512244_ep23s99.41239_test0.90934.pth'

"""original_PATH = './Saved_Models/shorcuts_14.4/2021-04-15 00:01:52.401491_ep20s98.84961_test0.90116.pth'
univ_rat_PATH = './Saved_Models/shorcuts_14.4/2021-04-14 22:42:32.815381_ep21s99.08013_test0.91146.pth'
multi_exp_PATH = './Saved_Models/shorcuts_14.4/2021-04-15 06:30:57.773067_ep22s99.26477_test0.90131.pth'"""

all_PATHS = [original_PATH, univ_rat_PATH, multi_exp_PATH]
model = rn20.resnet20()
original_mask: Mask
original_mask = Mask.make_initial_mask(model)

masks = [original_mask]
for p in range(len(all_PATHS)):
    PATH = all_PATHS[p]
    checkpoint = torch.load(PATH)
    mask = checkpoint['mask']
    masks.append(mask)

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


if prune_shortcuts:
    array_3_conv = ['conv. 0', 'conv. 1', 'conv. 2'] + ['conv. 0', 'conv. 1'] * 2
    array_0 = ['Layer 0'] + ['Layer 1'] + [''] * 5 + ['Layer 2'] + [''] * 6 + ['Layer 3'] + [''] * 6
    array_1 = [''] + ['BasicBlock 0', '', 'BasicBlock 1', '', '', 'BasicBlock 2', '', ''] * 3
    array_2 = ['conv 0'] + ['conv. 0', 'conv. 1'] * 3 + array_3_conv * 6

else:
    array_0 = ['Layer 0'] + ['Layer 1'] + [''] * 5 + ['Layer 2'] + [''] * 5 + ['Layer 3'] + [''] * 5
    array_1 = [''] + ['BasicBlock 0', '', 'BasicBlock 1', '', 'BasicBlock 2', ''] * 3
    array_2 = ['conv. 0'] + ['conv. 0', 'conv. 1'] * 9

arrays = [array_0, array_1, array_2]
tuples = list(zip(*arrays))
index = pd.MultiIndex.from_tuples(tuples)
df = pd.DataFrame(all_data, index=['original Model', 'ReLU ResNet20', 'univ. rational ResNet20', 'mix. exp. ResNet20'], columns=index)

display(df)
compression_opts = dict(method='zip', archive_name='out.csv')
df.to_csv('./Results/Masks/all_models/all_LTH_no_shortcuts_2.csv', index=True)


os.chdir('/home/viktoria/Git/Rationals-for-ResNets/LTH_for_Rational_ResNets/Saved_Models/shortcuts_14.4')
test_acc: float
test_accs_1 = []
test_accs_2 = []
test_accs_3 = []


sparsity: float
sparsities_1 = [0]
sparsities_2 = [0]
sparsities_3 = [0]


for file in sorted(os.listdir('/home/viktoria/Git/Rationals-for-ResNets/LTH_for_Rational_ResNets/Saved_Models/shortcuts_14.4/resnet20')):
    checkpoint = torch.load('/home/viktoria/Git/Rationals-for-ResNets/LTH_for_Rational_ResNets/Saved_Models/shortcuts_14.4/resnet20/' + file)
    test_acc = checkpoint['test_accuracy']
    test_accs_1.append(test_acc * 100)
    sparsity = checkpoint['sparsity']
    sparsities_1.append(sparsity)

for file in sorted(os.listdir('/home/viktoria/Git/Rationals-for-ResNets/LTH_for_Rational_ResNets/Saved_Models/shortcuts_14.4/rational_resnet20')):
    checkpoint = torch.load('/home/viktoria/Git/Rationals-for-ResNets/LTH_for_Rational_ResNets/Saved_Models/shortcuts_14.4/rational_resnet20/' + file)
    test_acc = checkpoint['test_accuracy']
    test_accs_2.append(test_acc * 100)
    sparsity = checkpoint['sparsity']
    sparsities_2.append(sparsity)

for file in sorted(os.listdir('/home/viktoria/Git/Rationals-for-ResNets/LTH_for_Rational_ResNets/Saved_Models/shortcuts_14.4/mix_exp_resnet20')):
    checkpoint = torch.load('/home/viktoria/Git/Rationals-for-ResNets/LTH_for_Rational_ResNets/Saved_Models/shortcuts_14.4/mix_exp_resnet20/' + file)
    test_acc = checkpoint['test_accuracy']
    test_accs_3.append(test_acc * 100)
    sparsity = checkpoint['sparsity']
    sparsities_3.append(sparsity)

print(test_accs_1)
print(sparsities_1)
print(test_accs_2)
print(sparsities_2)
print(test_accs_3)
print(sparsities_3)

all_test_accs = [test_accs_1, test_accs_2, test_accs_3]
all_sparsities = [sparsities_1[:-1], sparsities_2[:-1], sparsities_3[:-1]]

num_epochs = [20, 21, 22]

plots.plot_all(all_test_accs, all_sparsities, num_epochs)

make_csv('resnet20', sparsities_1[:-1], test_accs_1)
