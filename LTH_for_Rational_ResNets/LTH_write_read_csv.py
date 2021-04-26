import csv

import torch
import pandas as pd
from IPython.core.display import display

import argparser
from datetime import datetime

from LTH_for_Rational_ResNets import Mask
from LTH_for_Rational_ResNets.LTH_Models import resnet20_cifar10 as rn20
from LTH_for_Rational_ResNets.LTH_Models.resnet20_cifar10 import ResNet

args = argparser.get_arguments()


def make_csv(model, prune_percent: list, test_acc: list):
    time_stamp = datetime.now()
    PATH = 'CSV/{}'.format(model) + '/{}'.format(time_stamp) + '.csv'
    with open(PATH, 'w', newline='') as csvfile:
        fieldnames = ['Percentage of Weights pruned', 'Test Accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, dialect='excel')

        writer.writeheader()
        for i in range(len(prune_percent)):
            writer.writerow({'Percentage of Weights pruned': prune_percent[i], 'Test Accuracy': test_acc[i]})

    return PATH


def make_mask_csv(models):
    time_stamp = datetime.now()
    args = argparser.get_arguments()
    prune_shortcuts = args.prune_shortcuts
    model = models[0]
    original_mask: Mask
    original_mask = Mask.make_initial_mask(model)

    masks = [original_mask]
    for m in range(len(models)):
        model = models[m]
        mask = model.mask
        masks.append(mask)

    all_data_dim = []
    all_data_weights = []

    for m in range(len(masks)):
        mask = masks[m]
        data_dim = []
        data_weights = []
        for key, values in mask.items():
            print(key)
            x = torch.nonzero(values)
            data_weights.append(len(x) - 2)

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
            data_dim.append(x_y_data)
            print('x: ', x_counter)
            print('y: ', y_counter)
        all_data_dim.append(data_dim)
        all_data_weights.append(data_weights)

    num_layers = len(model.layers)

    array_0 = []
    array_1 = []
    array_2 = []
    array_3_conv = ['conv. 0', 'conv. 1', 'conv. 2'] + ['conv. 0', 'conv. 1'] * 2
    array_3_conv_1 = ['conv. 0', 'conv. 1', 'conv. 2'] + ['conv. 0', 'conv. 1']

    if prune_shortcuts:
        if num_layers == 1:
            array_0 = ['Layer 0'] + ['Layer 1'] + [''] * 5
            array_1 = [''] + ['BasicBlock 0', '', 'BasicBlock 1', '', 'BasicBlock 2', '']
            array_2 = ['conv. 0'] + ['conv. 0', 'conv. 1'] * 3
            num_bb = 0
        else:
            num_bb = model.layers[1]
        if num_layers == 3:
            if num_bb == 3:
                array_0 = ['Layer 0'] + ['Layer 1'] + [''] * 5 + ['Layer 2'] + [''] * 6 + ['Layer 3'] + [''] * 6
                array_1 = [''] + ['BasicBlock 0', '', 'BasicBlock 1', '', '', 'BasicBlock 2', '', ''] * 3
                array_2 = ['conv 0'] + ['conv. 0', 'conv. 1'] * 3 + array_3_conv * 6
            elif num_bb == 2:
                array_0 = ['Layer 0'] + ['Layer 1'] + [''] * 5 + ['Layer 2'] + [''] * 4 + ['Layer 3'] + [''] * 4
                array_1 = [''] + ['BasicBlock 0', '', 'BasicBlock 1', '', '', 'BasicBlock 2', '', ''] + ['BasicBlock 0', '', 'BasicBlock 1', '', ''] * 2
                array_2 = ['conv 0'] + ['conv. 0', 'conv. 1'] * 3 + array_3_conv_1 * 6
        elif num_layers == 2:
            array_0 = ['Layer 0'] + ['Layer 1'] + [''] * 5 + ['Layer 2'] + [''] * 5
            array_1 = [''] + ['BasicBlock 0', '', 'BasicBlock 1', '', 'BasicBlock 2', ''] * 2
            array_2 = ['conv. 0'] + ['conv. 0', 'conv. 1'] * 6

    else:
        if num_layers == 1:
            array_0 = ['Layer 0'] + ['Layer 1'] + [''] * 5
            array_1 = [''] + ['BasicBlock 0', '', 'BasicBlock 1', '', 'BasicBlock 2', '']
            array_2 = ['conv. 0'] + ['conv. 0', 'conv. 1'] * 3
            num_bb = 0
        else:
            num_bb = model.layers[1]
        if num_layers == 3:
            if num_bb == 3:
                array_0 = ['Layer 0'] + ['Layer 1'] + [''] * 5 + ['Layer 2'] + [''] * 5 + ['Layer 3'] + [''] * 5
                array_1 = [''] + ['BasicBlock 0', '', 'BasicBlock 1', '', 'BasicBlock 2', ''] * 3
                array_2 = ['conv. 0'] + ['conv. 0', 'conv. 1'] * 9
            elif num_bb == 2:
                array_0 = ['Layer 0'] + ['Layer 1'] + [''] * 5 + ['Layer 2'] + [''] * 3 + ['Layer 3'] + [''] * 3
                array_1 = [''] + ['BasicBlock 0', '', 'BasicBlock 1', ''] * 3
                array_2 = ['conv. 0'] + ['conv. 0', 'conv. 1'] * 6
        if num_layers == 2:
            array_0 = ['Layer 0'] + ['Layer 1'] + [''] * 5 + ['Layer 2'] + [''] * 5
            array_1 = [''] + ['BasicBlock 0', '', 'BasicBlock 1', '', 'BasicBlock 2', ''] * 2
            array_2 = ['conv. 0'] + ['conv. 0', 'conv. 1'] * 6
    arrays = [array_0, array_1, array_2]
    tuples = list(zip(*arrays))
    index = pd.MultiIndex.from_tuples(tuples)
    df_dim = pd.DataFrame(all_data_dim, index=['original Model', 'ReLU ResNet20', 'univ. rational ResNet20', 'mix. exp. ResNet20'], columns=index)
    df_weights = pd.DataFrame(all_data_weights, index=['original Model', 'ReLU ResNet20', 'univ. rational ResNet20', 'mix. exp. ResNet20'], columns=index)

    display(df_dim)
    display(df_weights)
    PATH_dim = './CSV/Masks/all_models/{}'.format(time_stamp) + '_dim' + '.csv'
    PATH_weights = './CSV/Masks/all_models/{}'.format(time_stamp) + '_weights' + '.csv'
    df_dim.to_csv(PATH_dim, index=True)
    df_weights.to_csv(PATH_weights, index=True)

    return PATH_dim, PATH_weights





