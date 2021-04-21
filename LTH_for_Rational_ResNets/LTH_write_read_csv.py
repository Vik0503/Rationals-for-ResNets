import csv

import torch
import pandas as pd
from IPython.core.display import display

import argparser
from datetime import datetime

from LTH_for_Rational_ResNets import Mask
from LTH_for_Rational_ResNets.LTH_Models import resnet20_cifar10 as rn20

args = argparser.get_arguments()


def make_csv(model, prune_percent: list, test_acc: list):
    time_stamp = datetime.now()
    PATH = 'CSV/{}'.format(model) + '/{}'.format(time_stamp) + '.csv'
    with open(PATH, 'w', newline='') as csvfile:
        fieldnames = ['Percentage of Weights pruned', 'Test Accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, dialect='excel')

        writer.writeheader()
        for i in range(len(prune_percent)):
            writer.writerow({'Percentage of Weights pruned': prune_percent[i].cpu().numpy(), 'Test Accuracy': test_acc[i]})

    return PATH


def make_mask_csv(PATHS: list):
    time_stamp = datetime.now()
    args = argparser.get_arguments()
    prune_shortcuts = args.prune_shortcuts

    model = rn20.resnet20()
    original_mask: Mask
    original_mask = Mask.make_initial_mask(model)

    masks = [original_mask]
    for p in range(len(PATHS)):
        PATH = PATHS[p]
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
    PATH = './CSV/Masks/all_models/{}'.format(time_stamp) + '.csv'
    df.to_csv(PATH, index=True)

    return PATH
