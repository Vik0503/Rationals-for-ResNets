import csv
import os

import torch
import pandas as pd
import yaml
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
            writer.writerow({'Percentage of Weights pruned': prune_percent[i], 'Test Accuracy': test_acc[i]})

    return PATH


def make_mask_csv(): # TODO: PATH or Model???
    time_stamp = datetime.now()
    args = argparser.get_arguments()
    prune_shortcuts = args.prune_shortcuts
    original_PATH = '/home/viktoria/Git/Rationals-for-ResNets/LTH_for_Rational_ResNets/Saved_Models/no_shortcuts_14.4/resnet20/2021-04-14 09:17:20.710430_ep20s98.84982_test0.91353.pth'
    univ_rat_PATH = '/home/viktoria/Git/Rationals-for-ResNets/LTH_for_Rational_ResNets/Saved_Models/no_shortcuts_14.4/rational_resnet20/2021-04-14 07:56:07.220106_ep25s99.62495_test0.90235.pth'
    multi_exp_PATH = '/home/viktoria/Git/Rationals-for-ResNets/LTH_for_Rational_ResNets/Saved_Models/no_shortcuts_14.4/mix_exp_resnet20/2021-04-14 16:17:30.512244_ep23s99.41239_test0.90934.pth'

    original_PATH = '/home/viktoria/Git/Rationals-for-ResNets/LTH_for_Rational_ResNets/Saved_Models/shortcuts_14.4/resnet20/2021-04-15 00:01:52.401491_ep20s98.84961_test0.90116.pth'
    univ_rat_PATH = '/home/viktoria/Git/Rationals-for-ResNets/LTH_for_Rational_ResNets/Saved_Models/shortcuts_14.4/rational_resnet20/2021-04-14 22:42:32.815381_ep21s99.08013_test0.91146.pth'
    multi_exp_PATH = '/home/viktoria/Git/Rationals-for-ResNets/LTH_for_Rational_ResNets/Saved_Models/shortcuts_14.4/mix_exp_resnet20/2021-04-15 06:30:57.773067_ep22s99.26477_test0.90131.pth'

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

    """model = models[0]
    original_mask: Mask
    original_mask = Mask.make_initial_mask(model)

    masks = [original_mask]
    for m in range(len(models)):
        model = models[m]
        mask = model.mask
        masks.append(mask)"""

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
                array_1 = [''] + ['BasicBlock 0', '', 'BasicBlock 1', '', 'BasicBlock 2', ''] + ['BasicBlock 0', '', '', 'BasicBlock 1', '', 'BasicBlock 2', ''] * 2
                array_2 = ['conv 0'] + ['conv. 0', 'conv. 1'] * 3 + array_3_conv * 6
            elif num_bb == 2:
                array_0 = ['Layer 0'] + ['Layer 1'] + [''] * 5 + ['Layer 2'] + [''] * 4 + ['Layer 3'] + [''] * 4
                array_1 = [''] + ['BasicBlock 0', '', '', 'BasicBlock 1', '', 'BasicBlock 2', ''] + ['BasicBlock 0', '', 'BasicBlock 1', '', ''] * 2
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


def make_yaml(models: list, saved_models, table=None, csv=None):  # TODO: add Rational Init + One Shot Option + saved models for run all + plot
    LTH_args = argparser.get_arguments()
    time_stamp = datetime.now()
    yaml_data = [{'Date': [time_stamp]}, {'Model(s)': models}, {'Dataset': [LTH_args.dataset]}, {'Batch Size': [LTH_args.batch_size]}, {'Pruning Percentage per Epoch': [LTH_args.pruning_percentage]},
                 {'Training Epochs per Pruning Epoch': [LTH_args.training_number_of_epochs]}, {'Learning Rate': [LTH_args.learning_rate]}, {'Warm-Up Iterations': [LTH_args.warmup_iterations]},
                 {'Shortcuts pruned': [LTH_args.prune_shortcuts]}, {'Saved Models': [saved_models]}]

    if LTH_args.stop_criteria is 'num_prune_epochs':
        yaml_data.append({'Iterative Pruning Epochs': [LTH_args.iterative_pruning_epochs]})
    elif LTH_args.stop_criteria is 'test_acc':
        yaml_data.append({'Test Accuracy Threshold': [LTH_args.test_acc]})

    if LTH_args.save_res_csv:
        yaml_data.append({'CSV File': [csv]})

    if table is not None:
        yaml_data.append({'Mask CSV File': [table]})

    PATH = 'YAML/{}'.format(time_stamp) + '.yaml'
    with open(PATH, 'w') as file:
        documents = yaml.dump(yaml_data, file)