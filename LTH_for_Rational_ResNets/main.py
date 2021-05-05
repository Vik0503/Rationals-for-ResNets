import inspect
import os
import sys
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import torch
from LTH_for_Rational_ResNets import argparser

LTH_args = argparser.get_arguments()
torch.cuda.manual_seed_all(LTH_args.data_seeds)
from torch import nn
import numpy as np

np.random.seed(LTH_args.data_seeds)

from LTH_for_Rational_ResNets import Lottery_Ticket_Hypothesis
from LTH_for_Rational_ResNets import plots
from LTH_for_Rational_ResNets.Datasets import CIFAR10 as cifar10
from LTH_for_Rational_ResNets.Datasets import SVHN
from LTH_for_Rational_ResNets.LTH_Models import univ_rational_resnet_imagenet as univ_rat_imagenet, univ_rational_resnet_cifar10 as univ_rat_cifar
from LTH_for_Rational_ResNets.LTH_Models import relu_resnet_imagenet as relu_imagenet, relu_resnet_cifar10 as relu_cifar
from LTH_for_Rational_ResNets.LTH_Models import mix_experts_resnet_cifar10 as mix_exp_cifar, mix_experts_resnet_imagenet as mix_exp_imagenet
from LTH_for_Rational_ResNets.LTH_Models import select_1_expert_group_rational_resnet as sel1exp
from LTH_for_Rational_ResNets import LTH_write_read_csv

time_stamp = datetime.now()
print_PATH = './Print_Logs/{}.txt'.format(time_stamp)
sys.stdout = open(print_PATH, 'wt')

global classes
global num_classes
global it_per_ep
global num_rationals
global rational_inits
global prune_shortcuts

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

rational_inits = LTH_args.initialize_rationals
num_rationals = len(rational_inits)

if LTH_args.prune_shortcuts:
    prune_shortcuts = True
else:
    prune_shortcuts = False

if LTH_args.dataset == 'cifar10':
    classes = cifar10.get_classes()
    num_classes = cifar10.get_num_classes()

elif LTH_args.dataset == 'SVHN':
    classes = SVHN.get_classes()
    num_classes = SVHN.get_num_classes()


def run_all():
    global model_PATH, last_checkpoint, csv_PATH
    num_epochs = 0
    test_accuracies = []
    sparsities = []
    num_epoch_list = []
    saved_model_PATHS = []
    if LTH_args.run_all_classic and LTH_args.arch_for_run_all == 'CIFAR10':
        model_names = ['relu_resnet20', 'univ_rational_resnet20', 'mix_experts_resnet20']
        models_run_all = [relu_cifar.relu_resnet20(), univ_rat_cifar.univ_rational_resnet20(), mix_exp_cifar.mix_exp_resnet20(rational_inits=rational_inits, num_rationals=num_rationals)]
    elif LTH_args.run_all_classic and LTH_args.arch_for_run_all == 'ImageNet':
        model_names = ['relu_resnet18', 'univ_rational_resnet18', 'mix_experts_resnet18']
        models_run_all = [relu_imagenet.relu_resnet18(), univ_rat_imagenet.univ_rational_resnet18(), mix_exp_imagenet.mix_exp_resnet18(rational_inits=rational_inits, num_rationals=num_rationals)]

    elif LTH_args.run_all_two_BB:
        model_names = ['relu_resnet20_2_BB', 'univ_rational_resnet20_2_BB', 'mix_experts_resnet20_2_BB']
        models_run_all = [relu_cifar.relu_resnet20_2_BB(), univ_rat_cifar.univ_rational_resnet20_2_BB(), mix_exp_cifar.mix_exp_resnet20_2_BB(rational_inits=rational_inits, num_rationals=num_rationals)]

    elif LTH_args.run_all_two_layers and LTH_args.arch_for_run_all == 'CIFAR10':
        model_names = ['relu_resnet20_2_layers', 'univ_rational_resnet20_2_layers', 'mix_experts_resnet20_2_layers']
        models_run_all = [relu_cifar.relu_resnet20_2_layers(), univ_rat_cifar.univ_rational_resnet20_2_layers(), mix_exp_cifar.mix_exp_resnet20_2_layers(rational_inits=rational_inits, num_rationals=num_rationals)]
    elif LTH_args.run_all_two_layers and LTH_args.arch_for_run_all == 'ImageNet':
        model_names = ['relu_resnet18_2_layers', 'univ_rational_resnet18_2_layers', 'mix_experts_resnet18_2_layers']
        models_run_all = [relu_imagenet.relu_resnet18_2_layers(), univ_rat_imagenet.univ_rational_resnet18_2_layers(), mix_exp_imagenet.mix_exp_resnet18_2_layers(rational_inits=rational_inits, num_rationals=num_rationals)]

    elif LTH_args.run_all_one_layer and LTH_args.arch_for_run_all == 'CIFAR10':
        model_names = ['relu_resnet20_1_layer', 'univ_rational_resnet20_1_layer', 'mix_experts_resnet20_1_layer']
        models_run_all = [relu_cifar.relu_resnet20_1_layer(), univ_rat_cifar.univ_rational_resnet20_1_layer(), mix_exp_cifar.mix_exp_resnet20_1_layer(rational_inits=rational_inits, num_rationals=num_rationals)]
    else:
        model_names = ['relu_resnet18_1_layer', 'univ_rational_resnet18_1_layer', 'mix_experts_resnet18_1_layer']
        models_run_all = [relu_imagenet.relu_resnet18_1_layer(), univ_rat_imagenet.univ_rational_resnet18_1_layer(), mix_exp_imagenet.mix_exp_resnet18_1_layer(rational_inits=rational_inits, num_rationals=num_rationals)]

    checkpoints = []
    all_test_accuracies = []
    all_sparsities = []
    all_models = []

    for m in range(len(models_run_all)):
        model = models_run_all[m]
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model = model.to(device)

        if LTH_args.stop_criteria is 'test_acc':
            num_epochs, test_accuracies, sparsities, model_PATH, last_checkpoint = Lottery_Ticket_Hypothesis.iterative_pruning_by_test_acc(prune_model=model)

        elif LTH_args.stop_criteria is 'num_prune_epochs':
            test_accuracies, sparsities, model_PATH, last_checkpoint = Lottery_Ticket_Hypothesis.iterative_pruning_by_num(prune_model=model)
            num_epochs = LTH_args.iterative_pruning_epochs

        elif LTH_args.stop_criteria is 'one_shot':
            Lottery_Ticket_Hypothesis.one_shot_pruning(model)

        if LTH_args.save_res_csv:
            csv_PATH = LTH_write_read_csv.make_csv(model_names[m], sparsities, test_accuracies)

        saved_model_PATHS.append(model_PATH)
        all_test_accuracies.append(test_accuracies)
        all_sparsities.append(sparsities)
        num_epoch_list.append(num_epochs)
        checkpoints.append(last_checkpoint)
        all_models.append(model)

    plot_PATH = plots.plot_all(test_accs=all_test_accuracies, sparsities=all_sparsities, num_epoch_list=num_epoch_list)
    act_func_plot_PATH = plots.plot_activation_func_overview(all_models[2], num_rationals, LTH_args.initialize_rationals)
    mask_path_dim, mask_path_weights, mask_path_percent = LTH_write_read_csv.make_mask_csv(checkpoints)
    LTH_write_read_csv.make_yaml(model_names, csv=[csv_PATH], saved_models=saved_model_PATHS, table=[mask_path_dim, mask_path_weights, mask_path_percent], plot=[plot_PATH], act_func_plot=[act_func_plot_PATH], print_log=[print_PATH])


def run_one():  # TODO: Model name str for plots + yaml
    global model, model_PATH, plot_PATH

    if LTH_args.model == 'relu_resnet20':
        model = relu_cifar.relu_resnet20()
    elif LTH_args.model == 'relu_resnet20_2_BB':
        model = relu_cifar.relu_resnet20_2_BB()
    elif LTH_args.model == 'relu_resnet20_2_layers':
        model = relu_cifar.relu_resnet20_2_layers()
    elif LTH_args.model == 'relu_resnet20_1_layer':
        model = relu_cifar.relu_resnet20_1_layer()

    elif LTH_args.model == 'univ_rational_resnet20':
        model = univ_rat_cifar.univ_rational_resnet20()
    elif LTH_args.model == 'univ_rational_resnet20_2_BB':
        model = univ_rat_cifar.univ_rational_resnet20_2_BB()
    elif LTH_args.model == 'univ_rational_resnet20_2_layers':
        model = univ_rat_cifar.univ_rational_resnet20_2_layers()
    elif LTH_args.model == 'univ_rational_resnet20_1_layer':
        model = univ_rat_cifar.univ_rational_resnet20_1_layer()

    elif LTH_args.model == 'mix_experts_resnet20':
        model = mix_exp_cifar.mix_exp_resnet20(rational_inits=rational_inits, num_rationals=num_rationals)
    elif LTH_args.model == 'mix_experts_resnet20_2_BB':
        model = mix_exp_cifar.mix_exp_resnet20_2_BB(rational_inits=rational_inits, num_rationals=num_rationals)
    elif LTH_args.model == 'mix_experts_resnet20_2_layers':
        model = mix_exp_cifar.mix_exp_resnet20_2_layers(rational_inits=rational_inits, num_rationals=num_rationals)
    elif LTH_args.model == 'mix_experts_resnet20_1_layer':
        model = mix_exp_cifar.mix_exp_resnet20_1_layer(rational_inits=rational_inits, num_rationals=num_rationals)

    elif LTH_args.model == 'relu_resnet18':
        model = relu_imagenet.relu_resnet18()
    elif LTH_args.model == 'relu_resnet18_2_layers':
        model = relu_imagenet.relu_resnet18_2_layers()
    elif LTH_args.model == 'relu_resnet20_1_layer':
        model = relu_imagenet.relu_resnet18_1_layer()

    elif LTH_args.model == 'univ_rational_resnet18':
        model = univ_rat_imagenet.univ_rational_resnet18()
    elif LTH_args.model == 'univ_rational_resnet18_2_layers':
        model = univ_rat_imagenet.univ_rational_resnet18_2_layers()
    elif LTH_args.model == 'univ_rational_resnet18_1_layer':
        model = univ_rat_imagenet.univ_rational_resnet18_1_layer()

    elif LTH_args.model == 'mix_experts_resnet18':
        model = mix_exp_imagenet.mix_exp_resnet18(rational_inits=rational_inits, num_rationals=num_rationals)
    elif LTH_args.model == 'mix_experts_resnet18_2_layers':
        model = mix_exp_imagenet.mix_exp_resnet18_2_layers(rational_inits=rational_inits, num_rationals=num_rationals)
    elif LTH_args.model == 'mix_experts_resnet18_1_layer':
        model = mix_exp_imagenet.mix_exp_resnet18_1_layer(rational_inits=rational_inits, num_rationals=num_rationals)

    else:
        model = sel1exp.select_1_expert_group_rational_resnet20(rational_inits=rational_inits, num_rationals=num_rationals)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    num_epochs = 0
    test_accuracies = []
    sparsities = []

    if LTH_args.stop_criteria is 'test_acc':
        num_epochs, test_accuracies, sparsities, model_PATH, last_checkpoint = Lottery_Ticket_Hypothesis.iterative_pruning_by_test_acc(prune_model=model)

    elif LTH_args.stop_criteria is 'num_prune_epochs':
        test_accuracies, sparsities, model_PATH, last_checkpoint = Lottery_Ticket_Hypothesis.iterative_pruning_by_num(prune_model=model)
        num_epochs = LTH_args.iterative_pruning_epochs

    elif LTH_args.stop_criteria is 'one_shot':
        Lottery_Ticket_Hypothesis.one_shot_pruning(model)
        num_epochs = 1
    if LTH_args.stop_criteria is not 'one_shot':
        plot_PATH = plots.final_plot_LTH(test_accuracies, sparsities, num_epochs)

    PATH = ''
    if LTH_args.save_res_csv:
        PATH = LTH_write_read_csv.make_csv(LTH_args.model, sparsities, test_accuracies)

    act_func_plot = plots.plot_activation_func_overview(model, num_rationals, LTH_args.initialize_rationals)
    LTH_write_read_csv.make_yaml([LTH_args.model], csv=PATH, saved_models=model_PATH, print_log=print_PATH, act_func_plot=act_func_plot, plot=plot_PATH)


if LTH_args.run_all_classic or LTH_args.run_all_two_BB or LTH_args.run_all_two_layers or LTH_args.run_all_one_layer:
    run_all()
else:
    run_one()
