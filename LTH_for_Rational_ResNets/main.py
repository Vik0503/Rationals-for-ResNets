import inspect
import os
import sys
from datetime import datetime
import time

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
    begin = time.time()
    num_epochs = 0
    test_accuracies = []
    sparsities = []
    num_epoch_list = []
    saved_model_PATHS = []
    if LTH_args.run_all_classic and LTH_args.arch_for_run_all == 'CIFAR10':
        model_names = ['ReLU ResNet20', 'univ. rational ResNet20', 'mix. exp. ResNet20']
        model_names_dir = ['relu_resnet20', 'univ_rational_resnet20', 'mix_experts_resnet20']
        models_run_all = [relu_cifar.relu_resnet20(), univ_rat_cifar.univ_rational_resnet20(), mix_exp_cifar.mix_exp_resnet20(rational_inits=rational_inits, num_rationals=num_rationals)]
    elif LTH_args.run_all_classic and LTH_args.arch_for_run_all == 'ImageNet':
        model_names = ['ReLU ResNet18', 'univ. rational ResNet18', 'mix. exp. ResNet18']
        models_run_all = [relu_imagenet.relu_resnet18(), univ_rat_imagenet.univ_rational_resnet18(), mix_exp_imagenet.mix_exp_resnet18(rational_inits=rational_inits, num_rationals=num_rationals)]

    elif LTH_args.run_all_two_BB:
        model_names = ['ReLU ResNet14_A', 'univ. rational ResNet14_A', 'mix. exp. ResNet14_A']
        model_names_dir = ['relu_resnet14_A', 'univ_rational_resnet14_A', 'mix_experts_resnet14_A']
        models_run_all = [relu_cifar.relu_resnet14_A(), univ_rat_cifar.univ_rational_resnet14_A(), mix_exp_cifar.mix_exp_resnet14_A(rational_inits=rational_inits, num_rationals=num_rationals)]

    elif LTH_args.run_all_two_layers and LTH_args.arch_for_run_all == 'CIFAR10':
        model_names = ['ReLU ResNet14_B', 'univ. rational ResNet14_B', 'mix. exp. ResNet14_B']
        model_names_dir = ['relu_resnet14_B', 'univ_rational_resnet14_B', 'mix_experts_resnet14_B']
        models_run_all = [relu_cifar.relu_resnet14_B(), univ_rat_cifar.univ_rational_resnet14_B(), mix_exp_cifar.mix_exp_resnet14_B(rational_inits=rational_inits, num_rationals=num_rationals)]
    elif LTH_args.run_all_two_layers and LTH_args.arch_for_run_all == 'ImageNet':
        model_names = ['relu_resnet18_2_layers', 'univ_rational_resnet18_2_layers', 'mix_experts_resnet18_2_layers']
        models_run_all = [relu_imagenet.relu_resnet18_2_layers(), univ_rat_imagenet.univ_rational_resnet18_2_layers(), mix_exp_imagenet.mix_exp_resnet18_2_layers(rational_inits=rational_inits, num_rationals=num_rationals)]

    elif LTH_args.run_all_one_layer and LTH_args.arch_for_run_all == 'CIFAR10':
        model_names = ['ReLU ResNet8', 'univ. rational ResNet8', 'mix. exp. ResNet8']
        model_names_dir = ['relu_resnet8', 'univ_rational_resnet8', 'mix_experts_resnet8']
        models_run_all = [relu_cifar.relu_resnet8(), univ_rat_cifar.univ_rational_resnet8(), mix_exp_cifar.mix_exp_resnet8(rational_inits=rational_inits, num_rationals=num_rationals)]
    else:
        model_names = ['relu_resnet18_1_layer', 'univ_rational_resnet18_1_layer', 'mix_experts_resnet18_1_layer']
        models_run_all = [relu_imagenet.relu_resnet18_1_layer(), univ_rat_imagenet.univ_rational_resnet18_1_layer(), mix_exp_imagenet.mix_exp_resnet18_1_layer(rational_inits=rational_inits, num_rationals=num_rationals)]

    checkpoints = []
    all_test_accuracies = []
    all_sparsities = []
    all_models = []

    original_model = models_run_all[0]
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
            csv_PATH = LTH_write_read_csv.make_csv(model_names_dir[m], sparsities, test_accuracies)

        saved_model_PATHS.append(model_PATH)
        all_test_accuracies.append(test_accuracies)
        all_sparsities.append(sparsities)
        num_epoch_list.append(num_epochs)
        checkpoints.append(last_checkpoint)
        all_models.append(model)

    plot_PATH = plots.plot_all(test_accs=all_test_accuracies, sparsities=all_sparsities, num_epoch_list=num_epoch_list, model_names=model_names)
    mask_path_dim, mask_path_weights, mask_path_percent = LTH_write_read_csv.make_mask_csv(original_model, checkpoints, model_names)
    LTH_write_read_csv.make_yaml(model_names, csv=[csv_PATH], saved_models=saved_model_PATHS, table=[mask_path_dim, mask_path_weights, mask_path_percent], plot=[plot_PATH], print_log=print_PATH)
    end = time.time()
    elapsed = end - begin
    print('time needed: {:.0f}m {:.0f}s', elapsed // 60, elapsed % 60)


def run_one():  # TODO: Model name 18 str for plots + yaml
    global model, model_PATH, plot_PATH

    if LTH_args.model == 'relu_resnet20':
        model = relu_cifar.relu_resnet20()
        model_name = 'ReLU ResNet20'
    elif LTH_args.model == 'relu_resnet14_A':
        model = relu_cifar.relu_resnet14_A()
        model_name = 'ReLU ResNet14_A'
    elif LTH_args.model == 'relu_resnet14_B':
        model = relu_cifar.relu_resnet14_B()
        model_name = 'ReLU ResNet14_B'
    elif LTH_args.model == 'relu_resnet8':
        model = relu_cifar.relu_resnet8()
        model_name = 'ReLU ResNet8'

    elif LTH_args.model == 'univ_rational_resnet20':
        model = univ_rat_cifar.univ_rational_resnet20()
        model_name = 'univ. rational ResNet20'
    elif LTH_args.model == 'univ_rational_resnet14_A':
        model = univ_rat_cifar.univ_rational_resnet14_A()
        model_name = 'univ. rational ResNet14_A'
    elif LTH_args.model == 'univ_rational_resnet14_B':
        model = univ_rat_cifar.univ_rational_resnet14_B()
        model_name = 'univ. rational ResNet14_B'
    elif LTH_args.model == 'univ_rational_resnet8':
        model = univ_rat_cifar.univ_rational_resnet8()
        model_name = 'univ. rational ResNet8'

    elif LTH_args.model == 'mix_experts_resnet20':
        model = mix_exp_cifar.mix_exp_resnet20(rational_inits=rational_inits, num_rationals=num_rationals)
        model_name = 'mix. exp. ResNet20'
    elif LTH_args.model == 'mix_experts_resnet14_A':
        model = mix_exp_cifar.mix_exp_resnet14_A(rational_inits=rational_inits, num_rationals=num_rationals)
        model_name = 'mix. exp. ResNet14_A'
    elif LTH_args.model == 'mix_experts_resnet14_B':
        model = mix_exp_cifar.mix_exp_resnet14_B(rational_inits=rational_inits, num_rationals=num_rationals)
        model_name = 'mix. exp. ResNet14_B'
    elif LTH_args.model == 'mix_experts_resnet8':
        model = mix_exp_cifar.mix_exp_resnet8(rational_inits=rational_inits, num_rationals=num_rationals)
        model_name = 'mix. exp. ResNet8'

    elif LTH_args.model == 'relu_resnet18':
        model = relu_imagenet.relu_resnet18()
    elif LTH_args.model == 'relu_resnet18_2_layers':
        model = relu_imagenet.relu_resnet18_2_layers()
    elif LTH_args.model == 'relu_resnet8':
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
        plot_PATH = plots.final_plot_LTH(test_accuracies, sparsities, num_epochs, model_name)

    PATH = ''
    if LTH_args.save_res_csv:
        PATH = LTH_write_read_csv.make_csv(LTH_args.model, sparsities, test_accuracies)

    LTH_write_read_csv.make_yaml([model_name], csv=[PATH], saved_models=model_PATH, print_log=print_PATH, plot=plot_PATH)


if LTH_args.run_all_classic or LTH_args.run_all_two_BB or LTH_args.run_all_two_layers or LTH_args.run_all_one_layer:
    run_all()
else:
    run_one()
