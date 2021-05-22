import inspect
import os
import sys

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import torch
from Rational_ResNets import argparser
from rational.torch import Rational

resnet_args = argparser.get_arguments()
torch.cuda.manual_seed_all(resnet_args.data_seeds)
import torch.nn as nn

import numpy as np

np.random.seed(resnet_args.data_seeds)
from datetime import datetime

from Rational_ResNets import train_val_test as tvt
from Rational_ResNets import plots
from Rational_ResNets.ResNet_Datasets import CIFAR10, SVHN
from Rational_ResNets.ResNet_Models import univ_rational_resnet_imagenet as univ_rat_imagenet, univ_rational_resnet_cifar10 as univ_rat_cifar
from Rational_ResNets.ResNet_Models import relu_resnet_imagenet as relu_imagenet, relu_resnet_cifar10 as relu_cifar
from Rational_ResNets.ResNet_Models import mix_experts_resnet_cifar10 as mix_exp_cifar, mix_experts_resnet_imagenet as mix_exp_imagenet
from Rational_ResNets.ResNet_Models.mix_experts_resnet_cifar10 import RationalResNet as mix_cifar
from Rational_ResNets.ResNet_Models.mix_experts_resnet_imagenet import RationalResNet as mix_img
from Rational_ResNets.ResNet_Models.univ_rational_resnet_cifar10 import RationalResNet as univ_cifar
from Rational_ResNets.ResNet_Models.univ_rational_resnet_imagenet import RationalResNet as univ_img
from Rational_ResNets import utils

time_stamp = datetime.now()
print_PATH = './Print_Logs/{}.txt'.format(time_stamp)
# sys.stdout = open(print_PATH, 'wt')

global classes
global num_classes
global it_per_ep

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

if resnet_args.dataset == 'cifar10':
    _, _, it_per_ep = CIFAR10.get_train_data(aug=resnet_args.augment_data, bs=resnet_args.batch_size)
    classes = CIFAR10.get_classes()
    num_classes = CIFAR10.get_num_classes()

elif resnet_args.dataset == 'SVHN':
    _, _, it_per_ep = SVHN.get_train_data(aug=resnet_args.augment_data, bs=resnet_args.batch_size)
    classes = SVHN.get_classes()
    num_classes = SVHN.get_num_classes()


def run_all():
    rational_inits = resnet_args.initialize_rationals
    num_rationals = len(rational_inits)
    if resnet_args.run_all_classic and resnet_args.arch_for_run_all == 'CIFAR10':
        model_names = ['ReLU ResNet20', 'univ. rational ResNet20', 'mix. exp. ResNet20']
        model_names_dir = ['relu_resnet20', 'univ_rational_resnet20', 'mix_experts_resnet20']
        models_run_all = [relu_cifar.relu_resnet20(), univ_rat_cifar.univ_rational_resnet20(), mix_exp_cifar.mix_exp_resnet20(rational_inits=rational_inits, num_rationals=num_rationals)]
    elif resnet_args.run_all_classic and resnet_args.arch_for_run_all == 'ImageNet':
        model_names = ['relu_resnet18', 'univ_rational_resnet18', 'mix_experts_resnet18']
        models_run_all = [relu_imagenet.relu_resnet18(), univ_rat_imagenet.univ_rational_resnet18(), mix_exp_imagenet.mix_exp_resnet18(rational_inits=rational_inits, num_rationals=num_rationals)]

    elif resnet_args.run_all_two_BB:
        model_names = ['ReLU ResNet14_A', 'univ. rational ResNet14_A', 'mix. exp. ResNet14_A']
        model_names_dir = ['relu_resnet14_A', 'univ_rational_resnet14_A', 'mix_experts_resnet14_A']
        models_run_all = [relu_cifar.relu_resnet14_A(), univ_rat_cifar.univ_rational_resnet14_A(), mix_exp_cifar.mix_exp_resnet14_A(rational_inits=rational_inits, num_rationals=num_rationals)]

    elif resnet_args.run_all_two_layers and resnet_args.arch_for_run_all == 'CIFAR10':
        model_names = ['relu_resnet14_B', 'univ_rational_resnet14_B', 'mix_experts_resnet14_B']
        model_names_dir = ['relu_resnet14_B', 'univ_rational_resnet14_B', 'mix_experts_resnet14_B']
        models_run_all = [relu_cifar.relu_resnet14_B(), univ_rat_cifar.univ_rational_resnet14_B(), mix_exp_cifar.mix_exp_resnet14_B(rational_inits=rational_inits, num_rationals=num_rationals)]
    elif resnet_args.run_all_two_layers and resnet_args.arch_for_run_all == 'ImageNet':
        model_names = ['relu_resnet18_2_layers', 'univ_rational_resnet18_2_layers', 'mix_experts_resnet18_2_layers']
        models_run_all = [relu_imagenet.relu_resnet18_2_layers(), univ_rat_imagenet.univ_rational_resnet18_2_layers(), mix_exp_imagenet.mix_exp_resnet18_2_layers(rational_inits=rational_inits, num_rationals=num_rationals)]

    elif resnet_args.run_all_one_layer and resnet_args.arch_for_run_all == 'CIFAR10':
        model_names = ['ReLU ResNet8', 'univ. rational ResNet8', 'mix. exp. ResNet8']
        model_names_dir = ['relu_resnet8', 'univ_rational_resnet8', 'mix_experts_resnet8']
        models_run_all = [relu_cifar.relu_resnet8(), univ_rat_cifar.univ_rational_resnet8(), mix_exp_cifar.mix_exp_resnet8(rational_inits=rational_inits, num_rationals=num_rationals)]
    else:
        model_names = ['relu_resnet18_1_layer', 'univ_rational_resnet18_1_layer', 'mix_experts_resnet18_1_layer']
        models_run_all = [relu_imagenet.relu_resnet18_1_layer(), univ_rat_imagenet.univ_rational_resnet18_1_layer(), mix_exp_imagenet.mix_exp_resnet18_1_layer(rational_inits=rational_inits, num_rationals=num_rationals)]

    accuracy_plot_x_vals = []
    train_accs = []
    val_accs = []
    test_accs = []
    best_test_accs = []
    avg_time = []
    csv_PATHS = []
    saved_models_PATHS = []
    act_func_PATH = []

    for m in range(len(models_run_all)):
        model = models_run_all[m]
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

        if resnet_args.hist:
            for mod in model.modules():
                if isinstance(mod, Rational):
                    mod.input_retrieve_mode(max_saves=1)
        model = model.to(device)

        scheduler, optimizer = utils.get_scheduler_optimizer(model, it_per_ep)
        cm, time_elapsed_epoch, best_acc, train_acc_plot_y_vals, val_acc_plot_y_vals, test_acc_plot_y_vals, accuracy_plot_x_vals = tvt.train_val_test_model(model, optimizer, scheduler)

        if resnet_args.save_res_csv:
            PATH = utils.make_csv(model_names_dir[m], accuracy_plot_x_vals, train_acc_plot_y_vals, val_acc_plot_y_vals, test_acc_plot_y_vals)
            csv_PATHS.append(PATH)
        train_accs.append(train_acc_plot_y_vals)
        val_accs.append(val_acc_plot_y_vals)
        test_accs.append(test_acc_plot_y_vals)
        best_test_accs.append(best_acc)
        avg_time.append(time_elapsed_epoch)

        time_stamp = datetime.now()
        saved_model_PATH = './Saved_Models/{}.pth'.format(time_stamp)
        torch.save(model, saved_model_PATH)
        saved_models_PATHS.append(saved_model_PATH)

    plot_PATH = plots.plot_overview_all(train_accs, val_accs, test_accs, accuracy_plot_x_vals, best_test_accs, avg_time, model_names)
    act_func_PATH.append(plots.plot_activation_func_overview_mix(models_run_all[2], num_rationals, rational_inits))
    act_func_PATH.append(plots.plot_activation_func_overview_univ(models_run_all[1]))

    utils.make_yaml(model_names, csv=csv_PATHS, print_log=print_PATH, plot=[plot_PATH], act_func_plot=act_func_PATH, saved_models=saved_models_PATHS)


def run_one():
    act_func_PATH = ''
    rational_inits = resnet_args.initialize_rationals
    num_rationals = 0
    if resnet_args.model == 'relu_resnet20':
        model = relu_cifar.relu_resnet20()
    elif resnet_args.model == 'relu_resnet14_A':
        model = relu_cifar.relu_resnet14_A()
    elif resnet_args.model == 'relu_resnet14_B':
        model = relu_cifar.relu_resnet14_B()
    elif resnet_args.model == 'relu_resnet8':
        model = relu_cifar.relu_resnet8()

    elif resnet_args.model == 'univ_rational_resnet20':
        model = univ_rat_cifar.univ_rational_resnet20()
        num_rationals = 2
    elif resnet_args.model == 'univ_rational_resnet14_A':
        model = univ_rat_cifar.univ_rational_resnet14_A()
        num_rationals = 2
    elif resnet_args.model == 'univ_rational_resnet14_B':
        model = univ_rat_cifar.univ_rational_resnet14_B()
        num_rationals = 2
    elif resnet_args.model == 'univ_rational_resnet8':
        model = univ_rat_cifar.univ_rational_resnet8()
        num_rationals = 2

    elif resnet_args.model == 'mix_experts_resnet20':
        num_rationals = len(rational_inits)
        model = mix_exp_cifar.mix_exp_resnet20(rational_inits=rational_inits, num_rationals=num_rationals)
        num_rationals = len(rational_inits) * 2
    elif resnet_args.model == 'mix_experts_resnet14_A':
        num_rationals = len(rational_inits)
        model = mix_exp_cifar.mix_exp_resnet14_A(rational_inits=rational_inits, num_rationals=num_rationals)
        num_rationals = len(rational_inits) * 2
    elif resnet_args.model == 'mix_experts_resnet14_B':
        num_rationals = len(rational_inits)
        model = mix_exp_cifar.mix_exp_resnet14_B(rational_inits=rational_inits, num_rationals=num_rationals)
        num_rationals = len(rational_inits) * 2
    elif resnet_args.model == 'mix_experts_resnet8':
        num_rationals = len(rational_inits)
        model = mix_exp_cifar.mix_exp_resnet8(rational_inits=rational_inits, num_rationals=num_rationals)
        num_rationals = len(rational_inits) * 2

    elif resnet_args.model == 'relu_resnet18':
        model = relu_imagenet.relu_resnet18()
    elif resnet_args.model == 'relu_resnet18_2_layers':
        model = relu_imagenet.relu_resnet18_2_layers()
    elif resnet_args.model == 'relu_resnet8':
        model = relu_imagenet.relu_resnet18_1_layer()

    elif resnet_args.model == 'univ_rational_resnet18':
        model = univ_rat_imagenet.univ_rational_resnet18()
        num_rationals = 2
    elif resnet_args.model == 'univ_rational_resnet18_2_layers':
        model = univ_rat_imagenet.univ_rational_resnet18_2_layers()
        num_rationals = 2
    elif resnet_args.model == 'univ_rational_resnet18_1_layer':
        model = univ_rat_imagenet.univ_rational_resnet18_1_layer()
        num_rationals = 2

    elif resnet_args.model == 'mix_experts_resnet18':
        num_rationals = len(rational_inits)
        model = mix_exp_imagenet.mix_exp_resnet18(rational_inits=rational_inits, num_rationals=num_rationals)
        num_rationals = len(rational_inits) * 2
    elif resnet_args.model == 'mix_experts_resnet18_2_layers':
        num_rationals = len(rational_inits)
        model = mix_exp_imagenet.mix_exp_resnet18_2_layers(rational_inits=rational_inits, num_rationals=num_rationals)
        num_rationals = len(rational_inits) * 2
    else:
        num_rationals = len(rational_inits)
        model = mix_exp_imagenet.mix_exp_resnet18_1_layer(rational_inits=rational_inits, num_rationals=num_rationals)
        num_rationals = len(rational_inits) * 2

    if resnet_args.hist:
        for mod in model.modules():
            if isinstance(mod, Rational):
                mod.input_retrieve_mode(max_saves=1)

    model = model.to(device)

    scheduler, optimizer = utils.get_scheduler_optimizer(model, it_per_ep)
    cm, avg_time, best_test_acc, train_acc_plot_y_vals, val_acc_plot_y_vals, test_acc_plot_y_vals, accuracy_plot_x_vals = tvt.train_val_test_model(model, optimizer=optimizer, scheduler=scheduler)

    plot_PATH = plots.final_plot(cm=cm, epoch_time=avg_time, best_test_acc=best_test_acc, num_rationals=num_rationals, test_acc_y_vals=test_acc_plot_y_vals, train_acc_y_vals=train_acc_plot_y_vals, acc_x_vals=accuracy_plot_x_vals,
                                 val_acc_y_vals=val_acc_plot_y_vals)

    if isinstance(model, mix_img) or isinstance(model, mix_cifar):
        act_func_PATH = plots.plot_activation_func_overview_mix(model, int(num_rationals / 2), rational_inits)
    if isinstance(model, univ_img) or isinstance(model, univ_cifar):
        act_func_PATH = plots.plot_activation_func_overview_univ(model)

    models = [resnet_args.model]
    csv_PATH = ''
    if resnet_args.save_res_csv:
        csv_PATH = utils.make_csv(resnet_args.model, accuracy_plot_x_vals, train_acc_plot_y_vals, val_acc_plot_y_vals, test_acc_plot_y_vals)
    time_stamp = datetime.now()
    saved_model_PATH = './Saved_Models/{}.pth'.format(time_stamp)
    torch.save(model, saved_model_PATH)
    utils.make_yaml(models, csv=[csv_PATH], saved_models=[saved_model_PATH], print_log=print_PATH, plot=plot_PATH, act_func_plot=[act_func_PATH])


if resnet_args.run_all_classic or resnet_args.run_all_two_BB or resnet_args.run_all_two_layers or resnet_args.run_all_one_layer:
    run_all()
else:
    run_one()
