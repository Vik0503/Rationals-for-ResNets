import inspect
import os
import sys

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import torch
from Rational_ResNets import argparser

resnet_args = argparser.get_args()
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
from Rational_ResNets import utils

global trainset
global valset
global testset
global trainloader
global valloader
global testloader
global classes
global num_classes
global it_per_ep

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

if resnet_args.dataset == 'cifar10':
    trainset, trainloader, it_per_ep = CIFAR10.get_train_data(aug=resnet_args.augment_data, bs=resnet_args.batch_size)
    valset, valloader = CIFAR10.get_validation_data(aug=resnet_args.augment_data, bs=resnet_args.batch_size)
    testset, testloader = CIFAR10.get_test_data(aug=resnet_args.augment_data, bs=resnet_args.batch_size)
    classes = CIFAR10.get_classes()
    num_classes = CIFAR10.get_num_classes()

elif resnet_args.dataset == 'SVHN':
    trainset, trainloader, it_per_ep = SVHN.get_train_data(aug=resnet_args.augment_data, bs=resnet_args.batch_size)
    valset, valloader = SVHN.get_validation_data(aug=resnet_args.augment_data, bs=resnet_args.batch_size)
    testset, testloader = SVHN.get_test_data(aug=resnet_args.augment_data, bs=resnet_args.batch_size)
    classes = SVHN.get_classes()
    num_classes = SVHN.get_num_classes()


def run_all():
    rational_inits = resnet_args.initialize_rationals  # TODO: catch exceptions
    num_rationals = len(rational_inits)
    if resnet_args.run_all_classic and resnet_args.run_all_architecture == 'CIFAR10':
        model_names = ['relu_resnet20', 'univ_rational_resnet20', 'mix_experts_resnet20']
        models_run_all = [relu_cifar.relu_resnet20(), univ_rat_cifar.univ_rational_resnet20(), mix_exp_cifar.mix_exp_resnet20(rational_inits=rational_inits, num_rationals=num_rationals)]
    elif resnet_args.run_all_classic and resnet_args.run_all_architecture == 'ImageNet':
        model_names = ['relu_resnet18', 'univ_rational_resnet18', 'mix_experts_resnet18']
        models_run_all = [relu_imagenet.relu_resnet18(), univ_rat_imagenet.univ_rational_resnet18(), mix_exp_imagenet.mix_exp_resnet18(rational_inits=rational_inits, num_rationals=num_rationals)]

    elif resnet_args.run_all_two_BB:
        model_names = ['relu_resnet20_2_BB', 'univ_rational_resnet20_2_BB', 'mix_experts_resnet20_2_BB']
        models_run_all = [relu_cifar.relu_resnet20_2_BB(), univ_rat_cifar.univ_rational_resnet20_2_BB(), mix_exp_cifar.mix_exp_resnet20_2_BB(rational_inits=rational_inits, num_rationals=num_rationals)]

    elif resnet_args.run_all_two_layers and resnet_args.run_all_architecture == 'CIFAR10':
        model_names = ['relu_resnet20_2_layers', 'univ_rational_resnet20_2_layers', 'mix_experts_resnet20_2_layers']
        models_run_all = [relu_cifar.relu_resnet20_2_layers(), univ_rat_cifar.univ_rational_resnet20_2_layers(), mix_exp_cifar.mix_exp_resnet20_2_layers(rational_inits=rational_inits, num_rationals=num_rationals)]
    elif resnet_args.run_all_two_layers and resnet_args.run_all_architecture == 'ImageNet':
        model_names = ['relu_resnet18_2_layers', 'univ_rational_resnet18_2_layers', 'mix_experts_resnet18_2_layers']
        models_run_all = [relu_imagenet.relu_resnet18_2_layers(), univ_rat_imagenet.univ_rational_resnet18_2_layers(), mix_exp_imagenet.mix_exp_resnet18_2_layers(rational_inits=rational_inits, num_rationals=num_rationals)]

    elif resnet_args.run_all_one_layer and resnet_args.run_all_architecture == 'CIFAR10':
        model_names = ['relu_resnet20_1_layer', 'univ_rational_resnet20_1_layer', 'mix_experts_resnet20_1_layer']
        models_run_all = [relu_cifar.relu_resnet20_1_layer(), univ_rat_cifar.univ_rational_resnet20_1_layer(), mix_exp_cifar.mix_exp_resnet20_1_layer(rational_inits=rational_inits, num_rationals=num_rationals)]
    else:
        model_names = ['relu_resnet18_1_layer', 'univ_rational_resnet18_1_layer', 'mix_experts_resnet18_1_layer']
        models_run_all = [relu_imagenet.relu_resnet18_1_layer(), univ_rat_imagenet.univ_rational_resnet18_1_layer(), mix_exp_imagenet.mix_exp_resnet18_1_layer(rational_inits=rational_inits, num_rationals=num_rationals)]

    accuracy_plot_x_vals = []
    train_accs = []
    val_accs = []
    test_accs = []
    best_test_accs = []
    avg_time = []
    PATHS = []
    for m in range(len(models_run_all)):
        model = models_run_all[m]
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()

        scheduler, optimizer = utils.get_scheduler_optimizer(resnet_args.warmup_iterations, resnet_args.learning_rate, model, it_per_ep)
        model, cm, time_elapsed_epoch, best_acc, train_acc_plot_y_vals, val_acc_plot_y_vals, test_acc_plot_y_vals, accuracy_plot_x_vals = tvt.train_val_test_model(model, criterion, optimizer, scheduler,
                                                                                                                                                                   num_epochs=resnet_args.training_number_of_epochs,
                                                                                                                                                                   testloader=testloader, valloader=valloader,
                                                                                                                                                                   trainset=trainset, trainloader=trainloader,
                                                                                                                                                                   testset=testset, valset=valset)

        if resnet_args.save_res_csv:
            PATH = utils.make_csv(model_names[m], accuracy_plot_x_vals, train_acc_plot_y_vals, val_acc_plot_y_vals, test_acc_plot_y_vals)
            PATHS.append(PATH)
        train_accs.append(train_acc_plot_y_vals)
        val_accs.append(val_acc_plot_y_vals)
        test_accs.append(test_acc_plot_y_vals)
        best_test_accs.append(best_acc)
        avg_time.append(time_elapsed_epoch)

    plots.plot_overview_all(train_accs, val_accs, test_accs, accuracy_plot_x_vals, best_test_accs, avg_time)
    utils.make_yaml(model_names, PATHS)


def run_one():
    rational_inits = resnet_args.initialize_rationals  # TODO: catch exceptions
    num_rationals = 0
    if resnet_args.model == 'relu_resnet20':
        model = relu_cifar.relu_resnet20()
    elif resnet_args.model == 'relu_resnet20_2_BB':
        model = relu_cifar.relu_resnet20_2_BB()
    elif resnet_args.model == 'relu_resnet20_2_layers':
        model = relu_cifar.relu_resnet20_2_layers()
    elif resnet_args.model == 'relu_resnet20_1_layer':
        model = relu_cifar.relu_resnet20_1_layer()

    elif resnet_args.model == 'univ_rational_resnet20':
        model = univ_rat_cifar.univ_rational_resnet20()
        num_rationals = 2
    elif resnet_args.model == 'univ_rational_resnet20_2_BB':
        model = univ_rat_cifar.univ_rational_resnet20_2_BB()
        num_rationals = 2
    elif resnet_args.model == 'univ_rational_resnet20_2_layers':
        model = univ_rat_cifar.univ_rational_resnet20_2_layers()
        num_rationals = 2
    elif resnet_args.model == 'univ_rational_resnet20_1_layer':
        model = univ_rat_cifar.univ_rational_resnet20_1_layer()
        num_rationals = 2

    elif resnet_args.model == 'mix_experts_resnet20':
        model = mix_exp_cifar.mix_exp_resnet20(rational_inits=rational_inits, num_rationals=num_rationals)
        num_rationals = len(rational_inits) * 2
    elif resnet_args.model == 'mix_experts_resnet20_2_BB':
        model = mix_exp_cifar.mix_exp_resnet20_2_BB(rational_inits=rational_inits, num_rationals=num_rationals)
        num_rationals = len(rational_inits) * 2
    elif resnet_args.model == 'mix_experts_resnet20_2_layers':
        model = mix_exp_cifar.mix_exp_resnet20_2_layers(rational_inits=rational_inits, num_rationals=num_rationals)
        num_rationals = len(rational_inits) * 2
    elif resnet_args.model == 'mix_experts_resnet20_1_layer':
        model = mix_exp_cifar.mix_exp_resnet20_1_layer(rational_inits=rational_inits, num_rationals=num_rationals)
        num_rationals = len(rational_inits) * 2

    elif resnet_args.model == 'relu_resnet18':
        model = relu_imagenet.relu_resnet18()
    elif resnet_args.model == 'relu_resnet18_2_layers':
        model = relu_imagenet.relu_resnet18_2_layers()
    elif resnet_args.model == 'relu_resnet20_1_layer':
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
        model = mix_exp_imagenet.mix_exp_resnet18(rational_inits=rational_inits, num_rationals=num_rationals)
        num_rationals = len(rational_inits) * 2
    elif resnet_args.model == 'mix_experts_resnet18_2_layers':
        model = mix_exp_imagenet.mix_exp_resnet18_2_layers(rational_inits=rational_inits, num_rationals=num_rationals)
        num_rationals = len(rational_inits) * 2
    else:
        model = mix_exp_imagenet.mix_exp_resnet18_1_layer(rational_inits=rational_inits, num_rationals=num_rationals)
        num_rationals = len(rational_inits) * 2

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    scheduler, optimizer = utils.get_scheduler_optimizer(resnet_args.warmup_iterations, resnet_args.learning_rate, model, it_per_ep)
    model, cm, avg_time, best_test_acc, train_acc_plot_y_vals, val_acc_plot_y_vals, test_acc_plot_y_vals, accuracy_plot_x_vals = tvt.train_val_test_model(model, criterion, optimizer=optimizer, scheduler=scheduler,
                                                                                                                                                          num_epochs=resnet_args.training_number_of_epochs,
                                                                                                                                                          testloader=testloader, valloader=valloader,
                                                                                                                                                          trainset=trainset, trainloader=trainloader,
                                                                                                                                                          testset=testset, valset=valset)

    plots.final_plot(cm, avg_time, best_test_acc, resnet_args.training_number_of_epochs, resnet_args.learning_rate,
                     num_rationals, resnet_args.dataset, resnet_args.model, resnet_args.batch_size)
    if resnet_args.model == 'mix_experts_resnet20':
        plots.plot_activation_func_overview(model, num_rationals, rational_inits)
    models = [resnet_args.model]
    PATH = ''
    if resnet_args.save_res_csv:
        PATH = utils.make_csv(resnet_args.model, accuracy_plot_x_vals, train_acc_plot_y_vals, val_acc_plot_y_vals, test_acc_plot_y_vals)
    utils.make_yaml(models, PATH)
    time_stamp = datetime.now()
    PATH = './Saved_Models/{}.pth'.format(time_stamp)
    torch.save(model, PATH)


if resnet_args.train_all:
    run_all()
else:
    run_one()
