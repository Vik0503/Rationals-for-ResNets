import inspect
import os
import sys

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import torch

torch.cuda.manual_seed_all(42)
import torch.nn as nn

import numpy as np

np.random.seed(42)

from datetime import datetime

from Rational_ResNets import argparser
from Rational_ResNets import train_val_test as tvt
from Rational_ResNets import plots
from Rational_ResNets.ResNet_Datasets import CIFAR10, SVHN
from Rational_ResNets.ResNet_Models import Pytorch_Rational_ResNets_ImageNet as PT
from Rational_ResNets.ResNet_Models import Rational_ResNet18_ImageNet as RRN18
from Rational_ResNets.ResNet_Models import Rational_ResNet20_CIFAR10 as RRN20
from Rational_ResNets.ResNet_Models import ResNet18_ImageNet as RN18
from Rational_ResNets.ResNet_Models import ResNet20_CIFAR10 as RN20
from Rational_ResNets.ResNet_Models import select_1_expert_group_rational_resnet as sel1exp
from Rational_ResNets.ResNet_Models import Recurrent_Rational_ResNet20_CIFAR10 as RecRRN20
from Rational_ResNets.ResNet_Models import select_2_expert_groups_rational_resnet as sel2exp
from Rational_ResNets import utils

resnet_args = argparser.get_args()

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
    models_run_all = [RN20.resnet20(), RRN20.rational_resnet20(), sel2exp.select_2_expert_groups_rational_resnet20(rational_inits=rational_inits, num_rationals=num_rationals)]
    model_names = ['resnet20_cifar10', 'rational_resnet20_cifar10', 'select_2_expert_groups_rational_resnet20']
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
    if resnet_args.model == 'rational_resnet20_cifar10':  # TODO: add rest of the models
        model = RRN20.rational_resnet20()
        num_rationals = 2
    elif resnet_args.model == 'resnet20_cifar10':
        model = RN20.resnet20()
        num_rationals = 0
    elif resnet_args.model == 'rational_resnet18_imagenet':
        model = RRN18.rational_resnet18()
        num_rationals = 2
    elif resnet_args.model == 'resnet18_imagenet':
        model = RN18.resnet18()
        num_rationals = 0
    elif resnet_args.model == 'pt':
        model = PT.resnet18()
        num_rationals = 2
    elif resnet_args.model == 'recurrent_rational_resnet20_cifar10':
        model = RecRRN20.rational_resnet20()
        num_rationals = 1
    elif resnet_args.model == 'resnet110_cifar10':
        model = RN20.resnet110()
        num_rationals = 0
    elif resnet_args.model == 'rational_resnet110_cifar10':
        model = RRN20.rational_resnet110()
        num_rationals = 2
    elif resnet_args.model == 'select_2_expert_groups_rational_resnet20':
        rational_inits = resnet_args.initialize_rationals  # TODO: catch exceptions
        num_rationals = len(rational_inits)
        model = sel2exp.select_2_expert_groups_rational_resnet20(num_rationals=num_rationals, rational_inits=rational_inits)
    elif resnet_args.model == 'select_1_expert_group_rational_resnet20':
        rational_inits = resnet_args.initialize_rationals  # TODO: catch exceptions
        num_rationals = len(rational_inits)
        model = sel1exp.select_1_expert_group_rational_resnet20(num_rationals=num_rationals, rational_inits=rational_inits)

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
    if resnet_args.model == 'select_2_expert_groups_rational_resnet20':
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
