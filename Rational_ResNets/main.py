import os
import sys
import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import torch

torch.cuda.manual_seed_all(42)
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import numpy as np

np.random.seed(42)

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

resnet_args = argparser.get_args()

global trainset
global valset
global testset
global trainloader
global valloader
global testloader
global classes
global num_classes
global model
global num_rationals

if resnet_args.initialize_rationals:
    rational_inits = resnet_args.initialize_rationals  # TODO: catch exceptions
    num_rationals = len(rational_inits)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

if resnet_args.dataset is 'cifar10':
    trainset, trainloader = CIFAR10.get_train_data(aug=resnet_args.augment_data, bs=resnet_args.batch_size)
    valset, valloader = CIFAR10.get_validation_data(aug=resnet_args.augment_data, bs=resnet_args.batch_size)
    testset, testloader = CIFAR10.get_test_data(aug=resnet_args.augment_data, bs=resnet_args.batch_size)
    classes = CIFAR10.get_classes()
    num_classes = CIFAR10.get_num_classes()
    # it_per_ep = CIFAR10.get_it_per_epoch(bs=resnet_args.batch_size)

elif resnet_args.dataset is 'SVHN':
    trainset, trainloader = SVHN.get_train_data(aug=resnet_args.augment_data, bs=resnet_args.batch_size)
    valset, valloader = SVHN.get_validation_data(aug=resnet_args.augment_data, bs=resnet_args.batch_size)
    testset, testloader = SVHN.get_test_data(aug=resnet_args.augment_data, bs=resnet_args.batch_size)
    classes = SVHN.get_classes()
    num_classes = SVHN.get_num_classes()
    # it_per_ep = SVHN.get_it_per_epoch(bs=resnet_args.batch_size)


def get_scheduler_optimizer(num_warmup_it, lr, model, it_per_ep):  # TODO: allow diff. milestones maybe in utils?
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)

    def lr_lambda(it):
        if it < num_warmup_it:
            if it % 500 == 0:
                print('Warmup')
            return min(1.0, it / num_warmup_it)
        elif it < 10 * it_per_ep:
            if it % 500 == 0:
                print('MS 1')
            return 1
        elif 10 * it_per_ep <= it < 15 * it_per_ep:
            if it % 500 == 0:
                print('MS 2')
            return 0.1
        elif 15 * it_per_ep <= it < 20 * it_per_ep:
            if it % 500 == 0:
                print('MS 3')
            return 0.01
        elif it >= 20 * it_per_ep:
            if it % 500 == 0:
                print('After MS 3')
            return 0.001

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda), optimizer


models_run_all = [RRN20.rational_resnet20(), RN20.resnet20(), sel2exp.select_2_expert_groups_rational_resnet20(rational_inits=rational_inits, num_rationals=num_rationals)]


def run_all():
    global accuracy_plot_x_vals
    train_accs = []
    val_accs = []
    test_accs = []
    best_test_accs = []
    avg_time = []
    for m in range(len(models_run_all)):
        model = models_run_all[m]
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()

        scheduler, optimizer = get_scheduler_optimizer(resnet_args.warmup_iterations, resnet_args.learning_rate, model, 430)  # TODO: find solution for it_per_ep
        model, cm, time_elapsed_epoch, best_acc, train_acc_plot_y_vals, val_acc_plot_y_vals, test_acc_plot_y_vals, accuracy_plot_x_vals = tvt.train_val_test_model(model, criterion, optimizer, scheduler,
                                                                                                                                                                   num_epochs=resnet_args.training_number_of_epochs,
                                                                                                                                                                   testloader=testloader, valloader=valloader,
                                                                                                                                                                   trainset=trainset, trainloader=trainloader,
                                                                                                                                                                   testset=testset, valset=valset)
        train_accs.append(train_acc_plot_y_vals)
        val_accs.append(val_acc_plot_y_vals)
        test_accs.append(test_acc_plot_y_vals)
        best_test_accs.append(best_acc)
        avg_time.append(time_elapsed_epoch)
    plots.plot_overview_all(train_accs, val_accs, test_accs, accuracy_plot_x_vals, best_test_accs, avg_time)


def run_one():
    return None


if resnet_args.train_all:
    run_all()
else:
    run_one()

if resnet_args.model is 'rational_resnet20_cifar10':  # TODO: add rest of the models
    model = RRN20.rational_resnet20()
    num_rationals = 2
elif resnet_args.model is 'resnet20_cifar10':
    model = RN20.resnet20()
    num_rationals = 0
elif resnet_args.model is 'rational_resnet18_imagenet':
    model = RRN18.rational_resnet18()
    num_rationals = 2
elif resnet_args.model is 'resnet18_imagenet':
    model = RN18.resnet18()
    num_rationals = 0
elif resnet_args.model is 'pt':
    model = PT.resnet18()
    num_rationals = 2
elif resnet_args.model is 'recurrent_rational_resnet20_cifar10':
    model = RecRRN20.rational_resnet20()
    num_rationals = 1
elif resnet_args.model is 'resnet110_cifar10':
    model = RN20.resnet110()
    num_rationals = 0
elif resnet_args.model is 'rational_resnet110_cifar10':
    model = RRN20.rational_resnet110()
    num_rationals = 2
elif resnet_args.model is 'select_2_expert_groups_rational_resnet20':
    model = sel2exp.select_2_expert_groups_rational_resnet20(num_rationals=num_rationals, rational_inits=rational_inits)
elif resnet_args.model is 'select_1_expert_group_rational_resnet20':
    model = sel1exp.select_1_expert_group_rational_resnet20(num_rationals=num_rationals, rational_inits=rational_inits)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

model, cm, avg_time, best_test_acc = tvt.train_val_test_model(model, criterion, optimizer, exp_lr_scheduler,
                                                              num_epochs=resnet_args.training_number_of_epochs,
                                                              testloader=testloader, valloader=valloader,
                                                              trainset=trainset, trainloader=trainloader,
                                                              testset=testset, valset=valset)

plots.final_plot(cm, avg_time, best_test_acc, resnet_args.training_number_of_epochs, resnet_args.learning_rate,
                 num_rationals, resnet_args.dataset, resnet_args.model, resnet_args.batch_size)
