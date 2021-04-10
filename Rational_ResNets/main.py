import os
import sys
import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler


from Rational_ResNets import argparser
from Rational_ResNets import train_val_test as tvt
from Rational_ResNets import plots
from Rational_ResNets.ResNet_Datasets import CIFAR10, SVHN
from Rational_ResNets.ResNet_Models import Pytorch_Rational_ResNets_ImageNet as PT
from Rational_ResNets.ResNet_Models import Rational_ResNet18_ImageNet as RRN18
from Rational_ResNets.ResNet_Models import Rational_ResNet20_CIFAR10 as RRN20
from Rational_ResNets.ResNet_Models import ResNet18_ImageNet as RN18
from Rational_ResNets.ResNet_Models import ResNet20_CIFAR10 as RN20
# from Rational_ResNets.ResNet_Models import select_1_expert_group_rational_resnet
from Rational_ResNets.ResNet_Models import Recurrent_Rational_ResNet20_CIFAR10 as RecRRN20
from Rational_ResNets.ResNet_Models import select_2_expert_groups_rational_resnet as sel2exp

resnet_argparser = argparser.get_argparser()
resnet_args = resnet_argparser.parse_args(
    ['--model', 'select_2_expert_groups_rational_resnet', '--dataset', 'SVHN', '--training_number_of_epochs', '2', '--augment_data', 'True', '--number_of_rationals_per_vector', '5', '--initialize_rationals', 'leaky_relu', 'gelu', 'swish', 'tanh', 'sigmoid'])

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

elif resnet_args.dataset is 'SVHN':
    trainset, trainloader = SVHN.get_train_data(aug=resnet_args.augment_data, bs=resnet_args.batch_size)
    valset, valloader = SVHN.get_validation_data(aug=resnet_args.augment_data, bs=resnet_args.batch_size)
    testset, testloader = SVHN.get_test_data(aug=resnet_args.augment_data, bs=resnet_args.batch_size)
    classes = SVHN.get_classes()
    num_classes = SVHN.get_num_classes()

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
elif resnet_args.model is 'select_2_expert_groups_rational_resnet':
    model = sel2exp.select_2_expert_groups_rational_resnet20(num_rationals=num_rationals, rational_inits=rational_inits)
    num_rationals = resnet_args.number_of_rationals_per_vector

model = model.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=resnet_args.learning_rate, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # TODO: Update scheduler

model, cm, avg_time, best_test_acc = tvt.train_val_test_model(model, criterion, optimizer, exp_lr_scheduler,
                                                              num_epochs=resnet_args.training_number_of_epochs,
                                                              testloader=testloader, valloader=valloader,
                                                              trainset=trainset, trainloader=trainloader,
                                                              testset=testset, valset=valset)

plots.final_plot(cm, avg_time, best_test_acc, resnet_args.training_number_of_epochs, resnet_args.learning_rate,
                 num_rationals, resnet_args.dataset, resnet_args.model, resnet_args.batch_size)

