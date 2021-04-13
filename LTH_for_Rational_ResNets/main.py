import inspect
import os
import sys
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import torch

torch.cuda.manual_seed_all(42)
from torch import nn
import numpy as np

np.random.seed(42)

from LTH_for_Rational_ResNets import Lottery_Ticket_Hypothesis
from LTH_for_Rational_ResNets import plots
from LTH_for_Rational_ResNets import argparser
from LTH_for_Rational_ResNets.Datasets import CIFAR10 as cifar10
from LTH_for_Rational_ResNets.Datasets import SVHN
from LTH_for_Rational_ResNets.LTH_Models import rational_resnet18_imagenet as rrn18, resnet20_cifar10 as rn20
from LTH_for_Rational_ResNets.LTH_Models import rational_resnet20_cifar10 as rrn20
from LTH_for_Rational_ResNets.LTH_Models import resnet18_imagenet as rn18
from LTH_for_Rational_ResNets.LTH_Models import select_2_expert_groups_rational_resnet as sel2exp
from LTH_for_Rational_ResNets.LTH_Models import select_1_expert_group_rational_resnet as sel1exp
from LTH_for_Rational_ResNets.Mask import make_initial_mask

LTH_arg_parser = argparser.get_argparser()
LTH_args = argparser.get_arguments()

global trainset
global valset
global testset
global trainloader
global valloader
global testloader
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

plt.style.use(["science", "grid"])
matplotlib.rcParams.update({
    "font.family": "serif",
    "text.usetex": False,
})

if LTH_args.initialize_rationals:
    rational_inits = LTH_args.initialize_rationals  # TODO: catch exceptions
    num_rationals = len(rational_inits)

if LTH_args.prune_shortcuts:
    prune_shortcuts = True
else:
    prune_shortcuts = False

if LTH_args.dataset is 'cifar10':
    trainset = cifar10.get_trainset()
    valset = cifar10.get_validationset()
    testset = cifar10.get_testset()
    trainloader = cifar10.get_trainloader(bs=LTH_args.batch_size)
    valloader = cifar10.get_valloader(bs=LTH_args.batch_size)
    testloader = cifar10.get_testloader(bs=LTH_args.batch_size)
    classes = cifar10.get_classes()
    num_classes = cifar10.get_num_classes()
    it_per_ep = cifar10.get_it_per_epoch(bs=LTH_args.batch_size)

elif LTH_args.dataset is 'SVHN':
    trainset = SVHN.get_trainset()
    valset = SVHN.get_validationset()
    testset = SVHN.get_testset()
    trainloader = SVHN.get_trainloader(bs=LTH_args.batch_size)
    valloader = SVHN.get_valloader(bs=LTH_args.batch_size)
    testloader = SVHN.get_testloader(bs=LTH_args.batch_size)
    classes = SVHN.get_classes()
    num_classes = SVHN.get_num_classes()
    it_per_ep = SVHN.get_it_per_epoch(bs=LTH_args.batch_size)

models_run_all = [rrn20.rational_resnet20(), rn20.resnet20(), sel2exp.select_2_expert_groups_rational_resnet20(rational_inits=rational_inits, num_rationals=num_rationals)]


def run_all():  # TODO: Solve Problem with select

    plot_labels = ['ResNet20 univ. Rat.', 'ResNet20 Original', 'ResNet20 mixture of 5 univ. Rat.']
    plot_colors = ['C1', 'C0', 'C2']
    plt.figure(figsize=(10, 6))
    plt.subplot(121)
    plt.xlabel('Percent of Pruned Weights')
    plt.ylabel('Test Accuracy in Percent')

    num_epochs = 0
    test_accuracies = []
    sparsities = []
    num_epoch_list = []
    criterion = nn.CrossEntropyLoss()

    for m in range(len(models_run_all)):
        model = models_run_all[m]
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model = model.to(device)
        mask = make_initial_mask(model)
        mask = mask.cuda()

        if LTH_args.stop_criteria is 'test_acc':
            num_epochs, test_accuracies, sparsities = Lottery_Ticket_Hypothesis.iterative_pruning_by_test_acc(model, mask,
                                                                                                              LTH_args.test_accuracy_threshold,
                                                                                                              criterion=criterion,
                                                                                                              testset=testset, testloader=testloader,
                                                                                                              trainset=trainset,
                                                                                                              trainloader=trainloader,
                                                                                                              valloader=valloader, valset=valset,
                                                                                                              pruning_percentage=LTH_args.pruning_percentage,
                                                                                                              training_number_of_epochs=LTH_args.training_number_of_epochs,
                                                                                                              lr=LTH_args.learning_rate,
                                                                                                              it_per_epoch=it_per_ep,
                                                                                                              num_warmup_it=LTH_args.warmup_iterations)

        elif LTH_args.stop_criteria is 'num_prune_epochs':
            test_accuracies, sparsities = Lottery_Ticket_Hypothesis.iterative_pruning_by_num(model, mask, LTH_args.iterative_pruning_epochs, criterion=criterion,
                                                                                             testset=testset,
                                                                                             testloader=testloader, trainset=trainset,
                                                                                             trainloader=trainloader, valloader=valloader, valset=valset,
                                                                                             pruning_percentage=LTH_args.pruning_percentage,
                                                                                             training_number_of_epochs=LTH_args.training_number_of_epochs,
                                                                                             lr=LTH_args.learning_rate,
                                                                                             it_per_epoch=it_per_ep,
                                                                                             num_warmup_it=LTH_args.warmup_iterations)
            num_epochs = LTH_args.iterative_pruning_epochs

        elif LTH_args.stop_criteria is 'one_shot':
            Lottery_Ticket_Hypothesis.one_shot_pruning(model, criterion=criterion, testset=testset,
                                                       testloader=testloader, trainset=trainset, trainloader=trainloader,
                                                       valloader=valloader, valset=valset, prune_mask=mask,
                                                       pruning_percentage=LTH_args.pruning_percentage,
                                                       training_number_of_epochs=LTH_args.training_number_of_epochs,
                                                       lr=LTH_args.learning_rate,
                                                       it_per_epoch=it_per_ep,
                                                       num_warmup_it=LTH_args.warmup_iterations)

        plt.plot(sparsities, test_accuracies, label=plot_labels[m], color=plot_colors[m])
        num_epoch_list.append(num_epochs)

    plt.subplots_adjust(bottom=0.3)
    plt.legend(['ResNet20 univ. Rat.', 'ResNet20 Original', 'ResNet20 mixture of 5 univ. Rat.'], bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=1)

    props = dict(boxstyle='round', facecolor='grey', alpha=0.5)
    text = 'dataset: {}, '.format(LTH_args.dataset) + 'batch size: {}, '.format(LTH_args.batch_size) + '\n' + '{} training epochs per pruning epoch, '.format(LTH_args.training_number_of_epochs) + '\n' + \
           'learning rate: {}, '.format(0.03) + '{}% pruning per epoch, '.format(LTH_args.pruning_percentage) + '\n' + '{} warm-up iterations'.format(LTH_args.warmup_iterations) + '\n' + 'shortcuts pruned: {}'.format(prune_shortcuts) + \
           '\n' + 'number of iterative pruning epochs: ' + '\n' + \
           '- ResNet20 univ. Rat: {}'.format(num_epoch_list[0]) + '\n' + '- ResNet20 Original: {}'.format(num_epoch_list[1]) + '\n' + '- ResNet20 mixture of 5 univ. Rat: {}'.format(num_epoch_list[2])

    plt.figtext(0.525, 0.5, text, bbox=props, size=9)

    time_stamp = datetime.now()
    PATH = './Results/LTH_all_models' + '/' + '{}'.format(time_stamp) + '_' + '{}'.format(LTH_args.dataset) + '.svg'
    plt.savefig(PATH)
    plt.show()


def run_one():
    global model
    if LTH_args.model is 'rational_resnet20_cifar10':
        model = rrn20.rational_resnet20()
    elif LTH_args.model is 'resnet20_cifar10':
        model = rn20.resnet20()
    elif LTH_args.model is 'rational_resnet18_imagenet':
        model = rrn18.rational_resnet18()
    elif LTH_args.model is 'resnet18_imagenet':
        model = rn18.resnet18()
    elif LTH_args.model is 'select_2_expert_groups_rational_resnet20':
        model = sel2exp.select_2_expert_groups_rational_resnet20(rational_inits=rational_inits, num_rationals=num_rationals)
    elif LTH_args.model is 'select_1_expert_group_rational_resnet20':
        model = sel1exp.select_1_expert_group_rational_resnet20(rational_inits=rational_inits, num_rationals=num_rationals)

    mask = make_initial_mask(model)
    mask = mask.cuda()
    criterion = nn.CrossEntropyLoss()

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    num_epochs = 0
    test_accuracies = []
    sparsities = []

    if LTH_args.stop_criteria is 'test_acc':
        num_epochs, test_accuracies, sparsities = Lottery_Ticket_Hypothesis.iterative_pruning_by_test_acc(model, mask,
                                                                                                          LTH_args.test_accuracy_threshold,
                                                                                                          criterion=criterion,
                                                                                                          testset=testset, testloader=testloader,
                                                                                                          trainset=trainset,
                                                                                                          trainloader=trainloader,
                                                                                                          valloader=valloader, valset=valset,
                                                                                                          pruning_percentage=LTH_args.pruning_percentage,
                                                                                                          training_number_of_epochs=LTH_args.training_number_of_epochs,
                                                                                                          lr=LTH_args.learning_rate,
                                                                                                          it_per_epoch=it_per_ep,
                                                                                                          num_warmup_it=LTH_args.warmup_iterations)

    elif LTH_args.stop_criteria is 'num_prune_epochs':
        test_accuracies, sparsities = Lottery_Ticket_Hypothesis.iterative_pruning_by_num(model, mask, LTH_args.iterative_pruning_epochs, criterion=criterion,
                                                                                         testset=testset,
                                                                                         testloader=testloader, trainset=trainset,
                                                                                         trainloader=trainloader, valloader=valloader, valset=valset,
                                                                                         pruning_percentage=LTH_args.pruning_percentage,
                                                                                         training_number_of_epochs=LTH_args.training_number_of_epochs,
                                                                                         lr=LTH_args.learning_rate,
                                                                                         it_per_epoch=it_per_ep,
                                                                                         num_warmup_it=LTH_args.warmup_iterations)
        num_epochs = LTH_args.iterative_pruning_epochs

    elif LTH_args.stop_criteria is 'one_shot':
        Lottery_Ticket_Hypothesis.one_shot_pruning(model, criterion=criterion, testset=testset,
                                                   testloader=testloader, trainset=trainset, trainloader=trainloader,
                                                   valloader=valloader, valset=valset, prune_mask=mask,
                                                   pruning_percentage=LTH_args.pruning_percentage,
                                                   training_number_of_epochs=LTH_args.training_number_of_epochs,
                                                   lr=LTH_args.learning_rate,
                                                   it_per_epoch=it_per_ep,
                                                   num_warmup_it=LTH_args.warmup_iterations)
        num_epochs = 1
    if LTH_args.stop_criteria is not 'one_shot':
        plots.make_LTH_test_acc_plot(test_accuracies, sparsities)
        plots.final_plot_LTH(LTH_args.model, LTH_args.dataset, LTH_args.batch_size, num_epochs,
                             LTH_args.training_number_of_epochs, LTH_args.learning_rate, LTH_args.pruning_percentage, LTH_args.warmup_iterations, prune_shortcuts)


if LTH_args.run_all:
    run_all()
else:
    run_one()


