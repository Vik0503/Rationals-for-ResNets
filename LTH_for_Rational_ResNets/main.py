import inspect
import os
import sys

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
from LTH_for_Rational_ResNets.LTH_Models import univ_rational_resnet_imagenet as rrn18, relu_resnet_cifar10 as rn20
from LTH_for_Rational_ResNets.LTH_Models import univ_rational_resnet_cifar10 as rrn20
from LTH_for_Rational_ResNets.LTH_Models import relu_resnet_imagenet as rn18
from LTH_for_Rational_ResNets.LTH_Models import mix_experts_resnet_cifar10 as sel2exp
from LTH_for_Rational_ResNets.LTH_Models import select_1_expert_group_rational_resnet as sel1exp
from LTH_for_Rational_ResNets.Mask import make_initial_mask
from LTH_for_Rational_ResNets import LTH_write_read_csv

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


rational_inits = LTH_args.initialize_rationals
num_rationals = len(rational_inits)

if LTH_args.prune_shortcuts:
    prune_shortcuts = True
else:
    prune_shortcuts = False

if LTH_args.dataset == 'cifar10':
    trainset = cifar10.get_trainset()
    valset = cifar10.get_validationset()
    testset = cifar10.get_testset()
    trainloader = cifar10.get_trainloader(bs=LTH_args.batch_size)
    valloader = cifar10.get_valloader(bs=LTH_args.batch_size)
    testloader = cifar10.get_testloader(bs=LTH_args.batch_size)
    classes = cifar10.get_classes()
    num_classes = cifar10.get_num_classes()
    it_per_ep = cifar10.get_it_per_epoch(bs=LTH_args.batch_size)

elif LTH_args.dataset == 'SVHN':
    trainset = SVHN.get_trainset()
    valset = SVHN.get_validationset()
    testset = SVHN.get_testset()
    trainloader = SVHN.get_trainloader(bs=LTH_args.batch_size)
    valloader = SVHN.get_valloader(bs=LTH_args.batch_size)
    testloader = SVHN.get_testloader(bs=LTH_args.batch_size)
    classes = SVHN.get_classes()
    num_classes = SVHN.get_num_classes()
    it_per_ep = SVHN.get_it_per_epoch(bs=LTH_args.batch_size)


def run_all():

    num_epochs = 0
    test_accuracies = []
    sparsities = []
    num_epoch_list = []
    PATHS = []
    if LTH_args.run_all_classic:
        model_names = ['relu_resnet20', 'univ_rational_resnet20', 'mix_experts_resnet20']
        models_run_all = [rn20.relu_resnet20(), rrn20.univ_rational_resnet20(), sel2exp.mix_exp_resnet20(rational_inits=rational_inits, num_rationals=num_rationals)]
    elif LTH_args.run_all_two_BB:
        model_names = ['relu_resnet20_2_BB', 'univ_rational_resnet20_2_BB', 'mix_experts_resnet20_2_BB']
        models_run_all = [rn20.relu_resnet20_2_BB(), rrn20.univ_rational_resnet20_2_BB(), sel2exp.mix_exp_resnet20_2_BB(rational_inits=rational_inits, num_rationals=num_rationals)]
    elif LTH_args.run_all_two_layers:
        model_names = ['relu_resnet20_2_layers', 'univ_rational_resnet20_2_layers', 'mix_experts_resnet20_2_layers']
        models_run_all = [rn20.relu_resnet20_2_layers(), rrn20.univ_rational_resnet20_2_layers(), sel2exp.mix_exp_resnet20_2_layers(rational_inits=rational_inits, num_rationals=num_rationals)]
    else:
        model_names = ['relu_resnet20_1_layer', 'univ_rational_resnet20_1_layer', 'mix_experts_resnet20_1_layer']
        models_run_all = [rn20.relu_resnet20_1_layer(), rrn20.univ_rational_resnet20_1_layer(), sel2exp.mix_exp_resnet20_1_layer(rational_inits=rational_inits, num_rationals=num_rationals)]

    criterion = nn.CrossEntropyLoss()
    checkpoints = []
    all_test_accuracies = []
    all_sparsities = []
    all_models = []

    for m in range(len(models_run_all)):
        model = models_run_all[m]
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model = model.to(device)
        mask = make_initial_mask(model)
        mask = mask.cuda()

        if LTH_args.stop_criteria is 'test_acc':
            num_epochs, test_accuracies, sparsities, path, last_checkpoint = Lottery_Ticket_Hypothesis.iterative_pruning_by_test_acc(model, mask,
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
            test_accuracies, sparsities, path, last_checkpoint = Lottery_Ticket_Hypothesis.iterative_pruning_by_num(model, mask, LTH_args.iterative_pruning_epochs, criterion=criterion,
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

        if LTH_args.save_res_csv:
            PATH = LTH_write_read_csv.make_csv(model_names[m], sparsities, test_accuracies)
            PATHS.append(PATH)

        all_test_accuracies.append(test_accuracies)
        all_sparsities.append(sparsities)
        num_epoch_list.append(num_epochs)
        checkpoints.append(last_checkpoint)
        all_models.append(model)

    plots.plot_all(test_accs=all_test_accuracies, sparsities=all_sparsities, num_epoch_list=num_epoch_list)
    plots.plot_activation_func_overview(all_models[2], LTH_args.init_rationals, num_rationals)
    mask_path_dim, mask_path_weights = LTH_write_read_csv.make_mask_csv(all_models)
    LTH_write_read_csv.make_yaml(model_names, csv=PATHS, saved_models=path, table=[mask_path_dim, mask_path_weights])


def run_one():
    global model

    if LTH_args.model == 'relu_resnet20':
        model = rn20.relu_resnet20()
    elif LTH_args.model == 'relu_resnet20_2_BB':
        model = rn20.relu_resnet20_2_BB()
    elif LTH_args.model == 'relu_resnet20_2_layers':
        model = rn20.relu_resnet20_2_layers()
    elif LTH_args.model == 'relu_resnet20_1_layer':
        model = rn20.relu_resnet20_1_layer()

    elif LTH_args.model == 'univ_rational_resnet20':
        model = rrn20.univ_rational_resnet20()
    elif LTH_args.model == 'univ_rational_resnet20_2_BB':
        model = rrn20.univ_rational_resnet20_2_BB()
    elif LTH_args.model == 'univ_rational_resnet20_2_layers':
        model = rrn20.univ_rational_resnet20_2_layers()
    elif LTH_args.model == 'univ_rational_resnet20_1_layer':
        model = rrn20.univ_rational_resnet20_1_layer()

    elif LTH_args.model == 'mix_experts_resnet20':
        model = sel2exp.mix_exp_resnet20(rational_inits=rational_inits, num_rationals=num_rationals)
    elif LTH_args.model == 'mix_experts_resnet20_2_BB':
        model = sel2exp.mix_exp_resnet20_2_BB(rational_inits=rational_inits, num_rationals=num_rationals)
    elif LTH_args.model == 'mix_experts_resnet20_2_layers':
        model = sel2exp.mix_exp_resnet20_2_layers(rational_inits=rational_inits, num_rationals=num_rationals)
    elif LTH_args.model == 'mix_experts_resnet20_1_layer':
        model = sel2exp.mix_exp_resnet20_1_layer(rational_inits=rational_inits, num_rationals=num_rationals)

    elif LTH_args.model == 'rational_resnet18_imagenet':
        model = rrn18.rational_resnet18()
    elif LTH_args.model == 'resnet18_imagenet':
        model = rn18.relu_resnet18()

    else:
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
        num_epochs, test_accuracies, sparsities, path = Lottery_Ticket_Hypothesis.iterative_pruning_by_test_acc(model, mask,
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

    PATH = ''
    if LTH_args.save_res_csv:
        PATH = LTH_write_read_csv.make_csv(LTH_args.model, sparsities, test_accuracies)

    LTH_write_read_csv.make_yaml([LTH_args.model], csv=PATH, saved_models=path)


if LTH_args.run_all_classic or LTH_args.run_all_two_BB or LTH_args.run_all_two_layers or LTH_args.run_all_one_layer:
    run_all()
else:
    run_one()
