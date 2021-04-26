import inspect
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
from rational.torch import Rational

import argparser
from LTH_for_Rational_ResNets.LTH_Models import select_2_expert_groups_rational_resnet as sel
from LTH_for_Rational_ResNets.LTH_Models.select_2_expert_groups_rational_resnet import RationalBasicBlock

plt.style.use(["science", "grid"])
matplotlib.rcParams.update({
    "font.family": "serif",
    "text.usetex": False,
})

LTH_args = argparser.get_arguments()


def make_LTH_test_acc_plot(test_accuracies: list, sparsity: list):
    """
    Plot test accuracy for each pruning epoch.

    Parameters
    ----------
    test_accuracies: list
                     A list containing the test accuracies after every pruning epoch.
    sparsity: list
              A list containing the different sparsity for each pruning epoch.
    """

    plt.figure(figsize=(10, 6))
    plt.subplot(121)
    plt.plot(sparsity, test_accuracies)
    plt.xlabel('Percent of Pruned Weights')
    plt.ylabel('Test Accuracy in Percent')
    plt.subplots_adjust(bottom=0.3)
    plt.legend(['Test Accuracy'], bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=1)
    return plt


def final_plot_LTH(model, dataset, batch_size, num_pruning_epochs, training_number_of_epochs, learning_rate, pruning_percentage, warmup_iterations, prune_shortcuts):  # TODO: Use argparser
    props = dict(boxstyle='round', facecolor='grey', alpha=0.5)
    text = 'model: {}, '.format(model) + 'dataset: {}, '.format(dataset) + '\n' + 'batch size: {}, '.format(batch_size) + '{} iterative pruning epochs, '.format(num_pruning_epochs) \
           + '\n' + '{} training epochs per pruning epoch, '.format(training_number_of_epochs) + '\n' + \
           'learning rate: {}, '.format(learning_rate) + '{}% pruning per epoch, '.format(pruning_percentage) + '\n' + '{} warm-up iterations, '.format(warmup_iterations) + 'shortcuts pruned: {}'.format(prune_shortcuts)

    plt.figtext(0.525, 0.5, text, bbox=props, size=9)

    time_stamp = datetime.now()
    PATH = './Results/LTH_{}'.format(model) + '/' + '{}'.format(time_stamp) + '_' + '{}'.format(model) + '_' + '{}'.format(dataset) + '.svg'
    plt.savefig(PATH)
    plt.show()


def plot_all(test_accs, sparsities, num_epoch_list):
    plt.figure(figsize=(10, 6))
    plt.subplot(121)
    ax = plt.gca()
    plt.xlabel('Percent of Pruned Weights')
    plt.ylabel('Test Accuracy in Percent')

    for t in range(len(test_accs)):
        ax.plot(sparsities[t], test_accs[t])

    def forward(x):
        return 1.2 ** x

    def inverse(x):
        return np.log(x)

    def forward_x(x):
        return 1.03 ** abs(x)

    def inverse_x(x):
        return np.log(abs(x))

    ax.set_xscale('function', functions=(forward_x, inverse_x))
    plt.xticks([0, 20, 40, 60, 80, 100])
    ax.set_yscale('function', functions=(forward, inverse))

    plt.subplots_adjust(bottom=0.3)
    plt.legend(['ReLU ResNet20', 'univ. rational ResNet20', 'mix. exp. ResNet20'], bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=1)

    props = dict(boxstyle='round', facecolor='grey', alpha=0.5)

    if LTH_args.stop_criteria is 'test_acc':
        stop_criterion = 'terminate at 90% test accuracy'
    else:
        stop_criterion = 'terminate after {} epochs'.format(LTH_args.iterative_pruning_epochs)
    text = 'dataset: {}, '.format(LTH_args.dataset) + 'batch size: {}, '.format(LTH_args.batch_size) + '\n' + '{} training epochs per pruning epoch, '.format(LTH_args.training_number_of_epochs) + '\n' + \
           'learning rate: {}, '.format(0.03) + '{}% pruning per epoch, '.format(LTH_args.pruning_percentage) + '\n' + '{} warm-up iterations, '.format(LTH_args.warmup_iterations) + '\n' + \
           'shortcuts pruned: {}, '.format(LTH_args.prune_shortcuts) + '\n' + 'number of iterative pruning epochs: ' + '\n' + \
           '- ReLU ResNet20: {}'.format(num_epoch_list[0]) + '\n' + '- univ. rational ResNet20: {}'.format(num_epoch_list[1]) + '\n' + '- mix. exp. ResNet20: {}'.format(num_epoch_list[2]) + '\n' + stop_criterion

    plt.figtext(0.525, 0.5, text, bbox=props, size=9)
    time_stamp = datetime.now()

    PATH = './Results/LTH_all_models/{}'.format(time_stamp) + '_{}'.format(LTH_args.dataset) + '.svg'
    plt.savefig(PATH)
    plt.show()


def activation_function_plots(model):
    """ Plot all Rational Activation Functions. """
    legend = []
    for mod in model.modules():
        if isinstance(mod, Rational):
            x = mod.show(display=False)['line']['x']
            y = mod.show(display=False)['line']['y']
            plt.plot(x, y)
            legend.append(mod)

    plt.legend(legend, bbox_to_anchor=(0.5, -0.4), ncol=1, loc='center')
    plt.tight_layout()
    time_stamp = datetime.now()

    PATH = './Results/activation_functions/{}'.format(time_stamp) + '_{}'.format(LTH_args.dataset) + '.svg'
    plt.savefig(PATH)
    plt.show()


def calc_mixture_plot(alphas, rationals):
    all_x = torch.zeros(5, 600)
    for i in range(len(alphas)):
        alpha_x = rationals[i].show(display=False)['line']['y'] * alphas[i].detach().numpy()
        all_x[i] = torch.tensor(alpha_x)
    return all_x.sum(dim=0), rationals[0].show(display=False)['line']['x']


def plot_activation_func_overview(model, num_rat, inits):
    c = 0
    tmp = []
    rat_groups = []
    alphas = []
    for m in model.modules():
        if isinstance(m, Rational):
            tmp.append(m)
            c += 1
        if c == num_rat:  # TODO: c to num_rats
            rat_groups.append(tmp)
            tmp = []
            c = 0

    for m, param in model.named_parameters():
        if 'alpha' in m:
            alphas.append(param)

    layers = model.layers

    plt.figure(figsize=(layers[0] * 8, len(layers) * 8))
    model_1 = False  # TODO: catch model 1

    plt_counter = 0
    for r in range(1, len(layers) * layers[0] * 2 + layers[0] * 2 + 1):
        show = True
        plt.subplot(len(layers) + 2, layers[0] * 2, r)
        if r == 1:
            plt.title('Layer 0', loc='left', fontsize=16)
        if layers[0] * 2 + 1 > r > 1:
            ax = plt.gca()
            ax.axis('off')
            show = False
        if r == layers[0] * 2 + 1:
            plt.title('Layer 1 \nBasic Block 0', loc='left', fontsize=16)
        if r == layers[0] * 2 + 3:
            plt.title('Basic Block 1', loc='left', fontsize=16)
        if r == layers[0] * 2 + 5:
            plt.title('Basic Block 2', loc='left', fontsize=16)
        if r == layers[0] * 4 + 1:
            plt.title('Layer 2 \nBasic Block 0', loc='left', fontsize=16)
        if model_1 and layers[0] * 4 + layers[0] * 2 - 2 < r < layers[0] * 4 + 2 * layers[0] + 1:
            ax = plt.gca()
            ax.axis('off')
            show = False
        elif r == layers[0] * 4 + 5:
            plt.title('Basic Block 2', loc='left', fontsize=16)
        if r == layers[0] * 4 + 3:
            plt.title('Basic Block 1', loc='left', fontsize=16)
        if r == layers[0] * 6 + 1:
            plt.title('Layer 3 \nBasic Block 0', loc='left', fontsize=16)
        if model_1 and layers[0] * 6 + layers[0] * 2 - 2 < r < layers[0] * 6 + 2 * layers[0] + 1:
            ax = plt.gca()
            ax.axis('off')
            show = False
        elif r == layers[0] * 6 + 5:
            plt.title('Basic Block 2', loc='left', fontsize=16)
        if r == layers[0] * 6 + 3:
            plt.title('Basic Block 1', loc='left', fontsize=16)

        if show:
            colors = ['C0', 'C1', 'C2', 'C4', 'C6']
            tmp = rat_groups[plt_counter]
            alpha_tmp = alphas[plt_counter]
            legend = []
            y, x = calc_mixture_plot(alpha_tmp, tmp)
            plt.plot(x, y, color='red')
            legend.append('mixture')

            for rational in range(len(tmp)):
                x = tmp[rational].show(display=False)['line']['x']
                y = tmp[rational].show(display=False)['line']['y']
                plt.plot(x, y, color=colors[rational])
                legend.append('\u03B1_{}: {:0.4f}, init.: {}, deg.: {}'.format(rational, alpha_tmp[rational], inits[rational], tmp[rational].degrees))

            plt.legend(legend, bbox_to_anchor=(0.5, -0.4), ncol=1, loc='center')
            plt_counter += 1

    plt.tight_layout()
    time_stamp = datetime.now()

    PATH = './Results/activation_functions/{}'.format(time_stamp) + '_{}'.format(LTH_args.dataset) + '_all.svg'
    plt.savefig(PATH)
    plt.show()



