import inspect
import os
import sys
from datetime import datetime
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from rational.torch import Rational
from LTH_for_Rational_ResNets.LTH_Models.mix_experts_resnet_cifar10 import RationalResNet as mix
from LTH_for_Rational_ResNets.LTH_Models.univ_rational_resnet_cifar10 import RationalResNet as univ

from LTH_for_Rational_ResNets import argparser

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

plt.style.use(["science", "grid"])
matplotlib.rcParams.update({
    "font.family": "serif",
    "text.usetex": False,
})

LTH_args = argparser.get_arguments()


def final_plot_LTH(test_accuracies: list, sparsity: list, num_pruning_epochs: int, model_name: str):
    """
    Plot test accuracy for each pruning epoch and a small legend.

    Parameters
    ----------
    test_accuracies: list
                     A list containing the test accuracies after every pruning epoch.
    sparsity: list
              A list containing the different sparsity for each pruning epoch.
    num_pruning_epochs: int
                        Number of pruning epochs.
    model_name: str

    Returns
    -------
    PATH:   The path to the saved plot.
    """

    plt.figure(figsize=(10, 6))
    plt.subplot(121)
    plt.plot(sparsity, test_accuracies)
    plt.xlabel('Percent of Pruned Weights')
    plt.ylabel('Test Accuracy in Percent')
    plt.subplots_adjust(bottom=0.3)
    plt.legend(['Test Accuracy'], bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=1)

    props = dict(boxstyle='round', facecolor='grey', alpha=0.5)
    text = 'model: {}, '.format(model_name) + 'dataset: {}, '.format(LTH_args.dataset) + '\n' + 'batch size: {}, '.format(LTH_args.batch_size) + '{} iterative pruning epochs, '.format(num_pruning_epochs) \
           + '\n' + '{} training epochs per pruning epoch, '.format(LTH_args.training_number_of_epochs) + '\n' + \
           'learning rate: {}, '.format(LTH_args.learning_rate) + '{}% pruning per epoch, '.format(LTH_args.pruning_percentage) + '\n' + '{} warm-up iterations, '.format(LTH_args.warmup_iterations) + \
           'shortcuts pruned: {}'.format(LTH_args.prune_shortcuts)

    plt.figtext(0.525, 0.5, text, bbox=props, size=9)

    time_stamp = datetime.now()
    PATH = './Plots/{}'.format(LTH_args.model) + '/' + '{}'.format(time_stamp) + '_' + '{}'.format(LTH_args.model) + '_' + '{}'.format(LTH_args.dataset) + '.svg'
    plt.savefig(PATH)
    plt.show()
    return PATH


def plot_all(test_accs, sparsities, num_epoch_list: list, model_names: List[str]):
    """
    Plot results of all three experiments in one graph for further comparison.

    Parameters
    ----------
    test_accs:
                A list of lists with all test accuracies for all three models
    sparsities:
                A list of lists with all sparsities for all three models
    num_epoch_list: list
                    A list with the pruning epoch each experiment terminated at.
    model_names:    List[str]

    Returns
    -------
    PATH:   The path to the saved plot.
    """
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
    plt.legend([model_names[0], model_names[1], model_names[2]], bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=1)

    props = dict(boxstyle='round', facecolor='grey', alpha=0.5)

    if LTH_args.stop_criteria is 'test_acc':
        stop_criterion = 'terminate at 90% test accuracy'
    else:
        stop_criterion = 'terminate after {} epochs'.format(LTH_args.iterative_pruning_epochs)
    text = 'dataset: {}, '.format(LTH_args.dataset) + 'batch size: {}, '.format(LTH_args.batch_size) + '\n' + '{} training epochs per pruning epoch, '.format(LTH_args.training_number_of_epochs) + '\n' + \
           'learning rate: {}, '.format(LTH_args.learning_rate) + '{}% pruning per epoch, '.format(LTH_args.pruning_percentage) + '\n' + '{} warm-up iterations, '.format(LTH_args.warmup_iterations) + '\n' + \
           'shortcuts pruned: {}, '.format(LTH_args.prune_shortcuts) + '\n' + 'number of iterative pruning epochs: ' + '\n' + \
           '- {}: {}'.format(model_names[0], num_epoch_list[0]) + '\n' + '- {}: {}'.format(model_names[1], num_epoch_list[1]) + '\n' + '- {}: {}'.format(model_names[2], num_epoch_list[2]) + '\n' + stop_criterion

    plt.figtext(0.525, 0.5, text, bbox=props, size=9)
    time_stamp = datetime.now()

    PATH = './Plots/LTH_all_models/{}'.format(time_stamp) + '_{}'.format(LTH_args.dataset) + '.svg'
    plt.savefig(PATH)
    plt.show()
    return PATH


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

    PATH = './Plots/activation_functions/{}'.format(time_stamp) + '_{}'.format(LTH_args.dataset) + '.svg'
    plt.savefig(PATH)
    plt.show()


def calc_mixture_plot(alphas, rationals):
    """
    Calculate the mixture of experts.

    Parameters
    ----------
    alphas:     torch.nn.parameter.Parameter
                The weights of the different experts for the weighted sum of experts.
    rationals:  List[Rational]
                The expert group.

    Returns
    -------
    torch.Tensor
    torch.Tensor
    """
    shape = rationals[0].show(display=False)['line']['y'].shape[0]
    all_x = torch.zeros(len(alphas), shape)

    for i in range(len(alphas)):
        alpha_x = rationals[i].show(display=False)['line']['y'] * alphas[i].detach().cpu().numpy()
        all_x[i] = torch.tensor(alpha_x)
    return all_x.sum(dim=0), rationals[0].show(display=False)['line']['x']


def plot_activation_func_overview_mix(model, num_rat: int, inits: List[str], saved_models_path: str):
    """
    Plot a graph with all Rational Activation Functions of the mixture of experts models.

    Parameters
    ----------
    model
    num_rat:    int
                Number of experts per expert group
    inits:  List[str]
            The Rational Activation Functions' initializations.
    saved_models_path:  str
                        Path to the directory where the model is saved.

    Returns
    -------
    PATH:   The path to the saved plot.
    """
    c = 0
    tmp = []
    rat_groups = []
    alphas = []
    softmax = torch.nn.Softmax(dim=0)
    for m in model.modules():
        if isinstance(m, Rational):
            tmp.append(m)
            c += 1
        if c == num_rat:
            rat_groups.append(tmp)
            tmp = []
            c = 0

    for m, param in model.named_parameters():
        if 'alpha' in m:
            alphas.append(softmax(param))

    layers = model.layers

    if isinstance(model, mix):
        resnet20_plot_mix(layers, rat_groups, alphas, inits)
    else:
        resnet18_plot_mix(layers, rat_groups, alphas, inits)

    plt.tight_layout()
    time_stamp = datetime.now()

    PATH = saved_models_path + '/{}'.format(time_stamp) + '_{}'.format(LTH_args.dataset) + '_all_mix.svg'
    plt.savefig(PATH)
    plt.show()
    return PATH


def resnet20_plot_mix(layers: List[int], rat_groups, alphas, inits: List[str]):
    """
    Method to plot activation function overview for CIFAR10 ResNet.

    Parameters
    ----------
    layers:     List[int]
    rat_groups:
                All expert groups grouped together.
    alphas:
            All alphas (weights for weighted sum) grouped together.
    inits:     List[str]
               The Rational Activation Functions' initializations.
    """
    if len(layers) == 1:
        x = 1.75
    elif len(layers) == 2:
        x = 2.5
    else:
        x = 3
    plt.figure(figsize=(layers[0] * 8, x * 8))

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
        elif r == layers[0] * 4 + 5:
            plt.title('Basic Block 2', loc='left', fontsize=16)
        if r == layers[0] * 4 + 3:
            plt.title('Basic Block 1', loc='left', fontsize=16)
        if r == layers[0] * 6 + 1:
            plt.title('Layer 3 \nBasic Block 0', loc='left', fontsize=16)
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
            plt.plot(x, y, color='C3', linewidth=1.75)
            legend.append('mixture')

            for rational in range(len(tmp)):
                x = tmp[rational].show(display=False)['line']['x']
                y = tmp[rational].show(display=False)['line']['y']
                plt.plot(x, y, color=colors[rational])
                legend.append('\u03B1_{}: {:0.4f}, init.: {}, deg.: {}'.format(rational, alpha_tmp[rational], inits[rational], tmp[rational].degrees))

            plt.legend(legend, bbox_to_anchor=(0.5, -0.4), ncol=1, loc='center')
            if LTH_args.hist:
                bins = tmp[0].show(display=False)['hist']['bins']
                freq = tmp[0].show(display=False)['hist']['freq']
                ax = plt.gca()
                ax2 = ax.twinx()
                ax2.set_yticks([])
                ax2.bar(bins, freq, width=bins[1] - bins[0], color='grey', edgecolor='grey', alpha=0.3)
            plt_counter += 1


def resnet18_plot_mix(layers, rat_groups, alphas, inits):
    """
    Method to plot activation function overview for ImageNet ResNet.

    Parameters
    ----------
    layers:     List[int]
    rat_groups:
                All expert groups grouped together.
    alphas:
            All alphas (weights for weighted sum) grouped together.
    inits:     List[str]
               The Rational Activation Functions' initializations.
    """
    if len(layers) == 1:
        x = 1.75
    elif len(layers) == 2:
        x = 2.5
    else:
        x = 3
    plt.figure(figsize=(layers[0] * 8, x * 8))

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
        if r == layers[0] * 3 + 1:
            plt.title('Basic Block 1', loc='left', fontsize=16)
        if r == layers[0] * 4 + 1:
            plt.title('Layer 2 \nBasic Block 0', loc='left', fontsize=16)
        if r == layers[0] * 5 + 1:
            plt.title('Basic Block 1', loc='left', fontsize=16)
        if r == layers[0] * 6 + 1:
            plt.title('Layer 3 \nBasic Block 0', loc='left', fontsize=16)
        if r == layers[0] * 7 + 1:
            plt.title('Basic Block 1', loc='left', fontsize=16)
        if r == layers[0] * 8 + 1:
            plt.title('Layer 4 \nBasic Block 0', loc='left', fontsize=16)
        if r == layers[0] * 9 + 1:
            plt.title('Basic Block 1', loc='left', fontsize=16)

        if show:
            colors = ['C0', 'C1', 'C2', 'C4', 'C6']
            tmp = rat_groups[plt_counter]
            alpha_tmp = alphas[plt_counter]
            legend = []
            y, x = calc_mixture_plot(alpha_tmp, tmp)
            plt.plot(x, y, color='C3', linewidth=1.75)
            legend.append('mixture')

            for rational in range(len(tmp)):
                x = tmp[rational].show(display=False)['line']['x']
                y = tmp[rational].show(display=False)['line']['y']
                plt.plot(x, y, color=colors[rational])
                legend.append('\u03B1_{}: {:0.4f}, init.: {}, deg.: {}'.format(rational, alpha_tmp[rational], inits[rational], tmp[rational].degrees))

            plt.legend(legend, bbox_to_anchor=(0.5, -0.4), ncol=1, loc='center')
            if LTH_args.hist:
                bins = tmp[0].show(display=False)['hist']['bins']
                freq = tmp[0].show(display=False)['hist']['freq']
                ax = plt.gca()
                ax2 = ax.twinx()
                ax2.set_yticks([])
                ax2.bar(bins, freq, width=bins[1] - bins[0], color='grey', edgecolor='grey', alpha=0.3)
            plt_counter += 1


def plot_activation_func_overview_univ(model, saved_models_path: str):
    """
    Plot a graph with all Rational Activation Functions of the mixture of experts models.

    Parameters
    ----------
    model
    saved_models_path:  str
                        Path to the saved model.

    Returns
    -------
    PATH:   The path to the saved plot.
    """
    rat = []
    for m in model.modules():
        if isinstance(m, Rational):
            rat.append(m)

    layers = model.layers

    if isinstance(model, univ):
        resnet20_plot_univ(layers, rat)

    plt.tight_layout()
    time_stamp = datetime.now()

    PATH = saved_models_path + '/{}'.format(time_stamp) + '_{}'.format(LTH_args.dataset) + '_all_univ.svg'
    plt.savefig(PATH)
    plt.show()
    return PATH


def resnet20_plot_univ(layers: List[int], rat):
    """
    Method to plot activation function overview for CIFAR10 ResNet.

    Parameters
    ----------
    layers:     List[int]
    rat:
                All Rationals.
    """
    if len(layers) == 1:
        x = 1.75
    elif len(layers) == 2:
        x = 2.5
    else:
        x = 3
    plt.figure(figsize=(layers[0] * 8, x * 7))

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
        elif r == layers[0] * 4 + 5:
            plt.title('Basic Block 2', loc='left', fontsize=16)
        if r == layers[0] * 4 + 3:
            plt.title('Basic Block 1', loc='left', fontsize=16)
        if r == layers[0] * 6 + 1:
            plt.title('Layer 3 \nBasic Block 0', loc='left', fontsize=16)
        elif r == layers[0] * 6 + 5:
            plt.title('Basic Block 2', loc='left', fontsize=16)
        if r == layers[0] * 6 + 3:
            plt.title('Basic Block 1', loc='left', fontsize=16)

        if show:
            tmp = rat[plt_counter]

            x = tmp.show(display=False)['line']['x']
            y = tmp.show(display=False)['line']['y']
            plt.plot(x, y)

            plt.legend(['init.: leaky relu, deg.: {}'.format(tmp.degrees)], bbox_to_anchor=(0.5, -0.2), ncol=1, loc='center')
            if LTH_args.hist:
                bins = tmp.show(display=False)['hist']['bins']
                freq = tmp.show(display=False)['hist']['freq']
                ax = plt.gca()
                ax2 = ax.twinx()
                ax2.set_yticks([])
                ax2.bar(bins, freq, width=bins[1] - bins[0], color='grey', edgecolor='grey', alpha=0.3)
            plt_counter += 1


def resnet18_plot_univ(layers: List[int], rat):
    """
    Method to plot activation function overview for ImageNet ResNet.

    Parameters
    ----------
    layers:     List[int]
    rat:
                All Rationals.
    """
    if len(layers) == 1:
        x = 1.75
    elif len(layers) == 2:
        x = 2.5
    else:
        x = 3
    plt.figure(figsize=(layers[0] * 8, x * 8))

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
        if r == layers[0] * 3 + 1:
            plt.title('Basic Block 1', loc='left', fontsize=16)
        if r == layers[0] * 4 + 1:
            plt.title('Layer 2 \nBasic Block 0', loc='left', fontsize=16)
        if r == layers[0] * 5 + 1:
            plt.title('Basic Block 1', loc='left', fontsize=16)
        if r == layers[0] * 6 + 1:
            plt.title('Layer 3 \nBasic Block 0', loc='left', fontsize=16)
        if r == layers[0] * 7 + 1:
            plt.title('Basic Block 1', loc='left', fontsize=16)
        if r == layers[0] * 8 + 1:
            plt.title('Layer 4 \nBasic Block 0', loc='left', fontsize=16)
        if r == layers[0] * 9 + 1:
            plt.title('Basic Block 1', loc='left', fontsize=16)

        if show:
            tmp = rat[plt_counter]

            x = tmp.show(display=False)['line']['x']
            y = tmp.show(display=False)['line']['y']
            plt.plot(x, y)

            plt.legend(['init.: leaky relu, deg.: {}'.format(tmp.degrees)], bbox_to_anchor=(0.5, -0.2), ncol=1, loc='center')
            if LTH_args.hist:
                bins = tmp.show(display=False)['hist']['bins']
                freq = tmp.show(display=False)['hist']['freq']
                ax = plt.gca()
                ax2 = ax.twinx()
                ax2.set_yticks([])
                ax2.bar(bins, freq, width=bins[1] - bins[0], color='grey', edgecolor='grey', alpha=0.3)
            plt_counter += 1
