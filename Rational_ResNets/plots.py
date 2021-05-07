from datetime import datetime
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from rational.torch import Rational
import argparser

from Rational_ResNets.ResNet_Models.mix_experts_resnet_cifar10 import RationalResNet as mix
from Rational_ResNets.ResNet_Models.univ_rational_resnet_cifar10 import RationalResNet as univ

plt.style.use(["science", "grid"])
matplotlib.rcParams.update({
    "font.family": "serif",
    "text.usetex": False,
})

resnet_args = argparser.get_arguments()


def final_plot(cm, epoch_time, best_test_acc: float, acc_x_vals: list, train_acc_y_vals: list, val_acc_y_vals: list, test_acc_y_vals: list, num_rationals):
    """
    Plot the train-, validation- and test accuracy and a small legend.

    Parameters
    ----------
    cm: Tensor
        Tensor with confusion matrix.
    epoch_time:
                Average time per epoch
    best_test_acc: float
              Best test accuracy.
    acc_x_vals: list
                A list with the x values for the plot.
    train_acc_y_vals: list
                      A list with all training accuracies.
    val_acc_y_vals: list
                    A list with all validation accuracies.
    test_acc_y_vals: list
                     A list with all test accuracies.

    num_rationals: int
                   Number of Rational Activation Functions per BasicBlock

    Returns
    -------
    PATH:   The path to the saved plot.
    """
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.plot(acc_x_vals, train_acc_y_vals)
    plt.plot(acc_x_vals, val_acc_y_vals)
    plt.plot(acc_x_vals, test_acc_y_vals)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Train Accuracy', 'Validation Accuracy', 'Test Accuracy'])

    plt.subplot(132)
    cm_1 = sns.heatmap(cm, linewidths=1, cmap='plasma')
    props = dict(boxstyle='round', facecolor='grey', alpha=0.5)
    text = '{} training epochs, '.format(resnet_args.training_number_of_epochs) + \
           'batch size: {}, '.format(resnet_args.batch_size) + 'lr: {}, '.format(resnet_args.learning_rate) + '\n' + \
           '{} rationals per BasicBlock, '.format(num_rationals) + '\n' + \
           'avg time per epoch: {:.0f}m {:.0f}s, '.format(epoch_time // 60, epoch_time % 60) + \
           'test accuracy: {:4f}, '.format(best_test_acc) + 'dataset: {}'.format(resnet_args.dataset)
    plt.text(15, 5, text, size=10, bbox=props)

    time_stamp = datetime.now()
    PATH = './Plots/{}'.format(resnet_args.model) + '/' + '{}'.format(time_stamp) + '_' + '{}'.format(resnet_args.model) + '_' + '{}'.format(resnet_args.dataset) + '.svg'
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

    PATH = './Plots/activation_functions/{}'.format(time_stamp) + '_{}'.format(resnet_args.dataset) + '.svg'
    plt.savefig(PATH)
    plt.show()


def plot_overview_all(training_accs, val_accs, test_accs, x_vals, best_test_accs, avg_epoch, model_names: List[str]):
    """
    Make an overview graph, containing three sub-graphs to compare the three models.

    Parameters
    ----------
    training_accs:
                    A list with all training accuracies of all three models.
    val_accs:
                A list with all validation accuracies of all three models.
    test_accs:
                A list with all test accuracies of all three models.
    x_vals:
                Values for the x-axes of the graphs.
    best_test_accs:
                        A list with the best test accuracy for each model.
    avg_epoch:
                The average time per epoch.
    model_names:    List[str]
                    A list with the model names.

    Returns
    -------
    PATH:   str
            Path to the saved plot.
    """

    plt.figure(figsize=(20, 6))
    plt.subplots_adjust(bottom=0.3)

    plt.subplot(141)
    for i in range(len(training_accs)):
        plt.plot(x_vals, training_accs[i], label=model_names[i])
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy in Percent')

    plt.subplot(142)
    for i in range(len(val_accs)):
        plt.plot(x_vals, val_accs[i], label=model_names[i])
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy in Percent')
    plt.legend([model_names[0], model_names[1], model_names[2]], bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)

    plt.subplot(143)
    for i in range(len(test_accs)):
        plt.plot(x_vals, test_accs[i], label=model_names[i])
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy in Percent')

    props = dict(boxstyle='round', facecolor='grey', alpha=0.5)
    text = 'dataset: {}, '.format(resnet_args.dataset) + 'batch size: {}, '.format(resnet_args.batch_size) + '\n' + '{} training epochs, '.format(resnet_args.training_number_of_epochs) + '\n' + \
           'learning rate: {}, '.format(resnet_args.learning_rate) + '{} warm-up iterations, '.format(resnet_args.warmup_iterations) + '\n' + \
           'best test accuracy: ' + '\n' + \
           '- {}: {:4f}'.format(model_names[0], best_test_accs[0]) + '\n' + '- {}: {:4f}'.format(model_names[1], best_test_accs[1]) + '\n' + '- {}: {:4f}'.format(model_names[2], best_test_accs[2]) + '\n' + \
           'average time per epoch: ' + '\n' + \
           '- {}: {:.0f}m {:.0f}s, '.format(model_names[0], avg_epoch[0] // 60, avg_epoch[0] % 60) + '\n' + '- {}: {:.0f}m {:.0f}s, '.format(model_names[1], avg_epoch[1] // 60, avg_epoch[1] % 60) + '\n' \
           + '- {}: {:.0f}m {:.0f}s, '.format(model_names[2], avg_epoch[2] // 60, avg_epoch[2] % 60)

    plt.figtext(0.725, 0.5, text, bbox=props, size=9)

    time_stamp = datetime.now()
    PATH = './Plots/all' + '/' + '{}'.format(time_stamp) + '_' + '{}'.format(resnet_args.dataset) + '.svg'
    plt.savefig(PATH)
    plt.show()
    return PATH


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


def plot_activation_func_overview_mix(model, num_rat: int, inits: List[str]):
    """
    Plot a graph with all Rational Activation Functions of the mixture of experts models.

    Parameters
    ----------
    model
    num_rat:    int
                Number of experts per expert group
    inits:  List[str]
            The Rational Activation Functions' initializations.

    Returns
    -------
    PATH:   The path to the saved plot.
    """
    c = 0
    tmp = []
    rat_groups = []
    alphas = []
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
            alphas.append(param)

    layers = model.layers

    if isinstance(model, mix):
        resnet20_plot_mix(layers, rat_groups, alphas, inits)
    else:
        resnet18_plot_mix(layers, rat_groups, alphas, inits)

    plt.tight_layout()
    time_stamp = datetime.now()

    PATH = './Plots/activation_functions/{}'.format(time_stamp) + '_{}'.format(resnet_args.dataset) + '_all_mix.svg'
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
            if resnet_args.hist:
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
            if resnet_args.hist:
                bins = tmp[0].show(display=False)['hist']['bins']
                freq = tmp[0].show(display=False)['hist']['freq']
                ax = plt.gca()
                ax2 = ax.twinx()
                ax2.set_yticks([])
                ax2.bar(bins, freq, width=bins[1] - bins[0], color='grey', edgecolor='grey', alpha=0.3)
            plt_counter += 1


def plot_activation_func_overview_univ(model):
    """
    Plot a graph with all Rational Activation Functions of the mixture of experts models.

    Parameters
    ----------
    model

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

    PATH = './Plots/activation_functions/{}'.format(time_stamp) + '_{}'.format(resnet_args.dataset) + '_all_univ.svg'
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
            if resnet_args.hist:
                bins = tmp[0].show(display=False)['hist']['bins']
                freq = tmp[0].show(display=False)['hist']['freq']
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
            if resnet_args.hist:
                bins = tmp[0].show(display=False)['hist']['bins']
                freq = tmp[0].show(display=False)['hist']['freq']
                ax = plt.gca()
                ax2 = ax.twinx()
                ax2.set_yticks([])
                ax2.bar(bins, freq, width=bins[1] - bins[0], color='grey', edgecolor='grey', alpha=0.3)
            plt_counter += 1

