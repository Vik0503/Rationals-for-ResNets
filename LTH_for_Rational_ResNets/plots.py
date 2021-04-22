import inspect
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
print(parent_dir)
print(current_dir)
# sys.path.insert(0, parent_dir)
print(sys.path)
import argparser



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


def plot_all(test_accs, sparsities, num_epoch_list):  # TODO: Reset PATH

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
        stop_criterion = 'terminate at {} epochs'.format(LTH_args.iterative_pruning_epochs)
    text = 'dataset: {}, '.format(LTH_args.dataset) + 'batch size: {}, '.format(LTH_args.batch_size) + '\n' + '{} training epochs per pruning epoch, '.format(LTH_args.training_number_of_epochs) + '\n' + \
           'learning rate: {}, '.format(0.03) + '{}% pruning per epoch, '.format(LTH_args.pruning_percentage) + '\n' + '{} warm-up iterations, '.format(LTH_args.warmup_iterations) + '\n' + \
           'shortcuts pruned: {}, '.format(LTH_args.prune_shortcuts) +  '\n' + 'number of iterative pruning epochs: ' + '\n' + \
           '- ReLU ResNet20: {}'.format(num_epoch_list[0]) + '\n' + '- univ. rational ResNet20: {}'.format(num_epoch_list[1]) + '\n' + '- mix. exp. ResNet20: {}'.format(num_epoch_list[2]) + '\n' + stop_criterion

    plt.figtext(0.525, 0.5, text, bbox=props, size=9)
    time_stamp = datetime.now()

    # PATH = '../Results/LTH_all_models/'
    PATH = 'shortcuts_14_4_both_scale_2.svg'
    plt.savefig(PATH)
    plt.show()

