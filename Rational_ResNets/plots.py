from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from rational.torch import Rational
import argparser

plt.style.use(["science", "grid"])
matplotlib.rcParams.update({
    "font.family": "serif",
    "text.usetex": False,
})

ResNet_args = argparser.get_args()


def accuracy_plot(acc_x_vals: list, train_acc_y_vals: list, val_acc_y_vals: list, test_acc_y_vals: list):  # TODO: reduce size
    """
    Plot the train-, validation- and test accuracy.

    Parameters
    ----------
    acc_x_vals: list
                A list with the x values for the plot.
    train_acc_y_vals: list
                      A list with all training accuracies.
    val_acc_y_vals: list
                    A list with all validation accuracies.
    test_acc_y_vals: list
                     A list with all test accuracies.
    """
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.plot(acc_x_vals, train_acc_y_vals)
    plt.plot(acc_x_vals, val_acc_y_vals)
    plt.plot(acc_x_vals, test_acc_y_vals)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Train Accuracy', 'Validation Accuracy', 'Test Accuracy'])


def final_plot(cm, epoch_time, test_acc: float, num_epochs: int, learning_rate: float, num_rationals: int, dataset: str, model: str, batch_size: int):  # TODO: Update Legend
    """
    Plot the confusion matrix and add the description box.

    Parameters
    ----------
    cm: Tensor
        Tensor with confusion matrix.
    epoch_time:
                Average time per epoch
    test_acc: float
              Best test accuracy.
    num_epochs: int
                Number of training epochs.
    learning_rate: float
    num_rationals: int
                   Number of Rational Activation Functions per BasicBlock
    dataset: str
    model: str
    batch_size: int
    """
    plt.subplot(132)
    cm_1 = sns.heatmap(cm, linewidths=1, cmap='plasma')
    props = dict(boxstyle='round', facecolor='grey', alpha=0.5)
    text = 'num epochs: {}, '.format(num_epochs) + \
           'batch size: {}, '.format(batch_size) + 'lr: {}, '.format(learning_rate) + '\n' + \
           'number of rationals per BasicBlock: {}, '.format(num_rationals) + '\n' + \
           'avg time per epoch: {:.0f}m {:.0f}s, '.format(epoch_time // 60, epoch_time % 60) + \
           'test accuracy: {:4f}, '.format(test_acc) + 'dataset: {}'.format(dataset)
    plt.text(15, 5, text, size=10, bbox=props)

    time_stamp = datetime.now()
    PATH = './Results/{}'.format(model) + '/' + '{}'.format(time_stamp) + '_' + '{}'.format(model) + '_' + '{}'.format(dataset) + '.svg'
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

    PATH = './Results/activation_functions/{}'.format(time_stamp) + '_{}'.format(ResNet_args.dataset) + '.svg'
    plt.savefig(PATH)
    plt.show()


def plot_overview_all(training_accs, val_accs, test_accs, x_vals, best_test_accs, avg_epoch):  # TODO: double check order
    plot_labels = ['ReLU ResNet20', 'univ. rational ResNet20', 'mix. exp. ResNet20']

    plt.figure(figsize=(20, 6))
    plt.subplots_adjust(bottom=0.3)

    plt.subplot(141)
    for i in range(len(training_accs)):
        plt.plot(x_vals, training_accs[i], label=plot_labels[i])
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy in Percent')

    plt.subplot(142)
    for i in range(len(val_accs)):
        plt.plot(x_vals, val_accs[i], label=plot_labels[i])
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy in Percent')
    plt.legend(['ReLU ResNet20', 'univ. rational ResNet20', 'mix. exp. ResNet20'], bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)

    plt.subplot(143)
    for i in range(len(test_accs)):
        plt.plot(x_vals, test_accs[i], label=plot_labels[i])
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy in Percent')

    props = dict(boxstyle='round', facecolor='grey', alpha=0.5)
    text = 'dataset: {}, '.format(ResNet_args.dataset) + 'batch size: {}, '.format(ResNet_args.batch_size) + '\n' + '{} training epochs, '.format(ResNet_args.training_number_of_epochs) + '\n' + \
           'learning rate: {}, '.format(ResNet_args.learning_rate) + '{} warm-up iterations, '.format(ResNet_args.warmup_iterations) + '\n' + \
           'best test accuracy: ' + '\n' + \
           '- ReLU ResNet: {:4f}'.format(best_test_accs[0]) + '\n' + '- univ. rational ResNet20: {:4f}'.format(best_test_accs[1]) + '\n' + '- mix. exp. ResNet20: {:4f}'.format(best_test_accs[2]) + '\n' + \
           'average time per epoch: ' + '\n' + \
           '- ReLU ResNet: {:.0f}m {:.0f}s, '.format(avg_epoch[0] // 60, avg_epoch[0] % 60) + '\n' + '- univ. rational ResNet20: {:.0f}m {:.0f}s, '.format(avg_epoch[1] // 60, avg_epoch[1] % 60) + '\n' \
           + '- mix. exp. ResNet20: {:.0f}m {:.0f}s, '.format(avg_epoch[2] // 60, avg_epoch[2] % 60)

    plt.figtext(0.725, 0.5, text, bbox=props, size=9)

    time_stamp = datetime.now()
    PATH = './Results/all' + '/' + '{}'.format(time_stamp) + '_' + '{}'.format(ResNet_args.dataset) + '.svg'
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
        if c == num_rat:
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

    PATH = './Results/activation_functions/{}'.format(time_stamp) + '_{}'.format(ResNet_args.dataset) + '.svg'
    plt.savefig(PATH)
    plt.show()