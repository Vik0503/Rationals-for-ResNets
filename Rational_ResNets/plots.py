from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
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
    for mod in model.modules():
        if isinstance(mod, Rational):
            mod.show()


def plot_overview_all(training_accs, val_accs, test_accs, x_vals, best_test_accs, avg_epoch):
    plot_labels = ['ResNet20 univ. Rat.', 'ResNet20 Original', 'ResNet20 mixture of 5 univ. Rat.']
    plot_colors = ['C1', 'C0', 'C2']

    plt.figure(figsize=(20, 6))
    plt.subplots_adjust(bottom=0.3)

    plt.subplot(141)
    for i in range(len(training_accs)):
        plt.plot(x_vals, training_accs[i], label=plot_labels[i], color=plot_colors[i])
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy in Percent')
    # plt.legend(['ResNet20 univ. Rat.', 'ResNet20 Original', 'ResNet20 mixture of 5 univ. Rat.'], bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=1)

    plt.subplot(142)
    for i in range(len(val_accs)):
        plt.plot(x_vals, val_accs[i], label=plot_labels[i], color=plot_colors[i])
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy in Percent')
    plt.legend(['ResNet20 univ. Rat.', 'ResNet20 Original', 'ResNet20 mixture of 5 univ. Rat.'], bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)

    plt.subplot(143)
    for i in range(len(test_accs)):
        plt.plot(x_vals, test_accs[i], label=plot_labels[i], color=plot_colors[i])
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy in Percent')
    # plt.legend(['ResNet20 univ. Rat.', 'ResNet20 Original', 'ResNet20 mixture of 5 univ. Rat.'], bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=1)

    props = dict(boxstyle='round', facecolor='grey', alpha=0.5)
    text = 'dataset: {}, '.format(ResNet_args.dataset) + 'batch size: {}, '.format(ResNet_args.batch_size) + '\n' + '{} training epochs, '.format(ResNet_args.training_number_of_epochs) + '\n' + \
           'learning rate: {}, '.format(ResNet_args.learning_rate) + '{} warm-up iterations, '.format(ResNet_args.warmup_iterations) + '\n' + \
           'best test accuracy: ' + '\n' + \
           '- ResNet20 univ. Rat: {:4f}'.format(best_test_accs[0]) + '\n' + '- ResNet20 Original: {:4f}'.format(best_test_accs[1]) + '\n' + '- ResNet20 mixture of 5 univ. Rat: {:4f}'.format(best_test_accs[2]) + '\n' + \
           'average time per epoch: ' + '\n' + \
           '- ResNet20 univ. Rat: {:.0f}m {:.0f}s, '.format(avg_epoch[0] // 60, avg_epoch[0] % 60) + '\n' + '- ResNet20 Original: {:.0f}m {:.0f}s, '.format(avg_epoch[1] // 60, avg_epoch[1] % 60) + '\n' \
           + '- ResNet20 mixture of 5 univ. Rat: {:.0f}m {:.0f}s, '.format(avg_epoch[2] // 60, avg_epoch[2] % 60)

    plt.figtext(0.725, 0.5, text, bbox=props, size=9)

    time_stamp = datetime.now()
    PATH = './Results/all' + '/' + '{}'.format(time_stamp) + '_' + '{}'.format(ResNet_args.dataset) + '.svg'
    plt.savefig(PATH)
    plt.show()
