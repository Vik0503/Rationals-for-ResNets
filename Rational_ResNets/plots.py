from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from rational.torch import Rational

plt.style.use(["science", "grid"])
matplotlib.rcParams.update({
    "font.family": "serif",
    "text.usetex": False,
})


def accuracy_plot(acc_x_vals: list, train_acc_y_vals: list, val_acc_y_vals: list, test_acc_y_vals: list):
    """
    Plot the train-, validation- and test accuracy.

    Parameters
    ----------
    acc_x_vals: list
                A list with the x values fr the plot.
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


def final_plot(cm, epoch_time, test_acc: float, num_epochs: int, learning_rate: float, num_rationals: int, dataset: str, model: str, batch_size: int):
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
