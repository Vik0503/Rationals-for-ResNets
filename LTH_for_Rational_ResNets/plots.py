from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib

plt.style.use(["science", "grid"])
matplotlib.rcParams.update({
    "font.family": "serif",
    "text.usetex": False,
})


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


def final_plot_LTH(model, dataset, batch_size, num_pruning_epochs, training_number_of_epochs, learning_rate, pruning_percentage, warmup_iterations, prune_shortcuts):
    props = dict(boxstyle='round', facecolor='grey', alpha=0.5)
    text = 'model: {}, '.format(model) + 'dataset: {}, '.format(dataset) + '\n' + 'batch size: {}, '.format(batch_size) + '{} iterative pruning epochs, '.format(num_pruning_epochs) \
           + '\n' + '{} training epochs per pruning epoch, '.format(training_number_of_epochs) + '\n' + \
           'learning rate: {}, '.format(learning_rate) + '{}% pruning per epoch, '.format(pruning_percentage) + '\n' + '{} warm-up iterations, '.format(warmup_iterations) + 'shortcuts pruned: {}'.format(prune_shortcuts)

    plt.figtext(0.525, 0.5, text, bbox=props, size=9)

    time_stamp = datetime.now()
    PATH = './Results/LTH_{}'.format(model) + '/' + '{}'.format(time_stamp) + '_' + '{}'.format(model) + '_' + '{}'.format(dataset) + '.svg'
    plt.savefig(PATH)
    plt.show()
