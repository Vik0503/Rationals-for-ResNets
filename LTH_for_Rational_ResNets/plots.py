import matplotlib.pyplot as plt
import matplotlib
import main


def make_LTH_plot(num_pruning_epochs: int, test_accuracies: list, sparsity: list):
    """
    Plot test accuracy for each pruning epoch.
    Parameters
    ----------
    num_pruning_epochs: int
    test_accuracies: list
                     A list containing the test accuracies after every pruning epoch.
    sparsity: list
              A list containing the different sparsity for each pruning epoch.
    """
    plt.style.use(["science", "grid"])
    matplotlib.rcParams.update({
        "font.family": "serif",
        "text.usetex": False,
    })

    LTH_args = main.get_LTH_args()
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(sparsity, test_accuracies)
    plt.xlabel('Percent of Pruned Weights')
    plt.ylabel('Test Accuracy in Percent')
    plt.legend(['Test Accuracy'])
    props = dict(boxstyle='round', facecolor='grey', alpha=0.5)
    text = 'model: {}, '.format(LTH_args.model) + 'dataset: {}, '.format(LTH_args.dataset) + 'batch size: {}, '.format(LTH_args.batch_size) + '\n' + 'number of iterative pruning epochs: {}, '.format(num_pruning_epochs) \
           + '\n' + 'number of training epochs per pruning epoch: {}, '.format(LTH_args.training_number_of_epochs) + '\n' + \
           'learning rate: {}, '.format(LTH_args.learning_rate) + 'pruning percentage per epoch: {}, '.format(LTH_args.pruning_percentage) + '\n' + 'number of warm-up iterations: {}'.format(LTH_args.warmup_iterations)

    plt.figtext(0.525, 0.5, text, bbox=props, size=9)
    plt.savefig('{}.svg'.format(LTH_args.exp_name))
    plt.show()
