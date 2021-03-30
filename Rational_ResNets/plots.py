import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from rational.torch import Rational
from Rational_ResNets.main import get_resnet_args

plt.style.use(["science", "grid"])
matplotlib.rcParams.update({
    "font.family": "serif",
    "text.usetex": False,
})


def final_plot(acc_x_vals, train_acc_y_vals, val_acc_y_vals, test_acc_y_vals, cm, epoch_time, test_acc):
    args = get_resnet_args()
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
    text = 'num epochs: {}, '.format(args.num_epochs) + \
           'batch size: {}, '.format(args.batch_size) + 'lr: {}, '.format(args.learning_rate) + '\n' + \
           'number of rationals per BasicBlock: {}, '.format(args.num_rationals) + '\n' + \
           'avg time per epoch: {:.0f}m {:.0f}s, '.format(epoch_time // 60, epoch_time % 60) + \
           'test accuracy: {:4f}, '.format(test_acc) + 'dataset: {}'.format(args.dataset)
    plt.text(15, 5, text, size=10, bbox=props)

    plt.savefig('{}.svg'.format(args.experiment_name))
    plt.show()


def activation_function_plots(model):
    for mod in model.modules():
        if isinstance(mod, Rational):
            mod.show()
