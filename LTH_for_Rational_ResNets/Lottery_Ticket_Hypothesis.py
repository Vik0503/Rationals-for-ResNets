import argparse as arg
from copy import deepcopy

import torch
from torch import nn, optim
from torch.optim import lr_scheduler

import matplotlib.pyplot as plt
import matplotlib

from LTH_for_Rational_ResNets.Datasets import CIFAR10 as cifar10
from LTH_for_Rational_ResNets.Datasets import SVHN
from LTH_for_Rational_ResNets import Train_Val_Test as tvt
from LTH_for_Rational_ResNets.LTH_Models import rational_resnet20_cifar10 as rrn20
from LTH_for_Rational_ResNets.LTH_Models import rational_resnet18_imagenet as rrn18
from LTH_for_Rational_ResNets.LTH_Models import resnet20_cifar10 as rn20
from LTH_for_Rational_ResNets.LTH_Models import resnet18_imagenet as rn18
from LTH_for_Rational_ResNets.LTH_Models import multi_variant_rational_resnet20_cifar10 as mvrrn20
from LTH_for_Rational_ResNets.Lottery_Ticket_Pruning import prune
from LTH_for_Rational_ResNets.Mask import make_initial_mask, mask_sparsity, Mask

LTH_arg_parser = arg.ArgumentParser()
LTH_arg_parser.add_argument('-exp_name', 'experiment_name', type=str, required=True)
LTH_arg_parser.add_argument('-bs', '--batch_size', default=128, type=int)
LTH_arg_parser.add_argument('-lr', '--learning_rate', default=0.03, type=float)
LTH_arg_parser.add_argument('-m', '--model', default='rational_resnet20_cifar10', type=str, choices=['rational_resnet20_cifar10', 'resnet20_cifar10', 'rational_resnet18_imagenet',
                                                                                                     'resnet18_imagenet'])  # , 'multi_variant_rational_resnet20_cifar10'
LTH_arg_parser.add_argument('-ds', '--dataset', default='cifar10', type=str, choices=['cifar10', 'SVHN'])
LTH_arg_parser.add_argument('-wi', '--warmup_iterations', default=0, type=int)
LTH_arg_parser.add_argument('-tnep', '--training_number_of_epochs', default=2, type=int)
LTH_arg_parser.add_argument('-pp', '--pruning_percentage', default=20.0, type=float)
LTH_arg_parser.add_argument('-ipe', '--iterative_pruning_epochs', default=15, type=int)
LTH_arg_parser.add_argument('-stop', '--stop_criteria', default='val_acc', type=str, choices=['test_acc', 'num_prune_epochs', 'one_shot'])
LTH_arg_parser.add_argument('-test_acc', '--test_accuracy_threshold', default=0.89, type=float)
LTH_args = LTH_arg_parser.parse_args(['--model', 'resnet20_cifar10', '--dataset', 'SVHN', '--warmup_iterations', '20000', '--iterative_pruning_epochs', '2', '--training_number_of_epochs', '25',
                                      '--stop_criteria', 'test_acc'])

global trainset
global valset
global testsetq
global trainloader
global valloader
global testloader
global classes
global num_classes
global model
global model_type
global checkpoint

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

plt.style.use(["science", "grid"])
matplotlib.rcParams.update({
    "font.family": "serif",
    "text.usetex": False,
})

if LTH_args.dataset is 'cifar10':
    trainset = cifar10.get_trainset()
    valset = cifar10.get_validationset()
    testset = cifar10.get_testset()
    trainloader = cifar10.get_trainloader(bs=LTH_args.batch_size)
    valloader = cifar10.get_valloader(bs=LTH_args.batch_size)
    testloader = cifar10.get_testloader(bs=LTH_args.batch_size)
    classes = cifar10.get_classes()
    num_classes = cifar10.get_num_classes()

elif LTH_args.dataset is 'SVHN':
    trainset = SVHN.get_trainset()
    valset = SVHN.get_validationset()
    testset = SVHN.get_testset()
    trainloader = SVHN.get_trainloader(bs=LTH_args.batch_size)
    valloader = SVHN.get_valloader(bs=LTH_args.batch_size)
    testloader = SVHN.get_testloader(bs=LTH_args.batch_size)
    classes = SVHN.get_classes()
    num_classes = SVHN.get_num_classes()

if LTH_args.model is 'rational_resnet20_cifar10':
    model = rrn20.rational_resnet20()
    model_type = rrn20
elif LTH_args.model is 'resnet20_cifar10':
    model = rn20.resnet20()
    model_type = rn20
elif LTH_args.model is 'rational_resnet18_imagenet':
    model = rrn18.rational_resnet18()
    model_type = rrn18
elif LTH_args.model is 'resnet18_imagenet':
    model = rn18.resnet18()
    model_type = rn18
elif LTH_args.model is 'multi_variant_rational_resnet20_cifar10':
    model = mvrrn20.multi_variant_rational_resnet20()
    model_type = mvrrn20

mask = make_initial_mask(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LTH_args.learning_rate, momentum=0.9, weight_decay=0.0001)

model = model.to(device)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
initial_state = deepcopy(model.state_dict())


if LTH_args.warmup_iterations is not 0:
    exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda it: min(1.0, it / LTH_args.warmup_iterations))
else:
    exp_lr_scheduler = lr_scheduler.LambdaLR(optimizer, lambda it: 1)


def plot_results(num_pruning_epochs: int, test_accuracies: list, sparsity: list):
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


def checkpoint_save(epoch: int, save_model, model_mask: Mask, test_accuracy: float, training_epochs: int, model_sparsity: float = 0):
    """
    Save a checkpoint.
    Parameters
    ----------
    epoch: int
    save_model:
                Model to be saved.
    model_mask: Mask
    test_accuracy: float
    training_epochs: int
                     The number of training epochs per pruning epoch.
    model_sparsity: float
                    The sparsity of the model.
    """
    PATH = 'Saved_Models_wo_rationals/{}ep{}s{:.5f}test{:.5f}.pth'.format(LTH_args.exp_name, epoch, model_sparsity, test_accuracy)
    torch.save({
        'epoch': epoch,
        'model_state_dict': save_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'mask': model_mask,
        'sparsity': model_sparsity,
        'test_accuracy': test_accuracy,
        'training_epochs': training_epochs,
    }, PATH)


def one_shot_pruning(prune_model, prune_mask: Mask):
    """
    Prune trained model once and reinitialize and test it.
    Parameters
    ----------
    prune_model:
                 Model that is going to be iteratively pruned.
    prune_mask: Mask
                The mask used to prune the model.
    """
    prune_model.mask = prune_mask

    prune_model, best_val_accuracy, num_iterations = tvt.train(prune_model, criterion, optimizer, exp_lr_scheduler, LTH_args.training_number_of_epochs, trainset, valset, trainloader, valloader)

    test_accuracy = tvt.test(prune_model, testset, testloader)
    print('Test Accuracy with 100 Percent of weights: ', test_accuracy)

    pruned_model, updated_mask = prune(LTH_args.pruning_percentage, prune_model, mask)
    pruned_model = pruned_model.to(device)

    model_type.reinit(pruned_model, updated_mask, initial_state)
    pruned_model.mask = updated_mask

    pruned_model, best_val_accuracy, num_iterations = tvt.train(pruned_model, criterion, optimizer, exp_lr_scheduler, LTH_args.training_number_of_epochs, trainset, valset, trainloader, valloader)

    test_accuracy = tvt.test(pruned_model, testset, testloader)
    print('Test Accuracy with {} Percent of weights: {}'.format(1 - LTH_args.pruning_precentage, test_accuracy))


def iterative_pruning_by_num(prune_model, prune_mask: Mask, epochs: int):  # eventuell Reihenfolge ändern?
    """
    Prune iteratively for a number of epochs. Save checkpoint after every pruning epoch.
    Parameters
    ----------
    prune_model:
                 Model that is going to be iteratively pruned.
    prune_mask: Mask
                The mask used to prune the model.
    epochs: int
            Stopping criterion for the iterative pruning.
    """
    sparsity = []
    test_accuracies = []
    for epoch in range(epochs):
        prune_model.mask = prune_mask
        prune_model, best_val_accuracy, num_iterations = tvt.train(prune_model, criterion, optimizer, exp_lr_scheduler, LTH_args.training_number_of_epochs, trainset, valset, trainloader, valloader)

        test_accuracy = tvt.test(prune_model, testset, testloader)
        test_accuracies.append(test_accuracy * 100)
        sparsity.append(mask_sparsity(prune_mask) * 100)

        pruned_model, updated_mask = prune(LTH_args.pruning_percentage, prune_model, prune_mask)

        pruned_model = pruned_model.to(device)
        prune_model = pruned_model
        prune_mask = updated_mask

        model_type.reinit(prune_model, prune_mask, initial_state)

        checkpoint_save(epoch, pruned_model, updated_mask, test_accuracy, LTH_args.training_number_of_epochs, mask_sparsity(prune_mask) * 100)

        print('Pruning Epoch {}/{}'.format(epoch, LTH_args.iterative_pruning_epochs - 1))
        print('+' * 18)
        print('Model Test Accuracy: ', test_accuracy)
        print('at {} Training Iterations'.format(num_iterations))
        print('Sparsity of Pruned Mask: ', mask_sparsity(updated_mask))

    plot_results(LTH_args.iterative_pruning_epochs, test_accuracies, sparsity)


def iterative_pruning_by_test_acc(prune_model, prune_mask: Mask, acc_threshold: float):
    """
    Prune iteratively until the test accuracy is lower than the threshold. Save checkpoint after every pruning epoch.
    Parameters
    ----------
    prune_model:
                 Model that is going to be iteratively pruned.
    prune_mask: Mask
                The mask used to prune the model.
    acc_threshold: float
                   Stopping criterion for the iterative pruning.
    """
    sparsity = []
    test_accuracies = []
    num_pruning_epochs = 0
    sparsity.append(0)
    prune_model.mask = prune_mask
    prune_model, best_val_accuracy, num_iterations = tvt.train(prune_model, criterion, optimizer, exp_lr_scheduler, LTH_args.training_number_of_epochs, trainset, valset, trainloader, valloader)

    test_accuracy = tvt.test(prune_model, testset, testloader)
    test_accuracies.append(test_accuracy * 100)
    print('test accuracy: ', test_accuracy)

    checkpoint_save(num_pruning_epochs, prune_model, prune_mask, test_accuracy, LTH_args.training_number_of_epochs)

    while test_accuracy > acc_threshold:
        num_pruning_epochs += 1

        pruned_model, updated_mask = prune(LTH_args.pruning_percentage, prune_model, prune_mask)

        pruned_model = pruned_model.to(device)
        prune_model = pruned_model
        prune_mask = updated_mask

        if num_pruning_epochs == 1:
            model_type.reinit(prune_model, prune_mask, initial_state)

        sparsity.append(mask_sparsity(prune_mask) * 100)

        prune_model.mask = prune_mask

        prune_model, best_val_accuracy, num_iterations = tvt.train(prune_model, criterion, optimizer, exp_lr_scheduler, LTH_args.training_number_of_epochs, trainset, valset, trainloader, valloader)  # train

        test_accuracy = tvt.test(prune_model, testset, testloader)  # test
        test_accuracies.append(test_accuracy * 100)

        model_type.reinit(prune_model, prune_mask, initial_state)  # reinit

        checkpoint_save(num_pruning_epochs, prune_model, prune_mask, test_accuracy, LTH_args.training_number_of_epochs, mask_sparsity(prune_mask) * 100)  # save

        print('Pruning Epoch {}'.format(num_pruning_epochs))
        print('+' * 18)
        print('Model Test Accuracy: ', test_accuracy)
        print('at {} Training Iterations'.format(num_iterations))
        print('Sparsity of Pruned Mask: ', mask_sparsity(updated_mask))

    plot_results(num_pruning_epochs, test_accuracies, sparsity)


if LTH_args.stop_criteria is 'test_acc':
    iterative_pruning_by_test_acc(model, mask, LTH_args.test_accuracy_threshold)
elif LTH_args.stop_criteria is 'num_prune_epochs':
    iterative_pruning_by_num(model, mask, LTH_args.iterative_pruning_epochs)
elif LTH_args.stop_criteria is 'one_shot':
    one_shot_pruning(model, mask)
