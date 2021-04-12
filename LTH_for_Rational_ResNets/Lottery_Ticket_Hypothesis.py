from datetime import datetime
from copy import deepcopy

import torch
from torch import optim
from torch.optim import lr_scheduler

from LTH_for_Rational_ResNets import plots
from LTH_for_Rational_ResNets import Train_Val_Test as tvt
from LTH_for_Rational_ResNets.Lottery_Ticket_Pruning import prune
from LTH_for_Rational_ResNets.Mask import mask_sparsity, Mask
from LTH_for_Rational_ResNets import utils

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def get_scheduler_optimizer(num_warmup_it, lr, model, it_per_ep):  # TODO: allow diff. milestones maybe in utils?
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)

    def lr_lambda(it):
        if it < num_warmup_it:
            return min(1.0, it / num_warmup_it)
        elif it < 10 * it_per_ep:
            return 1
        elif 10 * it_per_ep <= it < 15 * it_per_ep:
            return 0.1
        elif 15 * it_per_ep <= it < 20 * it_per_ep:
            return 0.01
        elif it >= 20 * it_per_ep:
            return 0.001

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda), optimizer


def checkpoint_save(optimizer, epoch: int, save_model, model_mask: Mask, test_accuracy: float, training_epochs: int, model_sparsity: float = 0):
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
    time_stamp = datetime.now()
    PATH = '/home/viktoria/Git/Rationals-for-ResNets/LTH_for_Rational_ResNets/Saved_Models/{}_ep{}s{:.5f}_test{:.5f}.pth'.format(time_stamp, epoch, model_sparsity, test_accuracy)  # TODO: Update PATH + saved models names
    torch.save({
        'epoch': epoch,
        'model_state_dict': save_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'mask': model_mask,
        'sparsity': model_sparsity,
        'test_accuracy': test_accuracy,
        'training_epochs': training_epochs,
    }, PATH)


def one_shot_pruning(prune_model, prune_mask: Mask, optimizer, criterion, exp_lr_scheduler, trainset, valset, trainloader, valloader, testset, testloader, pruning_percentage, training_number_of_epochs):
    """
    Prune trained model once and reinitialize and test it.

    Parameters
    ----------
    prune_model:
                 Model that is going to be iteratively pruned.
    prune_mask: Mask
                The mask used to prune the model.
    """
    prune_model.mask = Mask.cuda(prune_mask)

    initial_state = deepcopy(prune_model.state_dict())

    prune_model, best_val_accuracy, num_iterations = tvt.train(prune_model, criterion, optimizer, exp_lr_scheduler, training_number_of_epochs, trainset, valset, trainloader, valloader)

    test_accuracy = tvt.test(prune_model, testset, testloader)
    print('Test Accuracy with 100 Percent of weights: ', test_accuracy)

    pruned_model, updated_mask = prune(pruning_percentage, prune_model, prune_mask)

    utils.reinit(pruned_model, updated_mask, initial_state)
    pruned_model.mask = Mask.cuda(updated_mask)

    pruned_model, best_val_accuracy, num_iterations = tvt.train(pruned_model, criterion, optimizer, exp_lr_scheduler, training_number_of_epochs, trainset, valset, trainloader, valloader)

    test_accuracy = tvt.test(pruned_model, testset, testloader)
    print('Test Accuracy with {} Percent of weights: {}'.format(1 - pruning_percentage, test_accuracy))


def iterative_pruning_by_num(prune_model, prune_mask: Mask, epochs: int, criterion, trainset, valset, trainloader, valloader, testset, testloader, pruning_percentage, training_number_of_epochs,
                             lr: float, it_per_epoch: int, num_warmup_it: int):
    # TODO: update order
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
    initial_state = deepcopy(prune_model.state_dict())

    for epoch in range(epochs):
        prune_model.mask = Mask.cuda(prune_mask)
        scheduler, optimizer = get_scheduler_optimizer(lr=lr, it_per_ep=it_per_epoch, model=prune_model, num_warmup_it=num_warmup_it)
        prune_model, best_val_accuracy, num_iterations = tvt.train(prune_model, criterion, optimizer, scheduler, training_number_of_epochs, trainset, valset, trainloader, valloader)

        test_accuracy = tvt.test(prune_model, testset, testloader)
        test_accuracies.append(test_accuracy * 100)
        sparsity.append(mask_sparsity(prune_mask) * 100)
        pruned_model, updated_mask = prune(pruning_percentage, prune_model, prune_mask)

        prune_model = pruned_model
        prune_mask = Mask.cuda(updated_mask)

        utils.reinit(prune_model, prune_mask, initial_state)

        checkpoint_save(optimizer, epoch, pruned_model, updated_mask, test_accuracy, training_number_of_epochs, mask_sparsity(prune_mask) * 100)

        print('Pruning Epoch {}/{}'.format(epoch, epochs - 1))
        print('+' * 18)
        print('Model Test Accuracy: ', test_accuracy)
        print('at {} Training Iterations'.format(num_iterations))
        print('Sparsity of Pruned Mask: ', mask_sparsity(updated_mask))

    return test_accuracies, sparsity


def iterative_pruning_by_test_acc(prune_model, prune_mask: Mask, acc_threshold: float, criterion, trainset, valset, trainloader, valloader, testset, testloader, training_number_of_epochs,
                                  pruning_percentage, lr: float, it_per_epoch: int, num_warmup_it: int):
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
    prune_model.mask = Mask.cuda(prune_mask)
    initial_state = deepcopy(prune_model.state_dict())

    scheduler, optimizer = get_scheduler_optimizer(lr=lr, it_per_ep=it_per_epoch, model=prune_model, num_warmup_it=num_warmup_it)
    prune_model, best_val_accuracy, num_iterations = tvt.train(prune_model, criterion, optimizer, scheduler, training_number_of_epochs, trainset, valset, trainloader, valloader)

    test_accuracy = tvt.test(prune_model, testset, testloader)
    test_accuracies.append(test_accuracy * 100)
    print('test accuracy: ', test_accuracy)

    checkpoint_save(optimizer, num_pruning_epochs, prune_model, prune_mask, test_accuracy, training_number_of_epochs)

    while test_accuracy > acc_threshold:
        num_pruning_epochs += 1

        pruned_model, updated_mask = prune(pruning_percentage, prune_model, prune_mask)

        prune_model = pruned_model
        prune_mask = Mask.cuda(updated_mask)

        if num_pruning_epochs == 1:
            utils.reinit(prune_model, prune_mask, initial_state)

        sparsity.append(mask_sparsity(prune_mask) * 100)

        prune_model.mask = prune_mask
        scheduler, optimizer = get_scheduler_optimizer(lr=lr, it_per_ep=it_per_epoch, model=prune_model, num_warmup_it=num_warmup_it)
        prune_model, best_val_accuracy, num_iterations = tvt.train(prune_model, criterion, optimizer, scheduler, training_number_of_epochs, trainset, valset, trainloader, valloader)  # train

        test_accuracy = tvt.test(prune_model, testset, testloader)  # test
        test_accuracies.append(test_accuracy * 100)

        utils.reinit(prune_model, prune_mask, initial_state)  # reinit

        checkpoint_save(optimizer, num_pruning_epochs, prune_model, prune_mask, test_accuracy, training_number_of_epochs, mask_sparsity(prune_mask) * 100)  # save

        print('Pruning Epoch {}'.format(num_pruning_epochs))
        print('+' * 18)
        print('Model Test Accuracy: ', test_accuracy)
        print('at {} Training Iterations'.format(num_iterations))
        print('Sparsity of Pruned Mask: ', mask_sparsity(updated_mask))

    return num_pruning_epochs, test_accuracies, sparsity
