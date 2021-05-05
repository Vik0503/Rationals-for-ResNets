import os
from datetime import datetime

import torch

from LTH_for_Rational_ResNets import Train_Val_Test as tvt
from LTH_for_Rational_ResNets import argparser
from LTH_for_Rational_ResNets import utils
from LTH_for_Rational_ResNets.Lottery_Ticket_Pruning import prune
from LTH_for_Rational_ResNets.Mask import make_initial_mask
from LTH_for_Rational_ResNets.Mask import mask_sparsity, Mask

LTH_args = argparser.get_arguments()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def checkpoint_save(path, optimizer, epoch: int, save_model, model_mask: Mask, test_accuracy: float, training_epochs: int, model_sparsity: float = 0):
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
    PATH = '{}'.format(path) + '/{}_ep{}s{:.5f}_test{:.5f}.pth'.format(time_stamp, epoch, model_sparsity, test_accuracy)
    torch.save({
        'epoch': epoch,
        'model_state_dict': save_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'mask': model_mask,
        'sparsity': model_sparsity,
        'test_accuracy': test_accuracy,
        'training_epochs': training_epochs,
    }, PATH)

    return PATH


def one_shot_pruning(model):  # TODO: it per epoch in utils?
    """
    Prune trained model once and reinitialize and test it.

    Parameters
    ----------
    model:
                 Model that is going to be iteratively pruned.
    """
    mask = make_initial_mask(model)
    model.mask = mask

    initial_state = utils.initial_state(model=model)
    scheduler, optimizer = utils.get_scheduler_optimizer(model=model)
    best_val_accuracy = tvt.train(model, optimizer, scheduler)
    print('Best validation accuracy: {}'.format_map(best_val_accuracy))

    test_accuracy = tvt.test(model)
    print('Test Accuracy with 100 Percent of weights: ', test_accuracy)

    prune(LTH_args.pruning_percentage, model, mask)

    utils.reinit(model, mask, initial_state)
    model.mask = Mask.cuda(mask)

    best_val_accuracy = tvt.train(model, optimizer, scheduler)
    print('Best validation accuracy: {}'.format_map(best_val_accuracy))

    test_accuracy = tvt.test(model)
    print('Test Accuracy with {} Percent of weights: {}'.format(1 - LTH_args.pruning_percentage, test_accuracy))


def iterative_pruning_by_num(prune_model):
    """
    Prune iteratively for a number of epochs. Save checkpoint after every pruning epoch.

    Parameters
    ----------
    prune_model:
                 Model that is going to be iteratively pruned.
    """
    last_saved_checkpoint_PATH = ''
    time_stamp = datetime.now()
    path = './Saved_Models/{}'.format(time_stamp)
    os.makedirs(path)
    sparsity = [0]
    test_accuracies = []

    model = prune_model
    mask = make_initial_mask(model)
    model.mask = Mask.cuda(mask)

    initial_state = utils.initial_state(model=model)

    scheduler, optimizer = utils.get_scheduler_optimizer(model=model)

    best_val_accuracy = tvt.train(model, optimizer, scheduler)
    print('Best validation accuracy: {}'.format_map(best_val_accuracy))

    test_accuracy = tvt.test(model)
    test_accuracies.append(test_accuracy * 100)
    print('Before Pruning')
    print('+' * 18)
    print('Model Test Accuracy: ', test_accuracy)

    for epoch in range(1, LTH_args.iterative_pruning_epochs + 1):

        print('Pruning Epoch {}'.format(epoch))
        print('+' * 18)

        prune(LTH_args.pruning_percentage, model, mask)
        sparsity.append(mask_sparsity(mask) * 100)

        print('Sparsity of Pruned Mask: ', mask_sparsity(mask))

        last_saved_checkpoint_PATH = checkpoint_save(path, optimizer, epoch, model, mask, test_accuracy, LTH_args.training_number_of_epochs, mask_sparsity(mask) * 100)  # save

        utils.reinit(model, mask, initial_state)  # reinit

        scheduler, optimizer = utils.get_scheduler_optimizer(model=model)
        best_val_accuracy = tvt.train(model, optimizer, scheduler)  # train
        print('Best validation accuracy: {}'.format_map(best_val_accuracy))

        test_accuracy = tvt.test(model)  # test
        test_accuracies.append(test_accuracy * 100)

        print('Model Test Accuracy: ', test_accuracy)

    return test_accuracies, sparsity, path, last_saved_checkpoint_PATH


def iterative_pruning_by_test_acc(prune_model):
    last_saved_model_PATH = ''
    time_stamp = datetime.now()
    saved_models_PATH = './Saved_Models/{}'.format(time_stamp)
    os.makedirs(saved_models_PATH)
    sparsity = []
    test_accuracies = []
    num_pruning_epochs = 0
    sparsity.append(0)

    model = prune_model
    mask = make_initial_mask(model)
    model.mask = Mask.cuda(mask)

    model.mask = Mask.cuda(mask)

    initial_state = utils.initial_state(model=model)

    scheduler, optimizer = utils.get_scheduler_optimizer(model=model)

    best_val_accuracy = tvt.train(model, optimizer, scheduler)
    print('Best validation accuracy: {}'.format(best_val_accuracy))

    test_accuracy = tvt.test(model)
    test_accuracies.append(test_accuracy * 100)
    print('Before Pruning')
    print('+' * 18)
    print('Model Test Accuracy: ', test_accuracy)

    if test_accuracy <= LTH_args.test_accuracy_threshold:
        print('Test accuracy was too low, there was nothing pruned!')
        exit()

    while test_accuracy > LTH_args.test_accuracy_threshold:
        num_pruning_epochs += 1

        print('Pruning Epoch {}'.format(num_pruning_epochs))
        print('+' * 18)

        prune(LTH_args.pruning_percentage, model, mask)
        sparsity.append(mask_sparsity(mask) * 100)

        print('Sparsity of Pruned Mask: ', mask_sparsity(mask))

        last_saved_model_PATH = checkpoint_save(saved_models_PATH, optimizer, num_pruning_epochs, model, mask, test_accuracy, LTH_args.training_number_of_epochs, mask_sparsity(mask) * 100)  # save

        utils.reinit(model, mask, initial_state)  # reinit

        scheduler, optimizer = utils.get_scheduler_optimizer(model=model)
        best_val_accuracy = tvt.train(model, optimizer, scheduler)  # train
        print('Best validation accuracy: {}'.format(best_val_accuracy))

        test_accuracy = tvt.test(model)  # test
        test_accuracies.append(test_accuracy * 100)

        print('Model Test Accuracy: ', test_accuracy)

    return num_pruning_epochs, test_accuracies, sparsity, saved_models_PATH, last_saved_model_PATH
