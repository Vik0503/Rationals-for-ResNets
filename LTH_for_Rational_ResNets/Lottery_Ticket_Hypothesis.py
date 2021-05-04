from datetime import datetime

import torch
from torch import optim


from LTH_for_Rational_ResNets import Train_Val_Test as tvt
from LTH_for_Rational_ResNets import utils
from LTH_for_Rational_ResNets.Lottery_Ticket_Pruning import prune
from LTH_for_Rational_ResNets.Mask import mask_sparsity, Mask
import os

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
    PATH = '{}'.format(path) + '/{}_ep{}s{:.5f}_test{:.5f}.pth'.format(time_stamp, epoch, model_sparsity, test_accuracy)  # TODO: saved models names
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


def one_shot_pruning(prune_model, prune_mask: Mask, criterion, trainset, valset, trainloader, valloader, testset, testloader, pruning_percentage, training_number_of_epochs, lr: float, it_per_epoch: int, num_warmup_it: int):
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

    initial_state = utils.initial_state(model=prune_model)
    scheduler, optimizer = utils.get_scheduler_optimizer(lr=lr, it_per_ep=it_per_epoch, model=prune_model, num_warmup_it=num_warmup_it)
    prune_model, best_val_accuracy, num_iterations = tvt.train(prune_model, criterion, optimizer, scheduler, training_number_of_epochs, trainset, valset, trainloader, valloader)

    test_accuracy = tvt.test(prune_model, testset, testloader)
    print('Test Accuracy with 100 Percent of weights: ', test_accuracy)

    pruned_model, updated_mask = prune(pruning_percentage, prune_model, prune_mask)

    utils.reinit(pruned_model, updated_mask, initial_state)
    pruned_model.mask = Mask.cuda(updated_mask)

    pruned_model, best_val_accuracy, num_iterations = tvt.train(pruned_model, criterion, optimizer, scheduler, training_number_of_epochs, trainset, valset, trainloader, valloader)

    test_accuracy = tvt.test(pruned_model, testset, testloader)
    print('Test Accuracy with {} Percent of weights: {}'.format(1 - pruning_percentage, test_accuracy))


def iterative_pruning_by_num(prune_model, prune_mask: Mask, epochs: int, criterion, trainset, valset, trainloader, valloader, testset, testloader, pruning_percentage, training_number_of_epochs,
                             lr: float, it_per_epoch: int, num_warmup_it: int):
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
    global last_saved_checkpoint
    time_stamp = datetime.now()
    path = './Saved_Models/{}'.format(time_stamp)
    os.makedirs(path)
    sparsity = []
    test_accuracies = []
    initial_state = utils.initial_state(model=prune_model)

    prune_model.mask = Mask.cuda(prune_mask)

    initial_state = utils.initial_state(model=prune_model)

    scheduler, optimizer = utils.get_scheduler_optimizer(lr=lr, it_per_ep=it_per_epoch, model=prune_model, num_warmup_it=num_warmup_it)

    prune_model, best_val_accuracy, num_iterations = tvt.train(prune_model, criterion, optimizer, scheduler, training_number_of_epochs, trainset, valset, trainloader, valloader)

    test_accuracy = tvt.test(prune_model, testset, testloader)
    test_accuracies.append(test_accuracy * 100)
    print('Before Pruning')
    print('+' * 18)
    print('Model Test Accuracy: ', test_accuracy)

    for epoch in range(1, epochs + 1):

        print('Pruning Epoch {}'.format(epoch))
        print('+' * 18)

        pruned_model, updated_mask = prune(pruning_percentage, prune_model, prune_mask)
        sparsity.append(mask_sparsity(prune_mask) * 100)

        prune_model = pruned_model
        prune_mask = Mask.cuda(updated_mask)

        print('Sparsity of Pruned Mask: ', mask_sparsity(updated_mask))

        last_saved_checkpoint = checkpoint_save(path, optimizer, epoch, prune_model, prune_mask, test_accuracy, training_number_of_epochs, mask_sparsity(prune_mask) * 100)  # save

        utils.reinit(prune_model, prune_mask, initial_state)  # reinit

        scheduler, optimizer = utils.get_scheduler_optimizer(lr=lr, it_per_ep=it_per_epoch, model=prune_model, num_warmup_it=num_warmup_it)
        prune_model, best_val_accuracy, num_iterations = tvt.train(prune_model, criterion, optimizer, scheduler, training_number_of_epochs, trainset, valset, trainloader, valloader)  # train

        test_accuracy = tvt.test(prune_model, testset, testloader)  # test
        test_accuracies.append(test_accuracy * 100)

        print('Model Test Accuracy: ', test_accuracy)

    return test_accuracies, sparsity, path, last_saved_checkpoint


def iterative_pruning_by_test_acc(prune_model, prune_mask: Mask, acc_threshold: float, criterion, trainset, valset, trainloader, valloader, testset, testloader, training_number_of_epochs,
                                  pruning_percentage, lr: float, it_per_epoch: int, num_warmup_it: int):
    global last_saved_model_path
    time_stamp = datetime.now()
    path = './Saved_Models/{}'.format(time_stamp)
    os.makedirs(path)
    sparsity = []
    test_accuracies = []
    num_pruning_epochs = 0
    sparsity.append(0)

    prune_model.mask = Mask.cuda(prune_mask)

    initial_state = utils.initial_state(model=prune_model)

    scheduler, optimizer = utils.get_scheduler_optimizer(lr=lr, it_per_ep=it_per_epoch, model=prune_model, num_warmup_it=num_warmup_it)

    prune_model, best_val_accuracy, num_iterations = tvt.train(prune_model, criterion, optimizer, scheduler, training_number_of_epochs, trainset, valset, trainloader, valloader)

    test_accuracy = tvt.test(prune_model, testset, testloader)
    test_accuracies.append(test_accuracy * 100)
    print('Before Pruning')
    print('+' * 18)
    print('Model Test Accuracy: ', test_accuracy)

    while test_accuracy > acc_threshold:
        num_pruning_epochs += 1

        print('Pruning Epoch {}'.format(num_pruning_epochs))
        print('+' * 18)

        pruned_model, updated_mask = prune(pruning_percentage, prune_model, prune_mask)
        sparsity.append(mask_sparsity(prune_mask) * 100)

        prune_model = pruned_model
        prune_mask = Mask.cuda(updated_mask)

        print('Sparsity of Pruned Mask: ', mask_sparsity(updated_mask))

        last_saved_model_path = checkpoint_save(path, optimizer, num_pruning_epochs, prune_model, prune_mask, test_accuracy, training_number_of_epochs, mask_sparsity(prune_mask) * 100)  # save

        utils.reinit(prune_model, prune_mask, initial_state)  # reinit

        scheduler, optimizer = utils.get_scheduler_optimizer(lr=lr, it_per_ep=it_per_epoch, model=prune_model, num_warmup_it=num_warmup_it)
        prune_model, best_val_accuracy, num_iterations = tvt.train(prune_model, criterion, optimizer, scheduler, training_number_of_epochs, trainset, valset, trainloader, valloader)  # train

        test_accuracy = tvt.test(prune_model, testset, testloader)  # test
        test_accuracies.append(test_accuracy * 100)

        print('Model Test Accuracy: ', test_accuracy)

    return num_pruning_epochs, test_accuracies, sparsity, path, last_saved_model_path
