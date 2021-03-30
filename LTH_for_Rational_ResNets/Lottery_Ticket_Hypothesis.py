from copy import deepcopy

import torch

import main
import plots
from LTH_for_Rational_ResNets import Train_Val_Test as tvt
from LTH_for_Rational_ResNets.Lottery_Ticket_Pruning import prune
from LTH_for_Rational_ResNets.Mask import mask_sparsity, Mask

LTH_args = main.get_LTH_args()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


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


def one_shot_pruning(prune_model, prune_mask: Mask, optimizer, criterion, exp_lr_scheduler, trainset, valset, trainloader, valloader, model_type, testset, testloader):
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

    initial_state = deepcopy(prune_model.state_dict())

    prune_model, best_val_accuracy, num_iterations = tvt.train(prune_model, criterion, optimizer, exp_lr_scheduler, LTH_args.training_number_of_epochs, trainset, valset, trainloader, valloader)

    test_accuracy = tvt.test(prune_model, testset, testloader)
    print('Test Accuracy with 100 Percent of weights: ', test_accuracy)

    pruned_model, updated_mask = prune(LTH_args.pruning_percentage, prune_model, prune_mask)
    pruned_model = pruned_model.to(device)

    model_type.reinit(pruned_model, updated_mask, initial_state)
    pruned_model.mask = updated_mask

    pruned_model, best_val_accuracy, num_iterations = tvt.train(pruned_model, criterion, optimizer, exp_lr_scheduler, LTH_args.training_number_of_epochs, trainset, valset, trainloader, valloader)

    test_accuracy = tvt.test(pruned_model, testset, testloader)
    print('Test Accuracy with {} Percent of weights: {}'.format(1 - LTH_args.pruning_precentage, test_accuracy))


def iterative_pruning_by_num(prune_model, prune_mask: Mask, epochs: int, optimizer, criterion, exp_lr_scheduler, trainset, valset, trainloader, valloader, model_type, testset, testloader):  # eventuell Reihenfolge ändern?
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

        checkpoint_save(optimizer, epoch, pruned_model, updated_mask, test_accuracy, LTH_args.training_number_of_epochs, mask_sparsity(prune_mask) * 100)

        print('Pruning Epoch {}/{}'.format(epoch, LTH_args.iterative_pruning_epochs - 1))
        print('+' * 18)
        print('Model Test Accuracy: ', test_accuracy)
        print('at {} Training Iterations'.format(num_iterations))
        print('Sparsity of Pruned Mask: ', mask_sparsity(updated_mask))

    plots.make_LTH_plot(LTH_args.iterative_pruning_epochs, test_accuracies, sparsity)


def iterative_pruning_by_test_acc(prune_model, prune_mask: Mask, acc_threshold: float, optimizer, criterion, exp_lr_scheduler, trainset, valset, trainloader, valloader, model_type, testset, testloader):
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
    initial_state = deepcopy(prune_model.state_dict())

    prune_model, best_val_accuracy, num_iterations = tvt.train(prune_model, criterion, optimizer, exp_lr_scheduler, LTH_args.training_number_of_epochs, trainset, valset, trainloader, valloader)

    test_accuracy = tvt.test(prune_model, testset, testloader)
    test_accuracies.append(test_accuracy * 100)
    print('test accuracy: ', test_accuracy)

    checkpoint_save(optimizer, num_pruning_epochs, prune_model, prune_mask, test_accuracy, LTH_args.training_number_of_epochs)

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

        checkpoint_save(optimizer, num_pruning_epochs, prune_model, prune_mask, test_accuracy, LTH_args.training_number_of_epochs, mask_sparsity(prune_mask) * 100)  # save

        print('Pruning Epoch {}'.format(num_pruning_epochs))
        print('+' * 18)
        print('Model Test Accuracy: ', test_accuracy)
        print('at {} Training Iterations'.format(num_iterations))
        print('Sparsity of Pruned Mask: ', mask_sparsity(updated_mask))

    plots.make_LTH_plot(num_pruning_epochs, test_accuracies, sparsity)



