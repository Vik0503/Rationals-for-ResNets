from __future__ import print_function, division

import copy
import time

import torch

from Rational_ResNets.ResNet_Datasets import CIFAR10, SVHN

torch.cuda.manual_seed_all(42)

import numpy as np
np.random.seed(42)

from sklearn.metrics import confusion_matrix

import argparser

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

resnet_args = argparser.get_arguments()


if resnet_args.dataset == 'cifar10':
    trainset, trainloader, _ = CIFAR10.get_train_data(aug=resnet_args.augment_data, bs=resnet_args.batch_size)
    valset, valloader = CIFAR10.get_validation_data(aug=resnet_args.augment_data, bs=resnet_args.batch_size)
    testset, testloader = CIFAR10.get_test_data(aug=resnet_args.augment_data, bs=resnet_args.batch_size)
    classes = CIFAR10.get_classes()
    num_classes = CIFAR10.get_num_classes()

elif resnet_args.dataset == 'SVHN':
    trainset, trainloader, _ = SVHN.get_train_data(aug=resnet_args.augment_data, bs=resnet_args.batch_size)
    valset, valloader = SVHN.get_validation_data(aug=resnet_args.augment_data, bs=resnet_args.batch_size)
    testset, testloader = SVHN.get_test_data(aug=resnet_args.augment_data, bs=resnet_args.batch_size)
    classes = SVHN.get_classes()
    num_classes = SVHN.get_num_classes()


def train_val_test_model(model, optimizer, scheduler):
    criterion = nn.CrossEntropyLoss()
    best_model = copy.deepcopy(model.state_dict())
    since = time.time()
    avg_epoch_time = []
    best_acc = 0.0
    all_test_labels = []
    all_test_preds = []
    accuracy_plot_x_vals = []
    train_acc_plot_y_vals = []
    val_acc_plot_y_vals = []
    test_acc_plot_y_vals = []
    step = 0
    for epoch in range(resnet_args.training_number_of_epochs):
        print('Epoch {}/{}'.format(epoch, resnet_args.training_number_of_epochs - 1))
        print('*' * 10)
        since_epoch = time.time()

        # Each epoch has a training, a validation and a test phase
        for phase in ['train', 'val', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0

            if phase == 'train':
                dataloader = trainloader
            if phase == 'val':
                dataloader = valloader
            if phase == 'test':
                dataloader = testloader
            # Iterate over data.
            for i, data in enumerate(dataloader, 0):
                step += 1
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                if phase == 'train':
                    scheduler.step()

                if phase == 'test':
                    all_test_preds.append(preds.cpu().numpy())
                    all_test_labels.append(labels.cpu().numpy())

                # loss + accuracy
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                epoch_loss = running_loss / len(trainset)
                epoch_acc = running_corrects.double() / len(trainset)
                train_acc_plot_y_vals.append(epoch_acc.cpu() * 100)

            if phase == 'val':
                epoch_loss = running_loss / len(valset)
                epoch_acc = running_corrects.double() / len(valset)
                val_acc_plot_y_vals.append(epoch_acc.cpu() * 100)

            if phase == 'test':
                epoch_loss = running_loss / len(testset)
                epoch_acc = running_corrects.double() / len(testset)
                test_acc_plot_y_vals.append(epoch_acc.cpu() * 100)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model.state_dict())

        accuracy_plot_x_vals.append(epoch)

        cm = torch.tensor(confusion_matrix(labels.to('cpu'), preds.to('cpu')))
        print(cm)

        time_elapsed_epoch = time.time() - since_epoch
        avg_epoch_time.append(time_elapsed_epoch)
        print('Epoch finished in {:.0f}m {:.0f}s'.format(time_elapsed_epoch // 60, time_elapsed_epoch % 60))

    time_elapsed_epoch = average_epoch_time(avg_epoch_time)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model)

    return model, cm, time_elapsed_epoch, best_acc, train_acc_plot_y_vals, val_acc_plot_y_vals, test_acc_plot_y_vals, accuracy_plot_x_vals


def average_epoch_time(avg_epoch_time: list):
    """ Calculate the average time per epoch. """
    avg_epoch = np.sum(avg_epoch_time)
    return avg_epoch / len(avg_epoch_time)


