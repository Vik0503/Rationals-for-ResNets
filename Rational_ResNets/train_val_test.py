from __future__ import print_function, division

import copy
import time

import numpy as np
import torch
from rational.torch import Rational
from sklearn.metrics import confusion_matrix

from Rational_ResNets import plots

"""Method train_val_test_model based on https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html"""

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def train_val_test_model(model, criterion, optimizer, scheduler, num_epochs, trainloader, valloader, testloader, trainset, valset, testset, exp_name, learning_rate, batch_size):
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

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
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
                scheduler.step()

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

        for mod in model.modules():
            if isinstance(mod, Rational):
                # print(mod)
                # mod.show()
                print(mod.numerator)

        cm = torch.tensor(confusion_matrix(labels.to('cpu'), preds.to('cpu')))
        print(cm)

        time_elapsed_epoch = time.time() - since_epoch
        avg_epoch_time.append(time_elapsed_epoch)
        print('Epoch finished in {:.0f}m {:.0f}s'.format(time_elapsed_epoch // 60, time_elapsed_epoch % 60))

    time_elapsed_epoch = average_epoch_time(avg_epoch_time)

    plots.final_plot(acc_x_vals=accuracy_plot_x_vals, train_acc_y_vals=train_acc_plot_y_vals, val_acc_y_vals=val_acc_plot_y_vals, test_acc_y_vals=test_acc_plot_y_vals, cm=cm, num_epochs=num_epochs, epoch_time=time_elapsed_epoch,
                     test_acc=best_acc, exp_name=exp_name, batch_size=batch_size, learning_rate=learning_rate, dataset='SVHN')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model)

    return model


def average_epoch_time(avg_epoch_time):  # calculates average time per epoch
    avg_epoch = np.sum(avg_epoch_time)
    return avg_epoch / len(avg_epoch_time)


