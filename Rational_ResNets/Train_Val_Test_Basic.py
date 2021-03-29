from __future__ import print_function, division

import argparse as arg
import time

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

from sklearn.metrics import confusion_matrix
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from Rational_ResNets.ResNet_Datasets import CIFAR10, SVHN
from Rational_ResNets.ResNet_Models import Rational_ResNet18_ImageNet as RRN18
from Rational_ResNets.ResNet_Models import Rational_ResNet20_CiFAR10 as RRN20
from Rational_ResNets.ResNet_Models import ResNet18_ImageNet as RN18
from Rational_ResNets.ResNet_Models import ResNet20_CIFAR10 as RN20
from Rational_ResNets.ResNet_Models import Multi_Variant_Rational_ResNet20_CIFAR10 as MVRRN20
from Rational_ResNets.ResNet_Models import Pytorch_Rational_ResNets_ImageNet as PT

ResNet_arg_parser = arg.ArgumentParser()
ResNet_arg_parser.add_argument('-bs', '--batch_size', default=128, type=int)
ResNet_arg_parser.add_argument('-lr', '--learning_rate', default=0.01, type=float)
ResNet_arg_parser.add_argument('-m', '--model', default='rational_resnet20_cifar10', type=str,
                               choices=['rational_resnet20_cifar10', 'resnet20_cifar10', 'rational_resnet18_imagenet', 'resnet18_imagenet', 'multi_rational_resnet20_cifar10',
                                        'pt'])  # pt is the original ResNet18 model from Pytorch with Rationals
ResNet_arg_parser.add_argument('-ds', '--dataset', default='cifar10', type=str, choices=['cifar10', 'SVHN'])
ResNet_arg_parser.add_argument('-tnep', '--training_number_of_epochs', default=2, type=int)

ResNet_args = ResNet_arg_parser.parse_args(['--model', 'rational_resnet20_cifar10', '--dataset', 'SVHN'])

global trainset
global valset
global testset
global trainloader
global valloader
global testloader
global classes
global num_classes
global model
global model_type

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

if ResNet_args.dataset is 'cifar10':
    trainset = CIFAR10.get_train_data()
    valset = CIFAR10.get_validationset()
    testset = CIFAR10.get_testset()
    trainloader = CIFAR10.get_trainloader(ResNet_args.batch_size)
    valloader = CIFAR10.get_validation_data(ResNet_args.batch_size)
    testloader = CIFAR10.get_test_data(ResNet_args.batch_size)
    classes = CIFAR10.get_classes()
    num_classes = CIFAR10.get_num_classes()

elif ResNet_args.dataset is 'SVHN':
    trainset = SVHN.get_trainset()
    valset = SVHN.get_validationset()
    testset = SVHN.get_testset()
    trainloader = SVHN.get_trainloader(ResNet_args.batch_size)
    valloader = SVHN.get_valloader(ResNet_args.batch_size)
    testloader = SVHN.get_testloader(ResNet_args.batch_size)
    classes = SVHN.get_classes()
    num_classes = SVHN.get_num_classes()

if ResNet_args.model is 'rational_resnet20_cifar10':
    model = RRN20.rational_resnet20()
    model_type = RRN20
elif ResNet_args.model is 'resnet20_cifar10':
    model = RN20.resnet20()
    model_type = RN20
elif ResNet_args.model is 'rational_resnet18_imagenet':
    model = RRN18.rational_resnet18()
    model_type = RRN18
elif ResNet_args.model is 'resnet18_imagenet':
    model = RN18.resnet18()
    model_type = RN18
elif ResNet_args.model is 'multi_rational_resnet20_cifar10':
    model = MVRRN20.multi_variant_rational_resnet20()
    model_type = MVRRN20
elif ResNet_args.model is 'pt':
    model = PT.resnet18()
    model_type = PT

    writer = SummaryWriter('runs/rational_resnet18_SVHN_run21')

"""Method train_val_test_model based on https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html"""


def train_val_test_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    avg_epoch_time = []
    best_acc = 0.0
    all_train_preds = []
    all_train_labels = []
    all_test_labels = []
    all_test_preds = []
    accuracy_plot_x_vals = []
    train_acc_plot_y_vals = []
    val_acc_plot_y_vals = []
    test_acc_plot_y_vals = []

    for epoch in range(ResNet_args.training_number_of_epochs):
        print('Epoch {}/{}'.format(epoch, ResNet_args.training_number_of_epochs - 1))
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
                        loss = loss.to(device)
                        optimizer.step()
                        all_train_preds.append(preds.cpu().numpy())
                        all_train_labels.append(labels.cpu().numpy())

                if phase == 'test':
                    all_test_preds.append(preds.cpu().numpy())
                    all_test_labels.append(labels.cpu().numpy())

                # loss + accuracy
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                epoch_loss = running_loss / len(trainset)
                writer.add_scalar('training loss', epoch_loss, epoch)
                epoch_acc = running_corrects.double() / len(trainset)
                writer.add_scalar('training accuracy', epoch_acc, epoch)
                train_acc_plot_y_vals.append(epoch_acc.cpu() * 100)
                scheduler.step()

            if phase == 'val':
                epoch_loss = running_loss / len(valset)
                writer.add_scalar('validation loss', epoch_loss, epoch)
                epoch_acc = running_corrects.double() / len(valset)
                val_acc_plot_y_vals.append(epoch_acc.cpu() * 100)
                writer.add_scalar('validation accuracy', epoch_acc, epoch)

            if phase == 'test':
                epoch_loss = running_loss / len(testset)
                writer.add_scalar('test loss', epoch_loss, epoch)
                epoch_acc = running_corrects.double() / len(testset)
                test_acc_plot_y_vals.append(epoch_acc.cpu() * 100)
                writer.add_scalar('test accuracy', epoch_acc, epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc

        accuracy_plot_x_vals.append(epoch)

        cm = torch.tensor(confusion_matrix(labels.to('cpu'), preds.to('cpu')))
        print('confusion matrix: ', cm)

        time_elapsed_epoch = time.time() - since_epoch
        avg_epoch_time.append(time_elapsed_epoch)
        print('Epoch finished in {:.0f}m {:.0f}s'.format(time_elapsed_epoch // 60, time_elapsed_epoch % 60))

    summary_plot(accuracy_plot_x_vals, train_acc_plot_y_vals, val_acc_plot_y_vals, test_acc_plot_y_vals)
    train_labels_array = confusion_prepare(all_train_labels)
    train_preds_array = confusion_prepare(all_train_preds)
    cm_train = confusion_matrix(train_labels_array, train_preds_array, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    print('cm_train: ', cm_train)
    plot_confusion_matrix(cm_train, num_epochs, time_elapsed_epoch, 'train_1.svg')

    test_labels_array = confusion_prepare(all_test_labels)
    test_preds_array = confusion_prepare(all_test_preds)
    cm_test = confusion_matrix(test_labels_array, test_preds_array, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    print('cm_test: ', cm_test)
    time_elapsed_epoch = average_epoch_time(avg_epoch_time)
    plot_confusion_matrix(cm_test, num_epochs, time_elapsed_epoch, 'test_1.svg', best_acc)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    return model


def average_epoch_time(avg_epoch_time):  # calculates average time per epoch
    avg_epoch_int = 0.0
    for i in range(len(avg_epoch_time)):
        avg_epoch_int += avg_epoch_time[i]
    return avg_epoch_int / len(avg_epoch_time)


def confusion_prepare(pl):  # converts pred + acc arrays of arrays with tensors to array
    pl_array = []
    for i in range(len(pl)):
        k = pl[i]
        for j in range(len(k)):
            pl_array.append(k[j])
    return pl_array


def plot_confusion_matrix(cm, num_epochs, epoch_time, title, test_acc):  # plots confusion matrix as heatmap
    plt.subplot(132)
    cm_1 = sns.heatmap(cm, linewidths=1, cmap='plasma')
    props = dict(boxstyle='round', facecolor='grey', alpha=0.5)
    text = 'num epochs: {}, '.format(num_epochs) + 'num params: {}, '.format(num_param) + \
           'batch size: 128, ' + 'lr: 0.01, ' + '\n' + \
           'avg time per epoch: {:.0f}m {:.0f}s, '.format(epoch_time // 60, epoch_time % 60) + \
           'test accuracy: {:4f}, '.format(test_acc) + 'dataset: SVHN'
    plt.text(15, 5, text, bbox=props)
    cm_1.figure.savefig(title)


def summary_plot(acc_x_vals, train_acc_y_vals, val_acc_y_vals, test_acc_y_vals):  # train and val accuracy plot
    plt.figure(figsize=(40, 10))
    plt.subplot(131)
    plt.plot(acc_x_vals, train_acc_y_vals)
    plt.plot(acc_x_vals, val_acc_y_vals)
    plt.plot(acc_x_vals, test_acc_y_vals)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Train Accuracy', 'Validation Accuracy', 'Test Accuracy'])

    plt.savefig('test_plot.svg')


def final_plot():
    plt.savefig('summary_plot_rational_resnet_SVHN.svg')


def layer_plot():  # plots Rational Activation functions in all layers
    RRN18_model = torch.load('./saved_models/model9.pth', map_location=torch.device('cpu')).to('cpu')
    rational_0 = RRN18_model.relu.cpu() # warum überall relu???
    print(RRN18_model.layer1.__getitem__(0).relu.cpu())
    rational_1 = RRN18_model.layer1.__getitem__(0).relu.cpu()
    rational_2 = RRN18_model.layer1.__getitem__(1).relu.cpu()
    rational_3 = RRN18_model.layer2.__getitem__(0).relu.cpu()
    rational_4 = RRN18_model.layer2.__getitem__(1).relu.cpu()
    rational_5 = RRN18_model.layer3.__getitem__(0).relu.cpu()
    rational_6 = RRN18_model.layer3.__getitem__(1).relu.cpu()
    rational_7 = RRN18_model.layer4.__getitem__(0).relu.cpu()
    rational_8 = RRN18_model.layer4.__getitem__(1).relu.cpu()
    plot = rational_0.show(display=False)
    plot = rational_1.show(display=False)
    plot = rational_2.show(display=False)
    plot = rational_3.show(display=False)
    plot = rational_4.show(display=False)
    plot = rational_5.show(display=False)
    plot = rational_6.show(display=False)
    plot = rational_7.show(display=False)
    plot = rational_8.show(display=False)
    plot.legend(['Layer 0 Rational Function', 'Layer 1 Rational Function 0', 'Layer 1 Rational Function 1',
                 'Layer 2 Rational Function 0',
                 'Layer 2 Rational Function 1', 'Layer 3 Rational Function 0', 'Layer 3 Rational Function 1',
                 'Layer 4 Rational Function 0', 'Layer 4 Rational Function 1'])
    plot.savefig('resnet_plots_3.svg')


num_ftrs = model.fc.in_features
# Here the size of each output sample is set to nn.Linear(num_ftrs, len(class_names)).
class_names = trainset.labels
model.fc = nn.Linear(num_ftrs, len(class_names))

model = model.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

num_param = model.parameters().__sizeof__()

model = train_val_test_model(model, criterion, optimizer, exp_lr_scheduler,
                             num_epochs=2)

final_plot()

torch.save(model, './model9.pth')
