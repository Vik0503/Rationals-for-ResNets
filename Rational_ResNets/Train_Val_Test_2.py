from __future__ import print_function, division
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse as arg
import copy
import time

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import torch
import torch.nn as nn
from rational.torch import Rational
import numpy as np

from sklearn.metrics import confusion_matrix
from torch import optim
from torch.optim import lr_scheduler

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

ResNet_args = ResNet_arg_parser.parse_args(['--model', 'multi_rational_resnet20_cifar10', '--dataset', 'SVHN'])

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
    trainset = CIFAR10.get_trainset()
    valset = CIFAR10.get_validationset()
    testset = CIFAR10.get_testset()
    trainloader = CIFAR10.get_trainloader(ResNet_args.batch_size)
    valloader = CIFAR10.get_valloader(ResNet_args.batch_size)
    testloader = CIFAR10.get_testloader(ResNet_args.batch_size)
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


"""Method train_val_test_model based on https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html"""


def train_val_test_model(model, criterion, optimizer, scheduler, num_epochs):
    # import ipdb; ipdb.set_trace()
    torch.autograd.set_detect_anomaly(True)
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

    summary_plot(accuracy_plot_x_vals, train_acc_plot_y_vals, val_acc_plot_y_vals, test_acc_plot_y_vals)
    plot_confusion_matrix(cm, num_epochs, time_elapsed_epoch, best_acc)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model)

    return model


plt.style.use(["science", "grid"])
matplotlib.rcParams.update({
    "font.family": "serif",
    "text.usetex": False,
})


def average_epoch_time(avg_epoch_time):  # calculates average time per epoch. not yet tested!!!
    avg_epoch = np.sum(avg_epoch_time)
    return avg_epoch / len(avg_epoch_time)


def plot_confusion_matrix(cm, num_epochs, epoch_time, test_acc):  # plots confusion matrix as heatmap
    plt.subplot(132)
    cm_1 = sns.heatmap(cm, linewidths=1, cmap='plasma')
    props = dict(boxstyle='round', facecolor='grey', alpha=0.5)
    text = 'num epochs: {}, '.format(num_epochs) + \
           'batch size: {}, '.format(ResNet_args.batch_size) + 'lr: 0.01, ' + '\n' + \
           'avg time per epoch: {:.0f}m {:.0f}s, '.format(epoch_time // 60, epoch_time % 60) + \
           'test accuracy: {:4f}, '.format(test_acc) + 'dataset: {}'.format(ResNet_args.dataset)
    plt.text(15, 5, text, size=10, bbox=props)


def summary_plot(acc_x_vals, train_acc_y_vals, val_acc_y_vals, test_acc_y_vals):  # train and val accuracy plot
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.plot(acc_x_vals, train_acc_y_vals)
    plt.plot(acc_x_vals, val_acc_y_vals)
    plt.plot(acc_x_vals, test_acc_y_vals)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Train Accuracy', 'Validation Accuracy', 'Test Accuracy'])


def final_plot():
    plt.savefig('hopefully_finally_4.svg')
    plt.show()


model = model.to(device)

criterion = nn.CrossEntropyLoss()

for mod in model.modules():
    if isinstance(mod, Rational):
        # print(mod)
        # mod.show()
        print(mod.numerator)

# Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=ResNet_args.learning_rate, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_val_test_model(model, criterion, optimizer, exp_lr_scheduler,
                             num_epochs=ResNet_args.training_number_of_epochs)
final_plot()

# torch.save(test, 'Saved_Models_wo_rationals/hopefully_finally_4.pth')


for mod in model.modules():
    if isinstance(mod, Rational):
        # print(mod)
        # mod.show()
        print(mod.numerator)

model.layer2.__getitem__(2).rational.show()
