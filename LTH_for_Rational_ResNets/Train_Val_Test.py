import time

import torch
from LTH_for_Rational_ResNets import argparser
from LTH_for_Rational_ResNets.Datasets import CIFAR10 as cifar10
from LTH_for_Rational_ResNets.Datasets import SVHN

LTH_args = argparser.get_arguments()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

if LTH_args.dataset == 'cifar10':
    trainset = cifar10.get_trainset()
    valset = cifar10.get_validationset()
    testset = cifar10.get_testset()
    trainloader = cifar10.get_trainloader(bs=LTH_args.batch_size)
    valloader = cifar10.get_valloader(bs=LTH_args.batch_size)
    testloader = cifar10.get_testloader(bs=LTH_args.batch_size)
    classes = cifar10.get_classes()
    num_classes = cifar10.get_num_classes()
    it_per_ep = cifar10.get_it_per_epoch(bs=LTH_args.batch_size)

elif LTH_args.dataset == 'SVHN':
    trainset = SVHN.get_trainset()
    valset = SVHN.get_validationset()
    testset = SVHN.get_testset()
    trainloader = SVHN.get_trainloader(bs=LTH_args.batch_size)
    valloader = SVHN.get_valloader(bs=LTH_args.batch_size)
    testloader = SVHN.get_testloader(bs=LTH_args.batch_size)
    classes = SVHN.get_classes()
    num_classes = SVHN.get_num_classes()
    it_per_ep = SVHN.get_it_per_epoch(bs=LTH_args.batch_size)

criterion = torch.nn.CrossEntropyLoss()


def train(model, optimizer, scheduler):
    """Train and validate a model"""
    epoch_loss = 0
    epoch_acc = 0
    step = 0
    best_acc = 0
    num_epochs = LTH_args.training_number_of_epochs
    for epoch in range(num_epochs):
        print('Training Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('*' * 20)

        since_epoch = time.time()
        best_acc = 0.0

        for phase in ['train', 'val']:
            count_loss = 0.0
            running_corrects = 0
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = trainloader
            else:
                model.eval()  # Set model to evaluation mode
                dataloader = valloader

            for it, data in enumerate(dataloader, 0):
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
                if phase == 'train':
                    step += 1
                    scheduler.step()

                count_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                epoch_loss = count_loss / len(trainset)
                epoch_acc = running_corrects.double() / len(trainset)

            if phase == 'val':
                epoch_loss = count_loss / len(valset)
                epoch_acc = running_corrects.double() / len(valset)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc

            time_elapsed_epoch = time.time() - since_epoch

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        print('Epoch finished in {:.0f}m {:.0f}s'.format(time_elapsed_epoch // 60, time_elapsed_epoch % 60))
    return best_acc


def train_til_val_acc(model, criterion, optimizer, scheduler):
    """Train a model until it reaches a certain validation accuracy. WARNING: might lead to endless loop"""
    validation_accuracy = 0
    number_epochs = 0
    while validation_accuracy < 0.93:
        print('Training Epoch {}'.format(number_epochs))
        print('*' * 10)
        number_epochs += 1
        for phase in ['train', 'val']:
            count_loss = 0.0
            count_corrects = 0
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = trainloader
            else:
                model.eval()  # Set model to evaluation mode
                dataloader = valloader

            for it, data in enumerate(dataloader, 0):

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
                if phase == 'train':
                    scheduler.step()

                count_loss += loss.item() * inputs.size(0)
                count_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                epoch_loss = count_loss / len(trainset)
                epoch_acc = count_corrects.double() / len(trainset)

            if phase == 'val':
                epoch_loss = count_loss / len(valset)
                epoch_acc = count_corrects.double() / len(valset)
                validation_accuracy = epoch_acc

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
    return model, validation_accuracy, number_epochs


def test(model):
    """Test a model on a given test set."""
    corrects = 0
    model = model.eval()

    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        corrects += torch.sum(preds == labels.data)

    test_accuracy = corrects.double() / len(testset)
    return test_accuracy.cpu().numpy()
