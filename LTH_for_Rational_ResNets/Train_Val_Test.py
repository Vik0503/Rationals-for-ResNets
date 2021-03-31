import time

import torch

if torch.cuda.is_available():
    device = 'cuda'
    print(device)
else:
    device = 'cpu'


def train(model, criterion, optimizer, scheduler, num_epochs, trainset, valset, trainloader, valloader):
    """Train and validate a model"""
    step = 0
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
    return model, best_acc, step


def train_til_val_acc(model, criterion, optimizer, scheduler, trainset, valset, trainloader, valloader):
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


def test(model, testset, testloader):
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


