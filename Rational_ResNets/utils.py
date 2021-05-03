import csv

import torch
import yaml
from torch import optim
from torch.optim import lr_scheduler

import argparser
from datetime import datetime

args = argparser.get_arguments()


def make_csv(model, epoch, train_acc: list, val_acc: list, test_acc: list):
    time_stamp = datetime.now()
    PATH = 'CSV/{}'.format(model) + '/{}'.format(time_stamp) + '.csv'
    with open(PATH, 'w', newline='') as csvfile:
        fieldnames = ['Epoch', 'Train Accuracy', 'Validation Accuracy', 'Test Accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, dialect='excel')

        writer.writeheader()
        for i in range(len(train_acc)):
            writer.writerow({'Epoch': epoch[i], 'Train Accuracy': train_acc[i].cpu().numpy(), 'Validation Accuracy': val_acc[i].cpu().numpy(), 'Test Accuracy': test_acc[i].cpu().numpy()})

    return PATH


def get_scheduler_optimizer(num_warmup_it: int, lr: float, model, it_per_ep: int):
    """
    Return scheduler with custom milestones and optimizer

    Parameters
    ----------
    num_warmup_it: int
                   The number of warmup iterations.
    lr: float
        The learning rate.
    model
    it_per_ep: int
               The number of iterations per epoch

    Returns
    -------
    torch.optim.lr_scheduler.LambdaLR
    optimizer: torch.optim.SGD
    """
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    milestones = args.milestones
    milestones = list(map(int, milestones))
    milestones.sort()
    print(milestones)

    def lr_lambda(it):
        if it < num_warmup_it:
            if it % 430 == 0:
                print('Warmup')
            return min(1.0, it / num_warmup_it)
        else:
            for m in range(len(milestones)):
                if it < milestones[m] * it_per_ep:
                    if it % 430 == 0:
                        print('Milestone {}: {}'.format(m, 1 * 10 ** -m))
                    return 1 * 10 ** -m
            if it % 430 == 0:
                print('Milestone {}: {}'.format(len(milestones), 1 * 10 ** -(len(milestones))))
            return 1 * 10 ** -(len(milestones))

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda), optimizer


def make_yaml(models: list, csv=None):  # TODO: add Rational Init + plot PATH
    resnet_args = argparser.get_arguments()
    time_stamp = datetime.now()
    yaml_data = [{'Date': [time_stamp]}, {'Model(s)': models}, {'Dataset': [resnet_args.dataset]}, {'Batch Size': [resnet_args.batch_size]},
                 {'Learning Rate': [resnet_args.learning_rate]}, {'Epochs': [resnet_args.training_number_of_epochs]}, {'Warm-Up Iterations': [resnet_args.warmup_iterations]}]

    if resnet_args.save_res_csv:
        yaml_data.append({'CSV File(s)': [csv]})

    PATH = 'YAML/{}'.format(time_stamp) + '.yaml'
    with open(PATH, 'w') as file:
        documents = yaml.dump(yaml_data, file)


def initialize_alpha(b: int = 4) -> torch.Tensor:
    """Initialize the vector alpha.

    Parameters
    ----------
    b : int
        The length of the vector alpha.

    Returns
    -------
    alpha : torch.Tensor
            The tensor with initial values for alpha.
    """
    alpha = torch.rand(b, requires_grad=True)
    alpha = alpha / alpha.sum()
    return alpha