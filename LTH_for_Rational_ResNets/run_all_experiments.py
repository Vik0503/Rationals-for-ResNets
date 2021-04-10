import inspect
import os
import sys

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import torch
from torch import nn, optim
import numpy as np
from torch.optim import lr_scheduler

from LTH_for_Rational_ResNets import Lottery_Ticket_Hypothesis
from LTH_for_Rational_ResNets import plots
from LTH_for_Rational_ResNets import argparser
from LTH_for_Rational_ResNets.Datasets import CIFAR10 as cifar10
from LTH_for_Rational_ResNets.Datasets import SVHN
from LTH_for_Rational_ResNets.LTH_Models import rational_resnet18_imagenet as rrn18, resnet20_cifar10 as rn20
from LTH_for_Rational_ResNets.LTH_Models import rational_resnet20_cifar10 as rrn20
from LTH_for_Rational_ResNets.LTH_Models import resnet18_imagenet as rn18
from LTH_for_Rational_ResNets.LTH_Models import select_2_expert_groups_rational_resnet as sel2exp
from LTH_for_Rational_ResNets.LTH_Models import select_1_expert_group_rational_resnet as sel1exp
from LTH_for_Rational_ResNets.Mask import make_initial_mask

from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

plt.style.use(["science", "grid"])
matplotlib.rcParams.update({
    "font.family": "serif",
    "text.usetex": False,
})


def get_scheduler():
    lambdas = [lambda it: 1.0, lambda it: 0.1 ** (10 * it_per_ep), lambda it: 0.1 ** (15 * it_per_ep), lambda it: 0.1 ** (20 * it_per_ep)]
    warmup_iterations = 7167
    lambdas.append(lambda it: min(1.0, it / warmup_iterations))

    return lr_scheduler.LambdaLR(optimizer, lambda it: np.product([l(it) for l in lambdas]))


trainset = SVHN.get_trainset()
valset = SVHN.get_validationset()
testset = SVHN.get_testset()
trainloader = SVHN.get_trainloader(bs=128)
valloader = SVHN.get_valloader(bs=128)
testloader = SVHN.get_testloader(bs=128)
classes = SVHN.get_classes()
num_classes = SVHN.get_num_classes()
it_per_ep = SVHN.get_it_per_epoch(bs=128)
pruning_percentage = 20
training_epochs = 25

criterion = nn.CrossEntropyLoss()
dataset = 'SVHN'

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.xlabel('Percent of Pruned Weights')
plt.ylabel('Test Accuracy in Percent')

model_1 = rrn20.rational_resnet20()
model_type_1 = rrn20

mask_1 = make_initial_mask(model_1)
mask_1 = mask_1.cuda()

optimizer = optim.SGD(model_1.parameters(), lr=0.03, momentum=0.9, weight_decay=0.0001)
exp_lr_scheduler = get_scheduler()

num_ftrs = model_1.fc.in_features
model_1.fc = nn.Linear(num_ftrs, num_classes)
model_1 = model_1.to(device)

num_pruning_epochs, test_accuracies, sparsities = Lottery_Ticket_Hypothesis.iterative_pruning_by_test_acc(model_1, mask_1,
                                                                                                          0.89,
                                                                                                          model_type=model_type_1,
                                                                                                          optimizer=optimizer,
                                                                                                          criterion=criterion,
                                                                                                          exp_lr_scheduler=exp_lr_scheduler,
                                                                                                          testset=testset, testloader=testloader,
                                                                                                          trainset=trainset,
                                                                                                          trainloader=trainloader,
                                                                                                          valloader=valloader, valset=valset,
                                                                                                          pruning_percentage=pruning_percentage,
                                                                                                          training_number_of_epochs=training_epochs)
"""test_accuracies, sparsities = Lottery_Ticket_Hypothesis.iterative_pruning_by_num(model_1, mask_1, 2,
                                                                                 model_type=model_type_1, optimizer=optimizer, criterion=criterion,
                                                                                 exp_lr_scheduler=exp_lr_scheduler, testset=testset,
                                                                                 testloader=testloader, trainset=trainset,
                                                                                 trainloader=trainloader, valloader=valloader, valset=valset,
                                                                                 pruning_percentage=pruning_percentage,
                                                                                 training_number_of_epochs=training_epochs)"""
plt.plot(sparsities, test_accuracies)

model_2 = rn20.resnet20()
model_type_2 = rn20

mask_2 = make_initial_mask(model_2)
mask_2 = mask_2.cuda()

optimizer = optim.SGD(model_2.parameters(), lr=0.03, momentum=0.9, weight_decay=0.0001)
exp_lr_scheduler = get_scheduler()
num_ftrs = model_2.fc.in_features
model_2.fc = nn.Linear(num_ftrs, num_classes)
model_2 = model_2.to(device)

num_pruning_epochs_2, test_accuracies_2, sparsities_2 = Lottery_Ticket_Hypothesis.iterative_pruning_by_test_acc(model_2, mask_2,
                                                                                                                0.89,
                                                                                                                model_type=model_type_2,
                                                                                                                optimizer=optimizer,
                                                                                                                criterion=criterion,
                                                                                                                exp_lr_scheduler=exp_lr_scheduler,
                                                                                                                testset=testset, testloader=testloader,
                                                                                                                trainset=trainset,
                                                                                                                trainloader=trainloader,
                                                                                                                valloader=valloader, valset=valset,
                                                                                                                pruning_percentage=pruning_percentage,
                                                                                                                training_number_of_epochs=training_epochs)

"""test_accuracies_2, sparsities_2 = Lottery_Ticket_Hypothesis.iterative_pruning_by_num(model_2, mask_2, 2,
                                                                                     model_type=model_type_2, optimizer=optimizer, criterion=criterion,
                                                                                     exp_lr_scheduler=exp_lr_scheduler, testset=testset,
                                                                                     testloader=testloader, trainset=trainset,
                                                                                     trainloader=trainloader, valloader=valloader, valset=valset,
                                                                                     pruning_percentage=pruning_percentage,
                                                                                     training_number_of_epochs=training_epochs)"""
plt.plot(sparsities_2, test_accuracies_2)

model_3 = sel2exp.select_2_expert_groups_rational_resnet20(rational_inits=['leaky_relu', 'gelu', 'swish', 'tanh', 'sigmoid'], num_rationals=5)
model_type_3 = sel2exp

mask_3 = make_initial_mask(model_3)
mask_3 = mask_3.cuda()

optimizer = optim.SGD(model_3.parameters(), lr=0.03, momentum=0.9, weight_decay=0.0001)
exp_lr_scheduler = get_scheduler()
num_ftrs = model_3.fc.in_features
model_3.fc = nn.Linear(num_ftrs, num_classes)
model_3 = model_3.to(device)

num_pruning_epochs_3, test_accuracies_3, sparsities_3 = Lottery_Ticket_Hypothesis.iterative_pruning_by_test_acc(model_3, mask_3,
                                                                                                                0.89,
                                                                                                                model_type=model_type_3,
                                                                                                                optimizer=optimizer,
                                                                                                                criterion=criterion,
                                                                                                                exp_lr_scheduler=exp_lr_scheduler,
                                                                                                                testset=testset, testloader=testloader,
                                                                                                                trainset=trainset,
                                                                                                                trainloader=trainloader,
                                                                                                                valloader=valloader, valset=valset,
                                                                                                                pruning_percentage=pruning_percentage,
                                                                                                                training_number_of_epochs=training_epochs)
"""test_accuracies_3, sparsities_3 = Lottery_Ticket_Hypothesis.iterative_pruning_by_num(model_3, mask_3, 2,
                                                                                     model_type=model_type_3, optimizer=optimizer, criterion=criterion,
                                                                                     exp_lr_scheduler=exp_lr_scheduler, testset=testset,
                                                                                     testloader=testloader, trainset=trainset,
                                                                                     trainloader=trainloader, valloader=valloader, valset=valset,
                                                                                     pruning_percentage=pruning_percentage,
                                                                                     training_number_of_epochs=training_epochs)"""

plt.plot(sparsities_3, test_accuracies_3)
plt.legend(['ResNet20 with univ. Rat.', 'ResNet20 Original', 'ResNet20 with a mixture of 5 univ. Rat.'])

props = dict(boxstyle='round', facecolor='grey', alpha=0.5)
text = 'dataset: {}, '.format(dataset) + 'batch size: {}, '.format(128) + '\n' + '{} training epochs per pruning epoch, '.format(training_epochs) + '\n' + \
       'learning rate: {}, '.format(0.03) + '{}% pruning per epoch, '.format(pruning_percentage) + '\n' + '{} warm-up iterations'.format(7167)

plt.figtext(0.525, 0.5, text, bbox=props, size=9)

time_stamp = datetime.now()
PATH = './Results/LTH_all_models' + '/' + '{}'.format(time_stamp) + '_' + '{}'.format(dataset) + '.svg'
plt.savefig(PATH)
plt.show()
