import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from LTH_for_Rational_ResNets import Lottery_Ticket_Hypothesis
from LTH_for_Rational_ResNets import plots
import argparser
from LTH_for_Rational_ResNets.Datasets import CIFAR10 as cifar10
from LTH_for_Rational_ResNets.Datasets import SVHN
from LTH_for_Rational_ResNets.LTH_Models import multi_variant_rational_resnet20_cifar10 as mvrrn20
from LTH_for_Rational_ResNets.LTH_Models import rational_resnet18_imagenet as rrn18
from LTH_for_Rational_ResNets.LTH_Models import rational_resnet20_cifar10 as rrn20
from LTH_for_Rational_ResNets.LTH_Models import resnet18_imagenet as rn18
from LTH_for_Rational_ResNets.LTH_Models import resnet20_cifar10 as rn20
from LTH_for_Rational_ResNets.LTH_Models import mv_select_rational_resnet20_cifar10 as mvselrrn20
from LTH_for_Rational_ResNets.Mask import make_initial_mask

LTH_arg_parser = argparser.get_argparser()
LTH_args = LTH_arg_parser.parse_args(['--model', 'rational_resnet20_cifar10', '--dataset', 'SVHN', '--warmup_iterations', '2000', '--iterative_pruning_epochs', '2', '--training_number_of_epochs', '15',
                                      '--stop_criteria', 'test_acc', '--experiment_name', 'rational_test'])

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
global checkpoint
global num_pruning_epochs

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

if LTH_args.dataset is 'cifar10':
    trainset = cifar10.get_trainset()
    valset = cifar10.get_validationset()
    testset = cifar10.get_testset()
    trainloader = cifar10.get_trainloader(bs=LTH_args.batch_size)
    valloader = cifar10.get_valloader(bs=LTH_args.batch_size)
    testloader = cifar10.get_testloader(bs=LTH_args.batch_size)
    classes = cifar10.get_classes()
    num_classes = cifar10.get_num_classes()

elif LTH_args.dataset is 'SVHN':
    trainset = SVHN.get_trainset()
    valset = SVHN.get_validationset()
    testset = SVHN.get_testset()
    trainloader = SVHN.get_trainloader(bs=LTH_args.batch_size)
    valloader = SVHN.get_valloader(bs=LTH_args.batch_size)
    testloader = SVHN.get_testloader(bs=LTH_args.batch_size)
    classes = SVHN.get_classes()
    num_classes = SVHN.get_num_classes()

if LTH_args.model is 'rational_resnet20_cifar10':
    model = rrn20.rational_resnet20()
    model_type = rrn20
elif LTH_args.model is 'resnet20_cifar10':
    model = rn20.resnet20()
    model_type = rn20
elif LTH_args.model is 'rational_resnet18_imagenet':
    model = rrn18.rational_resnet18()
    model_type = rrn18
elif LTH_args.model is 'resnet18_imagenet':
    model = rn18.resnet18()
    model_type = rn18
elif LTH_args.model is 'multi_variant_rational_resnet20_cifar10':
    model = mvrrn20.multi_variant_rational_resnet20()
    model_type = mvrrn20
elif LTH_args.model is 'multi_select_variant_rational_resnet20_cifar10':
    model = mvselrrn20.multi_select_variant_rational_resnet20()
    model_type = mvselrrn20

mask = make_initial_mask(model)
mask = mask.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LTH_args.learning_rate, momentum=0.9, weight_decay=0.0001)


num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)
if LTH_args.warmup_iterations is not 0:
    exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda it: min(1.0, it / LTH_args.warmup_iterations))
else:
    exp_lr_scheduler = lr_scheduler.LambdaLR(optimizer, lambda it: 1)

if LTH_args.stop_criteria is 'test_acc':
    num_pruning_epochs = Lottery_Ticket_Hypothesis.iterative_pruning_by_test_acc(model, mask, LTH_args.test_accuracy_threshold, model_type=model_type, optimizer=optimizer, criterion=criterion, exp_lr_scheduler=exp_lr_scheduler,
                                                                                 testset=testset, testloader=testloader, trainset=trainset, trainloader=trainloader, valloader=valloader, valset=valset,
                                                                                 pruning_percentage=LTH_args.pruning_percentage, training_number_of_epochs=LTH_args.training_number_of_epochs)
elif LTH_args.stop_criteria is 'num_prune_epochs':
    Lottery_Ticket_Hypothesis.iterative_pruning_by_num(model, mask, LTH_args.iterative_pruning_epochs, model_type=model_type, optimizer=optimizer, criterion=criterion, exp_lr_scheduler=exp_lr_scheduler, testset=testset,
                                                       testloader=testloader, trainset=trainset, trainloader=trainloader, valloader=valloader, valset=valset, pruning_percentage=LTH_args.pruning_percentage,
                                                       training_number_of_epochs=LTH_args.training_number_of_epochs)
    num_pruning_epochs = LTH_args.iterative_pruning_epochs
elif LTH_args.stop_criteria is 'one_shot':
    Lottery_Ticket_Hypothesis.one_shot_pruning(model, optimizer=optimizer, criterion=criterion, exp_lr_scheduler=exp_lr_scheduler, testset=testset,
                                               testloader=testloader, trainset=trainset, trainloader=trainloader, valloader=valloader, valset=valset, prune_mask=mask, model_type=model_type, pruning_percentage=LTH_args.pruning_percentage,
                                               training_number_of_epochs=LTH_args.training_number_of_epochs)
    num_pruning_epochs = 1

plots.final_plot_LTH(LTH_args.model, LTH_args.dataset, LTH_args.batch_size, num_pruning_epochs, LTH_args.training_number_of_epochs, LTH_args.learning_rate, LTH_args.pruning_percentage, LTH_args.warmup_iterations, LTH_args.experiment_name)

