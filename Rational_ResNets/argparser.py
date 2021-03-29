import argparse as arg

ResNet_arg_parser = arg.ArgumentParser()
ResNet_arg_parser.add_argument('-exp_name', '--experiment_name', type=str, required=True)
ResNet_arg_parser.add_argument('-bs', '--batch_size', default=128, type=int)
ResNet_arg_parser.add_argument('-lr', '--learning_rate', default=0.01, type=float)
ResNet_arg_parser.add_argument('-m', '--model', default='rational_resnet20_cifar10', type=str,
                               choices=['rational_resnet20_cifar10', 'resnet20_cifar10', 'rational_resnet18_imagenet', 'resnet18_imagenet', 'multi_rational_resnet20_cifar10',
                                        'pt'])  # pt is the original ResNet18 model from Pytorch with Rationals
ResNet_arg_parser.add_argument('-ds', '--dataset', default='cifar10', type=str, choices=['cifar10', 'SVHN'])
ResNet_arg_parser.add_argument('-aug', '--augment_data', default=False, type=bool)
ResNet_arg_parser.add_argument('-tnep', '--training_number_of_epochs', default=25, type=int)


def make_argparser():
    return ResNet_arg_parser
