import argparse as arg


ResNet_arg_parser = arg.ArgumentParser()
ResNet_arg_parser.add_argument('-bs', '--batch_size', default=128, type=int)
ResNet_arg_parser.add_argument('-lr', '--learning_rate', default=0.01, type=float)
ResNet_arg_parser.add_argument('-m', '--model', default='rational_resnet20_cifar10', type=str,
                               choices=['rational_resnet20_cifar10', 'resnet20_cifar10', 'rational_resnet18_imagenet', 'resnet18_imagenet', 'multi_rational_resnet20_cifar10',
                                        'pt', 'rational_2_resnet20_cifar10', 'multi_select_variant_rational_resnet20_cifar10', 'test_mv_resnet20'])  # pt is the original ResNet18 model from Pytorch with Rationals
ResNet_arg_parser.add_argument('-ds', '--dataset', default='cifar10', type=str, choices=['cifar10', 'SVHN'])
ResNet_arg_parser.add_argument('-aug', '--augment_data', default=False, type=bool)
ResNet_arg_parser.add_argument('-tnep', '--training_number_of_epochs', default=25, type=int)
ResNet_arg_parser.add_argument('-num_rat', '--number_of_rationals_per_vector', default=1, type=int)


def get_argparser() -> arg.ArgumentParser:
    """
    Return the argument parser.

    Returns
    -------
    ResNet_arg_parser: arg.ArgumentParser
                       Argument Parser for all experiments with ResNets.
    """
    return ResNet_arg_parser
