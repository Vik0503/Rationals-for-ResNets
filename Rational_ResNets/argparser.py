import argparse as arg

ResNet_arg_parser = arg.ArgumentParser()
ResNet_arg_parser.add_argument('-bs', '--batch_size', default=128, type=int)
ResNet_arg_parser.add_argument('-lr', '--learning_rate', default=0.01, type=float)
ResNet_arg_parser.add_argument('-m', '--model', default='rational_resnet20_cifar10', type=str,
                               choices=['rational_resnet20_cifar10', 'resnet20_cifar10', 'rational_resnet18_imagenet', 'resnet18_imagenet',
                                        'pt', 'recurrent_rational_resnet20_cifar10', 'resnet110_cifar10',
                                        'rational_resnet110_cifar10', 'select_2_expert_groups_rational_resnet20',
                                        'select_1_expert_group_rational_resnet20'])  # pt is the original ResNet18 model from Pytorch with Rationals TODO: bigger models
ResNet_arg_parser.add_argument('-ds', '--dataset', default='cifar10', type=str, choices=['cifar10', 'SVHN'])
ResNet_arg_parser.add_argument('-aug', '--augment_data', default=False, type=bool)
ResNet_arg_parser.add_argument('-epochs', '--training_number_of_epochs', default=25, type=int)
ResNet_arg_parser.add_argument('-num_rat', '--number_of_rationals_per_vector', default=1, type=int)
ResNet_arg_parser.add_argument('-init_rationals', '--initialize_rationals',
                               type=str, nargs='+', default=['leaky_relu', 'gelu', 'swish', 'tanh', 'sigmoid'], choices=['leaky_relu', 'gelu', 'swish', 'tanh', 'sigmoid'],
                               help="Examples: -init_rationals leaky_relu gelu, -init_rationals tanh")
ResNet_arg_parser.add_argument("--train-all", default=False, action='store_true',
                               help="Flag to perform all three experiments `original`, `univariate rational` and `mixture of experts` in a sequence and plot the results in one graph for further comparison.")


def get_argparser() -> arg.ArgumentParser:
    """
    Return the argument parser.

    Returns
    -------
    ResNet_arg_parser: arg.ArgumentParser
                       Argument Parser for all experiments with ResNets.
    """
    return ResNet_arg_parser
