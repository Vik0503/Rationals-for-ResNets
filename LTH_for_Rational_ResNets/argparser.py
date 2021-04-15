import argparse as arg

LTH_arg_parser = arg.ArgumentParser()
LTH_arg_parser.add_argument('-bs', '--batch_size', default=128, type=int)
LTH_arg_parser.add_argument('-lr', '--learning_rate', default=0.03, type=float)
LTH_arg_parser.add_argument('-m', '--model', default='rational_resnet20_cifar10', type=str, choices=['rational_resnet20_cifar10', 'resnet20_cifar10', 'rational_resnet18_imagenet',
                                                                                                     'resnet18_imagenet', 'select_2_expert_groups_rational_resnet20', 'select_1_expert_group_rational_resnet20'])
LTH_arg_parser.add_argument('-ds', '--dataset', default='cifar10', type=str, choices=['cifar10', 'SVHN'])
LTH_arg_parser.add_argument('-wi', '--warmup_iterations', default=0, type=int)
LTH_arg_parser.add_argument('-tnep', '--training_number_of_epochs', default=2, type=int)
LTH_arg_parser.add_argument('-pp', '--pruning_percentage', default=20.0, type=float)
LTH_arg_parser.add_argument('-ipe', '--iterative_pruning_epochs', default=15, type=int)
LTH_arg_parser.add_argument('-stop', '--stop_criteria', default='test_acc', type=str, choices=['test_acc', 'num_prune_epochs', 'one_shot'])
LTH_arg_parser.add_argument('-test_acc', '--test_accuracy_threshold', default=0.89, type=float)
LTH_arg_parser.add_argument('-init_rationals', '--initialize_rationals',
                            type=str, nargs='+', default=['leaky_relu', 'gelu', 'swish', 'tanh', 'sigmoid'], choices=['leaky_relu', 'gelu', 'swish', 'tanh', 'sigmoid'],
                            help='Examples: -init_rationals leaky_relu gelu, -init_rationals tanh')
LTH_arg_parser.add_argument('--run_all', default=False, action='store_true',
                            help="Flag to perform all three experiments `original`, `univariate rational` and `mixture of experts` in a sequence and plot the results in one graph for further comparison.")
LTH_arg_parser.add_argument('--prune_shortcuts', default=False, action='store_true', help='Flag to prune shortcuts.')


def get_argparser():
    """
    Return the argument parser.

    Returns
    -------
    ResNet_arg_parser: arg.ArgumentParser
                       Argument Parser for all experiments with ResNets.
    """
    return LTH_arg_parser


LTH_args = LTH_arg_parser.parse_args(
    ['--model', 'select_2_expert_groups_rational_resnet20', '--dataset', 'SVHN', '--warmup_iterations', '7167',
     '--test_accuracy_threshold', '0.90', '--training_number_of_epochs', '25',
     '--stop_criteria', 'test_acc', '--run_all'])


def get_arguments():
    return LTH_args
