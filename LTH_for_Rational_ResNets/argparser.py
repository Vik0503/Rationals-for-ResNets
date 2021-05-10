import argparse as arg

LTH_arg_parser = arg.ArgumentParser()
LTH_arg_parser.add_argument('-bs', '--batch_size', default=128, type=int)
LTH_arg_parser.add_argument('-lr', '--learning_rate', default=0.03, type=float)
LTH_arg_parser.add_argument('-m', '--model', default='univ_rational_resnet20', type=str, choices=['univ_rational_resnet20', 'univ_rational_resnet14_A', 'univ_rational_resnet14_B', 'univ_rational_resnet8',
                                                                                                  'relu_resnet20', 'relu_resnet14_A', 'relu_resnet14_B', 'relu_resnet8',
                                                                                                  'mix_experts_resnet20', 'mix_experts_resnet14_A', 'mix_experts_resnet14_B', 'mix_experts_resnet8',
                                                                                                  'univ_rational_resnet18', 'univ_rational_resnet18_2_layers', 'univ_rational_resnet18_1_layer',
                                                                                                  'relu_resnet18', 'relu_resnet18_2_layers', 'relu_resnet18_1_layer',
                                                                                                  'mix_experts_resnet18', 'mix_experts_resnet18_2_layers', 'mix_experts_resnet18_1_layer',
                                                                                                  'select_1_expert_group_rational_resnet20'])
LTH_arg_parser.add_argument('-ds', '--dataset', default='SVHN', type=str, choices=['cifar10', 'SVHN', 'ImageNet'])
LTH_arg_parser.add_argument('-wi', '--warmup_iterations', default=0, type=int)
LTH_arg_parser.add_argument('-tnep', '--training_number_of_epochs', default=25, type=int)
LTH_arg_parser.add_argument('-pp', '--pruning_percentage', default=20.0, type=float)
stop_group = LTH_arg_parser.add_mutually_exclusive_group()
stop_group.add_argument('-ipe', '--iterative_pruning_epochs', default=15, type=int)
LTH_arg_parser.add_argument('-stop', '--stop_criteria', default='test_acc', type=str, choices=['test_acc', 'num_prune_epochs', 'one_shot'])
stop_group.add_argument('-test_acc', '--test_accuracy_threshold', default=0.9, type=float)
LTH_arg_parser.add_argument('-init_rationals', '--initialize_rationals',
                            type=str, nargs='+', default=['leaky_relu', 'gelu', 'swish', 'tanh', 'sigmoid'], choices=['leaky_relu', 'gelu', 'swish', 'tanh', 'sigmoid', 'relu'],
                            help='Examples: -init_rationals leaky_relu gelu, -init_rationals tanh')
run_all_groups = LTH_arg_parser.add_mutually_exclusive_group()
run_all_groups.add_argument('--run_all_classic', default=False, action='store_true',
                            help="Flag to perform all three experiments `original`, `univariate rational` and `mixture of experts` in a sequence and plot the results in one graph for further comparison.")
run_all_groups.add_argument('--run_all_two_BB', default=False, action='store_true',
                            help="Flag to perform all three experiments `original`, `univariate rational` and `mixture of experts` with a smaller architecture (only two BasicBlocks)"
                                 " in a sequence and plot the results in one graph for further comparison.")
run_all_groups.add_argument('--run_all_two_layers', default=False, action='store_true',
                            help="Flag to perform all three experiments `original`, `univariate rational` and `mixture of experts` with a smaller architecture (only two full layers) "
                                 "in a sequence and plot the results in one graph for further comparison.")
run_all_groups.add_argument('--run_all_one_layer', default=False, action='store_true',
                            help="Flag to perform all three experiments `original`, `univariate rational` and `mixture of experts` with a smaller architecture (only one full layer)"
                                 "in a sequence and plot the results in one graph for further comparison.")
LTH_arg_parser.add_argument('--prune_shortcuts', default=False, action='store_true', help='Flag to prune shortcuts.')
LTH_arg_parser.add_argument('--save_res_csv', default=True, action='store_true', help='Flag to save the results of the experiment as csv')
LTH_arg_parser.add_argument('-seed', '--data_seeds', default=2, type=int)
LTH_arg_parser.add_argument('--arch_for_run_all', default='CIFAR10', choices=['CIFAR10', 'ImageNet'])
LTH_arg_parser.add_argument('--milestones', type=str, nargs='+', default=[10, 15, 20],
                            help='Examples: --milestones 20 30, -milestones 5')
LTH_arg_parser.add_argument('--hist', default=False, action='store_true', help='Flag to save histograms')


def get_argparser():
    """
    Return the argument parser.

    Returns
    -------
    ResNet_arg_parser: arg.ArgumentParser
                       Argument Parser for all experiments with ResNets.
    """
    return LTH_arg_parser


"""'--model', 'univ_rational_resnet20', '--dataset', 'SVHN', '--warmup_iterations', '7167',
     '--iterative_pruning_epochs', '2', '--training_number_of_epochs', '2',
     '--stop_criteria', 'num_prune_epochs', '--save_res_csv', '--prune_shortcuts'"""


def get_arguments():
    """
    Return parsed arguments.
    """

    """['--model', 'mix_experts_resnet18', '--dataset', 'SVHN', '--warmup_iterations', '1',
                                          '--iterative_pruning_epochs', '1', '--training_number_of_epochs', '5',
                                          '--stop_criteria', 'num_prune_epochs', '--milestones', '2', '3']"""

    LTH_args = LTH_arg_parser.parse_args(['--dataset', 'SVHN', '--warmup_iterations', '7167', '--training_number_of_epochs', '25',
                                          '--stop_criteria', 'test_acc', '--save_res_csv', '--run_all_classic', '--arch_for_run_all', 'CIFAR10'])
    if LTH_args.arch_for_run_all == 'ImageNet' and LTH_args.run_all_two_BB:
        print('This option is not available for ResNet18.')
        exit()
    if (LTH_args.run_all_classic or LTH_args.run_all_two_BB or LTH_args.run_all_two_layers or LTH_args.run_all_one_layer) and not LTH_args.arch_for_run_all:
        print('Please choose architecture for command run_all.')
        exit()
    return LTH_args
