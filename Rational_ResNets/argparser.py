import argparse as arg

ResNet_arg_parser = arg.ArgumentParser()
ResNet_arg_parser.add_argument('-bs', '--batch_size', default=128, type=int)
ResNet_arg_parser.add_argument('-lr', '--learning_rate', default=0.03, type=float)
ResNet_arg_parser.add_argument('-m', '--model', default='univ_rational_resnet20', type=str,
                               choices=['univ_rational_resnet20', 'univ_rational_resnet14_A', 'univ_rational_resnet14_B', 'univ_rational_resnet8',
                                        'relu_resnet20', 'relu_resnet14_A', 'relu_resnet14_B', 'relu_resnet8',
                                        'mix_experts_resnet20', 'mix_experts_resnet14_A', 'mix_experts_resnet14_B', 'mix_experts_resnet8',
                                        'univ_rational_resnet18', 'univ_rational_resnet18_2_layers', 'univ_rational_resnet18_1_layer',
                                        'relu_resnet18', 'relu_resnet18_2_layers', 'relu_resnet18_1_layer',
                                        'mix_experts_resnet18', 'mix_experts_resnet18_2_layers', 'mix_experts_resnet18_1_layer',
                                        'select_1_expert_group_rational_resnet20'])
ResNet_arg_parser.add_argument('-ds', '--dataset', default='cifar10', type=str, choices=['cifar10', 'SVHN', 'ImageNet'])
ResNet_arg_parser.add_argument('-aug', '--augment_data', default=False, action='store_true')
ResNet_arg_parser.add_argument('-epochs', '--training_number_of_epochs', default=25, type=int)
ResNet_arg_parser.add_argument('-init_rationals', '--initialize_rationals',
                               type=str, nargs='+', default=['leaky_relu', 'gelu', 'swish', 'tanh', 'sigmoid'], choices=['leaky_relu', 'gelu', 'swish', 'tanh', 'sigmoid'],
                               help="Examples: -init_rationals leaky_relu gelu, -init_rationals tanh")
ResNet_arg_parser.add_argument('-wi', '--warmup_iterations', default=0, type=int)
ResNet_arg_parser.add_argument('--save_res_csv', default=True, action='store_true', help='Flag to save the results of the experiment as csv')
ResNet_arg_parser.add_argument('-seed', '--data_seeds', default=2, type=int)
run_all_groups = ResNet_arg_parser.add_mutually_exclusive_group()
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
ResNet_arg_parser.add_argument('--arch_for_run_all', default='CIFAR10', choices=['CIFAR10', 'ImageNet'])
ResNet_arg_parser.add_argument('--milestones', type=str, nargs='+', default=[10, 15, 20], help='Examples: --milestones 20 30, -milestones 5')
ResNet_arg_parser.add_argument('--hist', default=False, action='store_true', help='Flag to save histograms')
ResNet_arg_parser.add_argument('--clip_gradients', default=False, action='store_true', help='Flag to clip gradients')

"""'--model', 'univ_rational_resnet20', '--dataset', 'SVHN', '--warmup_iterations', '7167',
     '--iterative_pruning_epochs', '2', '--training_number_of_epochs', '10',
     '--stop_criteria', 'num_prune_epochs', '--save_res_csv', '--prune_shortcuts'   '--run_all_classic', '--arch_for_run_all', 'CIFAR10',"""


def get_argparser() -> arg.ArgumentParser:
    """
    Return the argument parser.

    Returns
    -------
    ResNet_arg_parser: arg.ArgumentParser
                       Argument Parser for all experiments with ResNets.
    """
    return ResNet_arg_parser


def get_arguments():
    resnet_args = ResNet_arg_parser.parse_args(['--model', 'mix_experts_resnet20', '--dataset', 'SVHN', '--training_number_of_epochs', '25', '--save_res_csv', '--data_seeds', '2',  '--learning_rate', '0.1', '--clip_gradients', '--augment_data', '--initialize_rationals', 'leaky_relu'])
    if resnet_args.arch_for_run_all == 'ImageNet' and resnet_args.run_all_two_BB:
        print('This option is not available for ResNet18.')
        exit()
    if (resnet_args.run_all_classic or resnet_args.run_all_two_BB or resnet_args.run_all_two_layers or resnet_args.run_all_one_layer) and not resnet_args.arch_for_run_all:
        print('Please choose architecture for command run_all.')
        exit()
    return resnet_args
