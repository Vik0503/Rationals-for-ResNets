import argparse as arg

LTH_arg_parser = arg.ArgumentParser()
LTH_arg_parser.add_argument('-exp_name', 'experiment_name', type=str, required=True)
LTH_arg_parser.add_argument('-bs', '--batch_size', default=128, type=int)
LTH_arg_parser.add_argument('-lr', '--learning_rate', default=0.03, type=float)
LTH_arg_parser.add_argument('-m', '--model', default='rational_resnet20_cifar10', type=str, choices=['rational_resnet20_cifar10', 'resnet20_cifar10', 'rational_resnet18_imagenet',
                                                                                                     'resnet18_imagenet'])  # , 'multi_variant_rational_resnet20_cifar10'
LTH_arg_parser.add_argument('-ds', '--dataset', default='cifar10', type=str, choices=['cifar10', 'SVHN'])
LTH_arg_parser.add_argument('-wi', '--warmup_iterations', default=0, type=int)
LTH_arg_parser.add_argument('-tnep', '--training_number_of_epochs', default=2, type=int)
LTH_arg_parser.add_argument('-pp', '--pruning_percentage', default=20.0, type=float)
LTH_arg_parser.add_argument('-ipe', '--iterative_pruning_epochs', default=15, type=int)
LTH_arg_parser.add_argument('-stop', '--stop_criteria', default='test_acc', type=str, choices=['test_acc', 'num_prune_epochs', 'one_shot'])
LTH_arg_parser.add_argument('-test_acc', '--test_accuracy_threshold', default=0.89, type=float)


def get_argparser():
    return LTH_arg_parser
