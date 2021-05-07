# Rationals for ResNets
## CIFAR10 ResNet Types
* ResNet20 as described in: 
Deep Residual Learning for Image Recognition
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
https://arxiv.org/abs/1512.03385
* ResNet14 A
  * 3 layers
  * 2 BasicBlocks per layer
* ResNet14 B
  * 2 layers
  * 3 BasicBlocks per layer
* ResNet8
  * 1 layer with 3 BasicBlocks
## LTH_for_Rational_ResNets
``python3 LTH_for_Rational_ResNets/main.py``
  
You can either choose one model (`-m`) or run an experiment with all three models of one model type. 
* ```-wi```: Number of warm-up iterations
  * Default: 0
    

* ```-pp```: Pruning percentage per pruning iteration
  * Default: 20.0
    

* ```-stop```: Stopping criterion
  * Default: test_acc
  * Choices: test_acc, num_prune_epochs, one_shot
    

* ```-test_acc```: Threshold for the test accuracy, when the chosen stopping criterion is test_acc
  

* ```-ipe```: Number of iterative pruning epochs, when the chosen stopping criterion is num_prune_epochs
  * Default: 15
    

* ```-bs```: Batch size
    * Default: 128
    

* ```-lr```: Learning rate
  * Default: 0.01
    

* ```-m```: Model
  * Default: rational_resnet20_cifar10
  * Choices: univ_rational_resnet20, univ_rational_resnet14_A, univ_rational_resnet14_B, univ_rational_resnet8,
                                    relu_resnet20, relu_resnet14_A, relu_resnet14_B, relu_resnet8,
                                        mix_experts_resnet20, mix_experts_resnet14_A, mix_experts_resnet14_B, mix_experts_resnet8,
                                        univ_rational_resnet18, univ_rational_resnet18_2_layers, univ_rational_resnet18_1_layer,
                                        relu_resnet18, relu_resnet18_2_layers, relu_resnet18_1_layer,
                                        mix_experts_resnet18, mix_experts_resnet18_2_layers, mix_experts_resnet18_1_layer,
                                        select_1_expert_group_rational_resnet20
    

* ```-ds```: Dataset
  * Default: SVHN
  * Choices: SVHN, cifar10
    

* ```-tnep```: Number of epochs for training
    * Default: 25


* `--milestones`: Epochs in training, where the leraning rate is divided by 10
    * Default: 10, 15, 20


* ```-init_rationals```: Different initializations for the Rational Activation Functions
    * Default: leaky_relu, gelu, swish, tanh, sigmoid
    * Choices: leaky_relu, gelu, swish, tanh, sigmoid, relu

* `--hist`: Flag to show histograms in activation function plots
    * Default: False
* `--arch_for_run_all`: Different architectures for the ResNets
    * Default: CIFAR10
    * Choices: CIFAR10, ImageNet

* `--run_all_two_BB`: Flag to perform all three experiments `original`, `univariate rational` and `mixture of experts` as `ResNet14 A` in a sequence.

* `--run_all_two_layers`: Flag to perform all three experiments `original`, `univariate rational` and `mixture of experts` as `ResNet14 B` in a sequence.
* `--run_all_one_layer`:`Flag to perform all three experiments `original`, `univariate rational` and `mixture of experts` as `ResNet8` in a sequence.
## Rational_ResNets
```python3 Rational_ResNets/main.py```

* ```-bs```: Batch size
    * Default: 128
    

* ```-lr```: Learning rate
  * Default: 0.01
    

* ```-m```: Model
  * Default: rational_resnet20_cifar10
  * Choices: univ_rational_resnet20, univ_rational_resnet14_A, univ_rational_resnet14_B, univ_rational_resnet8,
                                    relu_resnet20, relu_resnet14_A, relu_resnet14_B, relu_resnet8,
                                        mix_experts_resnet20, mix_experts_resnet14_A, mix_experts_resnet14_B, mix_experts_resnet8,
                                        univ_rational_resnet18, univ_rational_resnet18_2_layers, univ_rational_resnet18_1_layer,
                                        relu_resnet18, relu_resnet18_2_layers, relu_resnet18_1_layer,
                                        mix_experts_resnet18, mix_experts_resnet18_2_layers, mix_experts_resnet18_1_layer,
                                        select_1_expert_group_rational_resnet20
    

* ```-ds```: Dataset
  * Default: cifar10
  * Choices: SVHN, cifar10
    

* ```-epochs```: Number of epochs for training
    * Default: 25
  
* ```-init_rationals```: Different initializations for the Rational Activation Functions
    * Default: leaky_relu, gelu, swish, tanh, sigmoid
    * Choices: leaky_relu, gelu, swish, tanh, sigmoid
  
* `-seed`: Data seed for reproducability
    * Default: 42

* `--save_res_csv`: Flag to save results as CSV files
    * Default: True

* `--hist`: Flag to show histograms in activation function plots
    * Default: False
* `--arch_for_run_all`: Different architectures for the ResNets
    * Default: CIFAR10
    * Choices: CIFAR10, ImageNet

* `--run_all_two_BB`: Flag to perform all three experiments `original`, `univariate rational` and `mixture of experts` as `ResNet14 A` in a sequence.

* `--run_all_two_layers`: Flag to perform all three experiments `original`, `univariate rational` and `mixture of experts` as `ResNet14 B` in a sequence.
* `--run_all_one_layer`: Flag to perform all three experiments `original`, `univariate rational` and `mixture of experts` as `ResNet8` in a sequence.