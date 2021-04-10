# Rationals for ResNets

## LTH_for_Rational_ResNets
``python3 LTH_for_Rational_ResNets/main.py``
  

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
  * Choices: rational_resnet20_cifar10, resnet20_cifar10, rational_resnet18_imagenet, resnet18_imagenet, multi_rational_resnet20_cifar10,
                                        pt 
    

* ```-ds```: Dataset
  * Default: cifar10
  * Choices: SVHN, cifar10
    

* ```-tnep```: Number of epochs for training
    * Default: 2

## Rational_ResNets
```python3 Rational_ResNets/main.py```

* ```-bs```: Batch size
    * Default: 128
    

* ```-lr```: Learning rate
  * Default: 0.01
    

* ```-m```: Model
  * Default: rational_resnet20_cifar10
  * Choices: rational_resnet20_cifar10, resnet20_cifar10, rational_resnet18_imagenet, resnet18_imagenet,
                                        pt, recurrent_rational_resnet20_cifar10, resnet110_cifar10,
                                        rational_resnet110_cifar10, select_2_expert_groups_rational_resnet
    

* ```-ds```: Dataset
  * Default: cifar10
  * Choices: SVHN, cifar10
    

* ```-epochs```: Number of epochs for training
    * Default: 2
  
* ```-num_rat```: Number of Rational Activation Functions per Vector (2 Vectors per BasicBlock)
    * Default: 1
  
* ```-init_rationals```: Different initializations for the Rational Activation Functions
    * Default: leaky_relu, gelu, swish, tanh, sigmoid
    * Choices: leaky_relu, gelu, swish, tanh, sigmoid