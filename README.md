# Rationals for ResNets

## LTH_for_Rational_ResNets
``python3 Lottery_Ticket_Hypothesis.py``
* ``-exp_name``: Name of the experiment
  

* ```-wi```: number of warm-up iterations
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
```python3 Train_Val_Test_2.py```

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
 