# Rationals for ResNets

## LTH_for_Rational_ResNets
run Lottery_Ticket_Hypothesis.py

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
 