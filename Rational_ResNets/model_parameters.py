from Rational_ResNets.ResNet_Models import univ_rational_resnet_imagenet as univ_rat_imagenet, univ_rational_resnet_cifar10 as univ_rat_cifar
from Rational_ResNets.ResNet_Models import relu_resnet_imagenet as relu_imagenet, relu_resnet_cifar10 as relu_cifar
from Rational_ResNets.ResNet_Models import mix_experts_resnet_cifar10 as mix_exp_cifar, mix_experts_resnet_imagenet as mix_exp_imagenet
from Rational_ResNets.ResNet_Models.mix_experts_resnet_cifar10 import RationalResNet as mix_cifar
from Rational_ResNets.ResNet_Models.mix_experts_resnet_imagenet import RationalResNet as mix_img
from Rational_ResNets.ResNet_Models.univ_rational_resnet_cifar10 import RationalResNet as univ_cifar
from Rational_ResNets.ResNet_Models.univ_rational_resnet_imagenet import RationalResNet as univ_img
import numpy as np

model = mix_exp_cifar.mix_exp_resnet8(['leaky_relu', 'gelu', 'swish', 'tanh', 'sigmoid'], 5)
params = sum([np.prod(p.size()) for p in model.parameters()])
print(params)

