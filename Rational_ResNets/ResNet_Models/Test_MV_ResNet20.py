"""
ResNet20 Model for CIFAR10 as originally described in: Deep Residual Learning for Image Recognition (arXiv:1512.03385)
by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
with Padè Activation Units as activation functions instead of reLu activation functions.
"""

from __future__ import print_function, division

from typing import Type, Any, List

import torch
import torch.nn as nn
from rational.torch import Rational
from torch import Tensor

if torch.cuda.is_available():
    cuda = True
    device = 'cuda'
else:
    cuda = False
    device = 'cpu'


class RationalBasicBlock(nn.Module):
    """A Basic Block as described in the paper above, with Rationals as activation function instead of ReLu."""
    expansion = 1

    def __init__(self, planes_in, planes_out, num_rationals: int = 4, stride=1, downsample=False):
        """
        Initialize the Basic Block.

        Parameters
        ----------
        planes_in: int
                   The number of channels into the first convolutional layer.
        planes_out: int
                    The number of channels that go out of the convolutional layers.
        stride: int
        downsample: bool
        """
        super(RationalBasicBlock, self).__init__()
        self.conv_layer_1 = nn.Conv2d(planes_in, planes_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch_norm_1 = nn.BatchNorm2d(planes_out)
        # use Rationals instead of reLu activation function
        self.num_rationals = num_rationals
        self.rational_1 = Rational(cuda=cuda, approx_func="leaky_relu")
        self.rational_2 = Rational(cuda=cuda, approx_func="sigmoid")
        self.rational_3 = Rational(cuda=cuda, approx_func="gelu")
        self.rational_4 = Rational(cuda=cuda, approx_func="tanh")
        self.rational_5 = Rational(cuda=cuda, approx_func="swish")
        data = initialize_alpha(5)
        self.alpha = torch.nn.parameter.Parameter(data, requires_grad=True)
        self.conv_layer_2 = nn.Conv2d(planes_out, planes_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm_2 = nn.BatchNorm2d(planes_out)

        self.shortcut = nn.Sequential()
        if downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(planes_in, self.expansion * planes_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes_out)
            )

    def multi_rational(self, out: Tensor) -> Tensor:
        rat_out_1 = self.rational_1(out.clone())
        rat_out_2 = self.rational_2(out.clone())
        rat_out_3 = self.rational_3(out.clone())
        rat_out_4 = self.rational_4(out.clone())
        rat_out_5 = self.rational_5(out.clone())

        out = self.alpha[0].clone() * rat_out_1.clone() + self.alpha[1].clone() * rat_out_2.clone() + self.alpha[2].clone() * rat_out_3.clone() + self.alpha[3].clone() * rat_out_4.clone() + self.alpha[4].clone() * rat_out_5.clone()
        return out

    def forward(self, x: Tensor) -> Tensor:
        """
        Move input forward through the basic block.

        Parameters
        ----------
        x: Tensor
             Training input value.

        Returns
        -------
        out: Tensor
             Fed forward input value.
        """
        out = self.conv_layer_1(x)
        out = self.batch_norm_1(out)
        out = self.multi_rational(out)
        out = self.conv_layer_2(out)
        out = self.batch_norm_2(out)
        out += self.shortcut(x)
        out = self.multi_rational(out)

        return out


class RationalResNet(nn.Module):
    """A ResNet as described in the paper above."""

    def __init__(self, block: Type[RationalBasicBlock], layers: List[int], num_rationals: int, num_classes: int = 10) -> None:
        """
        Initialize parameters of the ResNet.

        Parameters
        ----------
        block: RationalBasicBlock
               THe block type that is used to build the layers of the ResNet.
        layers: List[int]
                The list with the number of layers, and the number of blocks in each layer.
        num_classes: int
                     The number of different classes in a dataset.
        """
        super(RationalResNet, self).__init__()

        self.norm_layer = nn.BatchNorm2d
        self.planes_in = 16

        self.conv_layer_1 = nn.Conv2d(3, self.planes_in, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm_1 = self.norm_layer(self.planes_in)
        self.num_rationals = num_rationals
        self.rational_1 = Rational(cuda=cuda, approx_func="leaky_relu")
        self.rational_2 = Rational(cuda=cuda, approx_func="sigmoid")
        self.rational_3 = Rational(cuda=cuda, approx_func="gelu")
        self.rational_4 = Rational(cuda=cuda, approx_func="tanh")
        self.rational_5 = Rational(cuda=cuda, approx_func="swish")
        data = initialize_alpha(5)
        self.alpha = torch.nn.parameter.Parameter(data, requires_grad=True)

        self.layer1 = self.make_layer(block=block, planes_out=16, num_blocks=layers[0], stride=1)
        self.layer2 = self.make_layer(block=block, planes_out=32, num_blocks=layers[1], stride=2)
        self.layer3 = self.make_layer(block=block, planes_out=64, num_blocks=layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                nn.init.kaiming_normal_(mod.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(mod, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(mod.weight, 1)
                nn.init.constant_(mod.bias, 0)

    def make_layer(self, block: Type[RationalBasicBlock], planes_out: int, num_blocks: int, stride: int):
        """
        Build ResNet's layers. Each layer contains a number of Basic Blocks.

        Parameters
        ----------
        block: RationalBasicBlock
        planes_out: int
        num_blocks: int
                    The number of RationalBasicBlocks in this layer.
        stride: int

        Returns
        -------
        nn.Sequential
                     A layer build with RationalBasicBlocks.
        """
        downsample = False
        if stride != 1 or planes_out != self.planes_in:
            downsample = True

        layers = []
        layers.append(block(self.planes_in, planes_out, self.num_rationals, stride, downsample=downsample))

        downsample = False
        stride = 1
        self.planes_in = planes_out * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.planes_in, planes_out, self.num_rationals, stride, downsample=downsample))
        print(nn.Sequential(*layers))

        return nn.Sequential(*layers)

    def multi_rational(self, out: Tensor) -> Tensor:
        rat_out_1 = self.rational_1(out.clone())
        rat_out_2 = self.rational_2(out.clone())
        rat_out_3 = self.rational_3(out.clone())
        rat_out_4 = self.rational_4(out.clone())
        rat_out_5 = self.rational_5(out.clone())

        out = self.alpha[0].clone() * rat_out_1.clone() + self.alpha[1].clone() * rat_out_2.clone() + self.alpha[2].clone() * rat_out_3.clone() + self.alpha[3].clone() * rat_out_4.clone() + self.alpha[4].clone() * rat_out_5.clone()
        return out

    def forward(self, out: Tensor):
        """
        Move input forward through the net.

        Parameters
        ----------
        out: Tensor
             Training input value.

        Returns
        -------
        out: Tensor
             Fed forward input value.
        """
        out = self.conv_layer_1(out)
        out = self.batch_norm_1(out)
        out = self.multi_rational(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


def initialize_alpha(b: int = 4) -> torch.Tensor:
    """Initialize the vector alpha.

    Parameters
    ----------
    b : int
        The length of the vector alpha.

    Returns
    -------
    alpha : torch.Tensor
            The tensor with inial values for alpha.
    """
    alpha = []
    while len(alpha) < b:
        tmp = torch.rand(1)
        if sum(alpha) + tmp <= 1.0:
            alpha.append(tmp)
    alpha = torch.tensor(alpha)
    alpha /= alpha.sum()
    return alpha


def _resnet(arch: str, block: Type[RationalBasicBlock], layers: List[int], num_rationals: int, **kwargs: Any) -> RationalResNet:
    """
    The universal ResNet definition.

    Parameters
    ----------
    arch: str
          The name of the ResNet.
    block: RationalBasicBlock
           The block type of the ResNet.
    layers: list
           The list with the number of layers, and the number of blocks in each layer.

    Returns
    -------
    model: RationalResNet
    """
    print('inside def 2', num_rationals)
    model = RationalResNet(block, layers, num_rationals, **kwargs)

    return model


def test_mv_resnet20(num_rationals: int = 4, **kwargs: Any) -> RationalResNet:
    """ResNet for CIFAR10 as mentioned in the paper above"""
    return _resnet('resnet20', RationalBasicBlock, [3, 3, 3], num_rationals=num_rationals, **kwargs)


def test_mv_resnet32(num_rationals: int = 4, **kwargs: Any) -> RationalResNet:
    """ResNet for CIFAR10 as mentioned in the paper above"""
    return _resnet('resnet32', RationalBasicBlock, [5, 5, 5], num_rationals=num_rationals, **kwargs)


def test_mv_resnet44(num_rationals: int = 4, **kwargs: Any) -> RationalResNet:
    """ResNet for CIFAR10 as mentioned in the paper above"""
    return _resnet('resnet44', RationalBasicBlock, [7, 7, 7], num_rationals=num_rationals, **kwargs)


def test_mv_resnet56(num_rationals: int = 4, **kwargs: Any) -> RationalResNet:
    """ResNet for CIFAR10 as mentioned in the paper above"""
    return _resnet('resnet56', RationalBasicBlock, [9, 9, 9], num_rationals=num_rationals, **kwargs)


def test_mv_resnet110(num_rationals: int = 4, **kwargs: Any) -> RationalResNet:
    """ResNet for CIFAR10 as mentioned in the paper above"""
    return _resnet('resnet110_cifar10', RationalBasicBlock, [18, 18, 18], num_rationals=num_rationals, **kwargs)


def test_mv_resnet1202(num_rationals: int = 4, **kwargs: Any) -> RationalResNet:
    """ResNet for CIFAR10 as mentioned in the paper above"""
    return _resnet('resnet1202', RationalBasicBlock, [200, 200, 200], num_rationals=num_rationals, **kwargs)
