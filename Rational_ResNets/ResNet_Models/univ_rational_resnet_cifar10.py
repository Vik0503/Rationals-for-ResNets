"""
ResNet20 Model for CIFAR10 as originally described in: Deep Residual Learning for Image Recognition (arXiv:1512.03385)
by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
with Rational Activation Functions as activation functions instead of reLu activation functions.
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

    def __init__(self, planes_in, planes_out, stride=1, downsample=False):
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
        self.rational = Rational(cuda=cuda)
        self.conv_layer_2 = nn.Conv2d(planes_out, planes_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm_2 = nn.BatchNorm2d(planes_out)
        self.rational_2 = Rational(cuda=cuda)

        self.shortcut = nn.Sequential()
        if downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(planes_in, self.expansion * planes_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes_out)
            )

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
        out = self.rational(out)
        out = self.conv_layer_2(out)
        out = self.batch_norm_2(out)
        out += self.shortcut(x)
        out = self.rational_2(out)

        return out


class RationalResNet(nn.Module):
    """A ResNet as described in the paper above."""

    def __init__(self, block: Type[RationalBasicBlock], layers: List[int], num_classes: int = 10) -> None:
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
        self.layers = layers

        self.conv_layer_1 = nn.Conv2d(3, self.planes_in, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm_1 = self.norm_layer(self.planes_in)

        self.rational = Rational(cuda=cuda)

        self.layer1 = self.make_layer(block=block, planes_out=16, num_blocks=layers[0], stride=1)
        out_size = 16
        if len(self.layers) > 1:
            self.layer2 = self.make_layer(block=block, planes_out=32, num_blocks=layers[1], stride=2)
            out_size = 32
        if len(self.layers) > 2:
            self.layer3 = self.make_layer(block=block, planes_out=64, num_blocks=layers[2], stride=2)
            out_size = 64

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_size, num_classes)

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
        if stride != 1 or planes_out * block.expansion != self.planes_in:
            downsample = True

        layers = []
        layers.append(block(self.planes_in, planes_out, stride, downsample=downsample))

        downsample = False
        stride = 1
        self.planes_in = planes_out * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.planes_in, planes_out, stride, downsample=downsample))
        print(nn.Sequential(*layers))

        return nn.Sequential(*layers)

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
        out = self.rational(out)

        out = self.layer1(out)
        if len(self.layers) > 1:
            out = self.layer2(out)
        if len(self.layers) > 2:
            out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


def _resnet(arch: str, block: Type[RationalBasicBlock], layers: List[int], **kwargs: Any) -> RationalResNet:
    """
    The universal ResNet definition.

    Parameters
    ----------
    arch: str
          The name of the ResNet.
    block: RationalBasicBlock
           The block type of the ResNet.
    layers: list
           The list with the number of layers and the number of blocks in each layer.

    Returns
    -------
    model: RationalResNet
    """
    model = RationalResNet(block, layers, **kwargs)

    return model


def univ_rational_resnet20(**kwargs: Any) -> RationalResNet:
    """ResNet for CIFAR10 as mentioned in the paper above"""
    return _resnet('resnet20', RationalBasicBlock, [3, 3, 3], **kwargs)


def univ_rational_resnet20_2_BB(**kwargs: any) -> RationalResNet:
    return _resnet('resnet20_2_BB', RationalBasicBlock, [2, 2, 2], **kwargs)


def univ_rational_resnet20_2_layers(**kwargs: Any) -> RationalResNet:
    """ResNet for CIFAR10 as mentioned in the paper above"""
    return _resnet('resnet20', RationalBasicBlock, [3, 3], **kwargs)


def univ_rational_resnet20_1_layer(**kwargs: Any) -> RationalResNet:
    """ResNet for CIFAR10 as mentioned in the paper above"""
    return _resnet('resnet20', RationalBasicBlock, [3], **kwargs)


def rational_resnet32(**kwargs: Any) -> RationalResNet:
    """ResNet for CIFAR10 as mentioned in the paper above"""
    return _resnet('resnet32', RationalBasicBlock, [5, 5, 5], **kwargs)


def rational_resnet44(**kwargs: Any) -> RationalResNet:
    """ResNet for CIFAR10 as mentioned in the paper above"""
    return _resnet('resnet44', RationalBasicBlock, [7, 7, 7], **kwargs)


def rational_resnet56(**kwargs: Any) -> RationalResNet:
    """ResNet for CIFAR10 as mentioned in the paper above"""
    return _resnet('resnet56', RationalBasicBlock, [9, 9, 9], **kwargs)


def rational_resnet110(**kwargs: Any) -> RationalResNet:
    """ResNet for CIFAR10 as mentioned in the paper above"""
    return _resnet('resnet110_cifar10', RationalBasicBlock, [18, 18, 18], **kwargs)


def rational_resnet1202(**kwargs: Any) -> RationalResNet:
    """ResNet for CIFAR10 as mentioned in the paper above"""
    return _resnet('resnet1202', RationalBasicBlock, [200, 200, 200], **kwargs)
