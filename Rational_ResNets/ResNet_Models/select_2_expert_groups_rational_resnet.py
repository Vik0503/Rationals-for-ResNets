"""
ResNet20 Model for CIFAR10 as originally described in: Deep Residual Learning for Image Recognition (arXiv:1512.03385)
by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
with Padè Activation Units as activation functions instead of reLu activation functions.
"""

from __future__ import print_function, division  # TODO: Update Doc

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

    def __init__(self, planes_in, planes_out, rational_inits: List[str], num_rationals: int = 4, stride=1, downsample=False):
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
        self.rational_inits = rational_inits
        self.conv_layer_1 = nn.Conv2d(planes_in, planes_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch_norm_1 = nn.BatchNorm2d(planes_out)
        # use Rationals instead of reLu activation function
        self.num_rationals = num_rationals
        self.expert_group_1 = []
        self.expert_group_2 = []

        for n in range(self.num_rationals):
            self.expert_group_1.append(Rational(cuda=cuda, approx_func=self.rational_inits[n]))
            self.expert_group_2.append(Rational(cuda=cuda, approx_func=self.rational_inits[n]))

        p_0 = torch.randn(1)
        p_1 = torch.randn(1)

        self.pow_1 = torch.nn.parameter.Parameter(p_0, requires_grad=True)
        self.pow_2 = torch.nn.parameter.Parameter(p_1, requires_grad=True)

        self.rational_expert_group_1 = nn.Sequential(*self.expert_group_1)
        self.rational_expert_group_2 = nn.Sequential(*self.expert_group_2)

        data_alpha_1 = initialize_alpha(self.num_rationals)
        self.alpha_1 = torch.nn.parameter.Parameter(data_alpha_1, requires_grad=True)
        self.alpha_sum_1 = torch.nn.parameter.Parameter(data_alpha_1.sum(), requires_grad=True)
        self.alpha_1 = torch.nn.parameter.Parameter(self.alpha_1 / self.alpha_sum_1, requires_grad=True)

        data_alpha_2 = initialize_alpha(self.num_rationals)
        self.alpha_2 = torch.nn.parameter.Parameter(data_alpha_2, requires_grad=True)
        self.alpha_sum_2 = torch.nn.parameter.Parameter(data_alpha_2.sum(), requires_grad=True)
        self.alpha_2 = torch.nn.parameter.Parameter(self.alpha_2 / self.alpha_sum_2, requires_grad=True)

        self.conv_layer_2 = nn.Conv2d(planes_out, planes_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm_2 = nn.BatchNorm2d(planes_out)

        self.shortcut = nn.Sequential()
        if downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(planes_in, self.expansion * planes_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes_out)
            )

    def multi_rational(self, out: Tensor, alphas, rationals, sums) -> Tensor:
        # alphas = torch.nn.parameter.Parameter(torch.tensor(alphas / sums, requires_grad=True))
        out_tensor = torch.zeros_like(out)
        for n in range(self.num_rationals):
            rational = rationals[n]
            rational_out = rational(out.clone())
            out_tensor = out_tensor.clone() + alphas[n].clone() * rational_out.clone()
        out = out_tensor.clone()
        return out

    def pow_multi_rational(self, out: Tensor, alphas: torch.Tensor, rationals, p) -> Tensor:
        alphas = torch.nn.Parameter(alphas / alphas.sum(), requires_grad=True)
        p = torch.nn.parameter.Parameter(abs(p), requires_grad=True)
        out_tensor = torch.zeros_like(out)
        for n in range(self.num_rationals):
            rational = rationals[n]
            rational_out = rational(out.clone())
            out_tensor = out_tensor.clone() + alphas[n].clone() * rational_out.clone() ** p.clone()
        out = out_tensor.clone() ** (1 / p.clone())
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
        out = self.multi_rational(out, self.alpha_1, self.rational_expert_group_1, self.alpha_sum_1)
        out = self.conv_layer_2(out)
        out = self.batch_norm_2(out)
        out += self.shortcut(x)
        out = self.multi_rational(out, self.alpha_2, self.rational_expert_group_2, self.alpha_sum_2)

        return out


class RationalResNet(nn.Module):
    """A ResNet as described in the paper above."""

    def __init__(self, block: Type[RationalBasicBlock], layers: List[int], rational_inits: List[str], num_rationals: int, num_classes: int = 10) -> None:
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
        self.rational_inits = rational_inits
        self.conv_layer_1 = nn.Conv2d(3, self.planes_in, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm_1 = self.norm_layer(self.planes_in)
        self.num_rationals = num_rationals
        self.expert_group = []

        for n in range(self.num_rationals):
            self.expert_group.append(Rational(cuda=cuda, approx_func=self.rational_inits[n]))

        self.rational_expert_group = nn.Sequential(*self.expert_group)

        p = torch.randn(1)
        self.pow = torch.nn.parameter.Parameter(p, requires_grad=True)
        data = initialize_alpha(self.num_rationals)
        self.alpha = torch.nn.parameter.Parameter(data, requires_grad=True)
        self.alpha_sum = torch.nn.parameter.Parameter(data.sum(), requires_grad=True)
        self.alpha = torch.nn.parameter.Parameter(self.alpha / self.alpha_sum, requires_grad=True)

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
        if stride != 1 or planes_out * block.expansion != self.planes_in:
            downsample = True

        layers = []
        layers.append(block(self.planes_in, planes_out, self.rational_inits, self.num_rationals, stride, downsample=downsample))

        downsample = False
        stride = 1
        self.planes_in = planes_out * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.planes_in, planes_out, self.rational_inits, self.num_rationals, stride, downsample=downsample))
        print(nn.Sequential(*layers))

        return nn.Sequential(*layers)

    def multi_rational(self, out: Tensor) -> Tensor:
        out_tensor = torch.zeros_like(out)
        # self.alpha = torch.nn.parameter.Parameter(torch.tensor(self.alpha / self.alpha_sum, requires_grad=True))
        # print(self.alpha)
        for n in range(self.num_rationals):
            rational = self.rational_expert_group[n]
            rational_out = rational(out.clone())
            out_tensor = out_tensor.clone() + self.alpha[n].clone() * rational_out.clone()
        out = out_tensor.clone()
        return out

    def pow_multi_rational(self, out: Tensor) -> Tensor:
        self.pow = torch.nn.parameter.Parameter(abs(self.pow), requires_grad=True)
        out_tensor = torch.zeros_like(out)
        # self.alpha = torch.nn.parameter.Parameter(self.alpha.clone() / self.alpha.sum(), requires_grad=True)
        print(self.alpha)
        for n in range(self.num_rationals):
            rational = self.rational_expert_group[n]
            rational_out = rational(out.clone())
            out_tensor = out_tensor.clone() + self.alpha[n].clone() * rational_out.clone() ** self.pow.clone()
        out = out_tensor.clone() ** (1 / self.pow.clone())
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
            The tensor with initial values for alpha.
    """
    alpha = torch.rand(b, requires_grad=True)
    # alpha = alpha / alpha.sum()
    return alpha


def _resnet(arch: str, block: Type[RationalBasicBlock], layers: List[int], rational_inits: List[str], num_rationals: int, **kwargs: Any) -> RationalResNet:
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
    model = RationalResNet(block, layers, rational_inits, num_rationals, **kwargs)

    return model


def select_2_expert_groups_rational_resnet20(rational_inits: List[str], num_rationals: int = 4, **kwargs: Any) -> RationalResNet:
    """ResNet for CIFAR10 as mentioned in the paper above"""
    return _resnet('resnet20', RationalBasicBlock, [3, 3, 3], rational_inits=rational_inits, num_rationals=num_rationals, **kwargs)


def select_2_expert_groups_rational_resnet32(rational_inits: List[str], num_rationals: int = 4, **kwargs: Any) -> RationalResNet:
    """ResNet for CIFAR10 as mentioned in the paper above"""
    return _resnet('resnet32', RationalBasicBlock, [5, 5, 5], rational_inits=rational_inits, num_rationals=num_rationals, **kwargs)


def select_2_expert_groups_rational_resnet44(rational_inits: List[str], num_rationals: int = 4, **kwargs: Any) -> RationalResNet:
    """ResNet for CIFAR10 as mentioned in the paper above"""
    return _resnet('resnet44', RationalBasicBlock, [7, 7, 7], rational_inits=rational_inits, num_rationals=num_rationals, **kwargs)


def select_2_expert_groups_rational_resnet56(rational_inits: List[str], num_rationals: int = 4, **kwargs: Any) -> RationalResNet:
    """ResNet for CIFAR10 as mentioned in the paper above"""
    return _resnet('resnet56', RationalBasicBlock, [9, 9, 9], rational_inits=rational_inits, num_rationals=num_rationals, **kwargs)


def select_2_expert_groups_rational_resnet110(rational_inits: List[str], num_rationals: int = 4, **kwargs: Any) -> RationalResNet:
    """ResNet for CIFAR10 as mentioned in the paper above"""
    return _resnet('resnet110_cifar10', RationalBasicBlock, [18, 18, 18], rational_inits=rational_inits, num_rationals=num_rationals, **kwargs)


def select_2_expert_groups_rational_resnet1202(rational_inits: List[str], num_rationals: int = 4, **kwargs: Any) -> RationalResNet:
    """ResNet for CIFAR10 as mentioned in the paper above"""
    return _resnet('resnet1202', RationalBasicBlock, [200, 200, 200], rational_inits=rational_inits, num_rationals=num_rationals, **kwargs)
