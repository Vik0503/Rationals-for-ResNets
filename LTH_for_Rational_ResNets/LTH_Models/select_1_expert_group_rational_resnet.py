"""
ResNet Model for CIFAR10 as originally described in: Deep Residual Learning for Image Recognition (arXiv:1512.03385)
by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
with a mixture of rational expert groups as activation functions instead of reLu activation functions.
"""

from __future__ import print_function, division
from typing import Type, Any, List

import torch
import torch.nn as nn
from rational.torch import Rational
from torch import Tensor

from LTH_for_Rational_ResNets.Mask import Mask
from LTH_for_Rational_ResNets import argparser
from LTH_for_Rational_ResNets import utils

args = argparser.get_arguments()
prune_shortcuts = args.prune_shortcuts

if torch.cuda.is_available():
    cuda = True
    device = 'cuda'
else:
    cuda = False
    device = 'cpu'


class RationalBasicBlock(nn.Module):
    """A Basic Block as described in the paper above, with two rational expert groups as activation function instead of ReLu."""
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
        self.conv_layer_1 = nn.Conv2d(planes_in, planes_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch_norm_1 = nn.BatchNorm2d(planes_out)

        self.rational_inits = rational_inits
        self.num_rationals = num_rationals

        expert_group = []
        for n in range(self.num_rationals):
            self.expert_group.append(Rational(cuda=cuda, approx_func=self.rational_inits[n]))

        # expert group with num_rationals Rational Activation Functions
        self.rational_expert_group = nn.Sequential(*expert_group)

        # learnable parameter alpha for the weighted sum of the rational experts
        data_alpha = utils.initialize_alpha(self.num_rationals)
        self.alpha = torch.nn.parameter.Parameter(data_alpha, requires_grad=True)

        self.conv_layer_2 = nn.Conv2d(planes_out, planes_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm_2 = nn.BatchNorm2d(planes_out)

        self.shortcut = nn.Sequential()
        if downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(planes_in, self.expansion * planes_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes_out)
            )

    def multi_rational(self, out: torch.Tensor) -> torch.Tensor:
        """
        Calculate weighted sum of rational experts.

        Parameters
        ----------
        out:    torch.Tensor

        Returns
        -------
        out:    torch.Tensor
        """
        out_tensor = torch.zeros_like(out)
        for n in range(self.num_rationals):
            rational = self.rational_expert_group[n]
            rational_out = rational(out.clone())
            out_tensor = out_tensor.clone() + self.alpha[n].clone() * rational_out.clone()
        out = out_tensor.clone()
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Move input forward through the basic block.

        Parameters
        ----------
        x: torch.Tensor
             Training input value.

        Returns
        -------
        out: torch.Tensor
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
    """A ResNet as described in the paper above with one rational expert group as activation function instead of ReLU."""

    def __init__(self, block: Type[RationalBasicBlock], layers: List[int], rational_inits: List[str], num_rationals: int, num_classes: int = 10, mask: Mask = None) -> None:
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
        self.rational_inits = rational_inits
        expert_group = []
        for n in range(self.num_rationals):
            self.expert_group.append(Rational(cuda=cuda, approx_func=self.rational_inits[n]))

        # expert group with num_rationals Rational Activation Functions
        self.rational_expert_group = nn.Sequential(*expert_group)

        # learnable parameter alpha for the weighted sum of the rational experts
        data_alpha = utils.initialize_alpha(self.num_rationals)
        self.alpha = torch.nn.parameter.Parameter(data_alpha, requires_grad=True)

        self.layer1 = self.make_layer(block=block, planes_out=16, num_blocks=self.layers[0], stride=1)
        self.layer2 = self.make_layer(block=block, planes_out=32, num_blocks=self.layers[1], stride=2)
        self.layer3 = self.make_layer(block=block, planes_out=64, num_blocks=self.layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        # init model
        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                nn.init.kaiming_normal_(mod.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(mod, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(mod.weight, 1)
                nn.init.constant_(mod.bias, 0)

        self.mask = mask
        if self.mask is not None:
            self.apply_mask(mask=mask)

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

        layers = [block(self.planes_in, planes_out, self.rational_inits, self.num_rationals, stride, downsample=downsample)]

        downsample = False
        stride = 1
        self.planes_in = planes_out * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.planes_in, planes_out, self.rational_inits, self.num_rationals, stride, downsample=downsample))
        print(nn.Sequential(*layers))

        return nn.Sequential(*layers)

    def apply_mask(self, mask: Mask):
        """
        Apply the net's mask.

        Parameters
        ----------
        mask: Mask
        """
        if mask is not None:
            for name, param in self.named_parameters():
                if prune_shortcuts:
                    if 'weight' not in name or 'batch_norm' in name or 'fc' in name or 'shortcut.1.' in name:
                        continue
                    param.data *= mask[name]
                else:
                    if 'weight' not in name or 'batch_norm' in name or 'shortcut' in name or 'fc' in name:
                        continue
                    param.data *= mask[name]

    def multi_rational(self, out: torch.Tensor) -> torch.Tensor:
        """
        Calculate weighted sum of rational experts.

        Parameters
        ----------
        out:    torch.Tensor

        Returns
        -------
        out:    torch.Tensor
        """
        out_tensor = torch.zeros_like(out)
        for n in range(self.num_rationals):
            rational = self.rational_expert_group[n]
            rational_out = rational(out.clone())
            out_tensor = out_tensor.clone() + self.alpha[n].clone() * rational_out.clone()
        out = out_tensor.clone()
        return out

    def forward(self, out: torch.Tensor) -> torch.Tensor:
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
        # apply mask
        if self.mask is not None:
            self.apply_mask(mask=self.mask)

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


def select_1_expert_group_rational_resnet20(rational_inits: List[str], num_rationals: int = 4, mask: Mask = None, **kwargs: Any) -> RationalResNet:
    """ResNet for CIFAR10 as mentioned in the paper above with one expert group per BasicBlock."""
    return _resnet('resnet20', RationalBasicBlock, [3, 3, 3], rational_inits=rational_inits, num_rationals=num_rationals, mask=mask, **kwargs)


def select_1_expert_group_rational_resnet32(rational_inits: List[str], num_rationals: int = 4, mask: Mask = None, **kwargs: Any) -> RationalResNet:
    """ResNet for CIFAR10 as mentioned in the paper above with one expert group per BasicBlock."""
    return _resnet('resnet32', RationalBasicBlock, [5, 5, 5], rational_inits=rational_inits, num_rationals=num_rationals, mask=mask, **kwargs)


def select_1_expert_group_rational_resnet44(rational_inits: List[str], num_rationals: int = 4, mask: Mask = None, **kwargs: Any) -> RationalResNet:
    """ResNet for CIFAR10 as mentioned in the paper above with one expert group per BasicBlock."""
    return _resnet('resnet44', RationalBasicBlock, [7, 7, 7], rational_inits=rational_inits, num_rationals=num_rationals, mask=mask, **kwargs)


def select_1_expert_group_rational_resnet56(rational_inits: List[str], num_rationals: int = 4, mask: Mask = None, **kwargs: Any) -> RationalResNet:
    """ResNet for CIFAR10 as mentioned in the paper above with one expert group per BasicBlock."""
    return _resnet('resnet56', RationalBasicBlock, [9, 9, 9], rational_inits=rational_inits, num_rationals=num_rationals, mask=mask, **kwargs)


def select_1_expert_group_rational_resnet110(rational_inits: List[str], num_rationals: int = 4, mask: Mask = None, **kwargs: Any) -> RationalResNet:
    """ResNet for CIFAR10 as mentioned in the paper above with one expert group per BasicBlock."""
    return _resnet('resnet110_cifar10', RationalBasicBlock, [18, 18, 18], rational_inits=rational_inits, num_rationals=num_rationals, mask=mask, **kwargs)


def select_1_expert_group_rational_resnet1202(rational_inits: List[str], num_rationals: int = 4, mask: Mask = None, **kwargs: Any) -> RationalResNet:
    """ResNet for CIFAR10 as mentioned in the paper above with one expert group per BasicBlock."""
    return _resnet('resnet1202', RationalBasicBlock, [200, 200, 200], rational_inits=rational_inits, num_rationals=num_rationals, mask=mask, **kwargs)