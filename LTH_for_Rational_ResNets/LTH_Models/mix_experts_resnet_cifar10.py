"""
ResNet20 Model for CIFAR10 as originally described in: Deep Residual Learning for Image Recognition (arXiv:1512.03385)
by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
with Rational Activation Function expert groups as activation functions instead of reLu activation functions.
"""

from __future__ import print_function, division

from typing import Type, Any, List

import torch
import torch.nn as nn
from rational.torch import Rational

from LTH_for_Rational_ResNets import argparser
from LTH_for_Rational_ResNets import utils
from LTH_for_Rational_ResNets.Mask import Mask

args = argparser.get_arguments()
prune_shortcuts = args.prune_shortcuts


if torch.cuda.is_available():
    cuda = True
    device = 'cuda'
else:
    cuda = False
    device = 'cpu'


class RationalBasicBlock(nn.Module):
    """A Basic Block as described in the paper above, with two groups of rational experts as activation functions instead of ReLu."""
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
        self.softmax = torch.nn.Softmax(dim=0)
        # use Rationals instead of reLu activation function
        self.num_rationals = num_rationals
        self.expert_group_1 = []
        self.expert_group_2 = []

        for n in range(self.num_rationals):
            self.expert_group_1.append(Rational(cuda=cuda, approx_func=self.rational_inits[n]))
            self.expert_group_2.append(Rational(cuda=cuda, approx_func=self.rational_inits[n]))

        # two expert groups with num_rationals Rational Activation Functions
        self.rational_expert_group_1 = nn.Sequential(*self.expert_group_1)
        self.rational_expert_group_2 = nn.Sequential(*self.expert_group_2)

        # learnable parameter alpha for the weighted sum of the rational experts
        data_alpha_1 = utils.initialize_alpha(self.num_rationals)
        self.alpha_1 = torch.nn.parameter.Parameter(data_alpha_1, requires_grad=True)

        data_alpha_2 = utils.initialize_alpha(self.num_rationals)
        self.alpha_2 = torch.nn.parameter.Parameter(data_alpha_2, requires_grad=True)

        self.conv_layer_2 = nn.Conv2d(planes_out, planes_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm_2 = nn.BatchNorm2d(planes_out)

        self.shortcut = nn.Sequential()
        if downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(planes_in, self.expansion * planes_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes_out)
            )

    def multi_rational(self, out: torch.Tensor, alphas: torch.Tensor, rationals) -> torch.Tensor:
        """
        Calculate weighted sum of rational experts.

        Parameters
        ----------
        out:    torch.Tensor
        alphas: torch.Tensor
                The weights of the rational experts.
        rationals: nn.Sequential
                   The rational expert group.

        Returns
        -------
        out:    torch.Tensor
        """
        out_tensor = torch.zeros_like(out)
        softmax_alpha = self.softmax(alphas)
        for n in range(self.num_rationals):
            rational = rationals[n]
            rational_out = rational(out.clone())
            out_tensor += softmax_alpha[n] * rational_out
        out = out_tensor
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
        out = self.multi_rational(out, self.alpha_1, self.rational_expert_group_1)
        out = self.conv_layer_2(out)
        out = self.batch_norm_2(out)
        out += self.shortcut(x)
        out = self.multi_rational(out, self.alpha_2, self.rational_expert_group_2)

        return out


class RationalResNet(nn.Module):
    """A ResNet as described in the paper above."""

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

        self.layers = layers
        self.norm_layer = nn.BatchNorm2d
        self.softmax = torch.nn.Softmax(dim=0)
        self.planes_in = 16

        self.rational_inits = rational_inits

        self.conv_layer_1 = nn.Conv2d(3, self.planes_in, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm_1 = self.norm_layer(self.planes_in)
        self.num_rationals = num_rationals
        self.expert_group = []

        for n in range(self.num_rationals):
            self.expert_group.append(Rational(cuda=cuda, approx_func=self.rational_inits[n]))

        # expert group with num_rationals Rational Activation Functions
        self.rational_expert_group = nn.Sequential(*self.expert_group)

        # learnable parameter alpha for the weighted sum of the rational experts
        data = utils.initialize_alpha(self.num_rationals)
        self.alpha = torch.nn.parameter.Parameter(data, requires_grad=True)

        out_size = 16
        self.layer1 = self.make_layer(block=block, planes_out=16, num_blocks=self.layers[0], stride=1)
        if len(self.layers) > 1:
            self.layer2 = self.make_layer(block=block, planes_out=32, num_blocks=self.layers[1], stride=2)
            out_size = 32
        if len(self.layers) > 2:
            self.layer3 = self.make_layer(block=block, planes_out=64, num_blocks=self.layers[2], stride=2)
            out_size = 64

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_size, num_classes)

        # init model
        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                nn.init.kaiming_normal_(mod.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(mod, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(mod.weight, 1)
                nn.init.constant_(mod.bias, 0)

        # apply mask
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
        if stride != 1 or planes_out * block.expansion != self.planes_in:
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
        Apply net's mask.

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
        alpha_softmax = self.softmax(self.alpha)
        for n in range(self.num_rationals):
            rational = self.rational_expert_group[n]
            rational_out = rational(out.clone())
            out_tensor += alpha_softmax[n] * rational_out
        out = out_tensor
        return out

    def forward(self, out: torch.Tensor):
        """
        Move input forward through the net.

        Parameters
        ----------
        out: torch.Tensor
             Training input value.

        Returns
        -------
        out: torch.Tensor
             Fed forward input value.
        """
        # apply mask
        if self.mask is not None:
            self.apply_mask(mask=self.mask)

        out = self.conv_layer_1(out)
        out = self.batch_norm_1(out)
        out = self.multi_rational(out)

        out = self.layer1(out)
        if len(self.layers) > 1:
            out = self.layer2(out)
        if len(self.layers) > 2:
            out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


def _resnet(arch: str, block: Type[RationalBasicBlock], layers: List[int], rational_inits: List[str], num_rationals: int, mask: Mask, **kwargs: Any) -> RationalResNet:
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
    model = RationalResNet(block, layers, rational_inits, num_rationals, mask=mask, **kwargs)

    return model


def mix_exp_resnet20(rational_inits: List[str], num_rationals: int = 4, mask: Mask = None, **kwargs: Any) -> RationalResNet:
    """ResNet for CIFAR10 as mentioned in the paper above"""
    return _resnet('mix_experts_resnet20', RationalBasicBlock, [3, 3, 3], rational_inits=rational_inits, num_rationals=num_rationals, mask=mask, **kwargs)


def mix_exp_resnet14_A(rational_inits: List[str], num_rationals: int = 4, mask: Mask = None, **kwargs: Any) -> RationalResNet:
    """ResNet for CIFAR10 as mentioned in the paper above"""
    return _resnet('mix_exp_resnet14_A', RationalBasicBlock, [2, 2, 2], rational_inits=rational_inits, num_rationals=num_rationals, mask=mask, **kwargs)


def mix_exp_resnet14_B(rational_inits: List[str], num_rationals: int = 4, mask: Mask = None, **kwargs: Any) -> RationalResNet:
    """ResNet for CIFAR10 as mentioned in the paper above"""
    return _resnet('mix_exp_resnet14_B', RationalBasicBlock, [3, 3], rational_inits=rational_inits, num_rationals=num_rationals, mask=mask, **kwargs)


def mix_exp_resnet8(rational_inits: List[str], num_rationals: int = 4, mask: Mask = None, **kwargs: Any) -> RationalResNet:
    """ResNet for CIFAR10 as mentioned in the paper above"""
    return _resnet('mix_exp_resnet8', RationalBasicBlock, [3], rational_inits=rational_inits, num_rationals=num_rationals, mask=mask, **kwargs)


