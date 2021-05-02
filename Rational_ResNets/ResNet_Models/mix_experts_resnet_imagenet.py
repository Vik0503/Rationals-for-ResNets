"""
ResNet18 Model for ImageNet as originally described in: Deep Residual Learning for Image Recognition (arXiv:1512.03385)
by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
with Rational Activation Functions as activation functions instead of reLu activation functions
"""

from __future__ import print_function, division

from typing import Type, Any, List

import torch
from torch import Tensor
import torch.nn as nn
from rational.torch import Rational

from Rational_ResNets import argparser, utils

args = argparser.get_arguments()
prune_shortcuts = args.prune_shortcuts

if torch.cuda.is_available():
    cuda = True
    device = 'cuda'
else:
    cuda = False
    device = 'cpu'


class RationalBasicBlock(nn.Module):
    """A Basic Block as described in the paper above, with Rationals as activation function instead of ReLu"""
    expansion = 1

    def __init__(self, planes_in, planes_out, rational_inits: List[str], num_rationals: int = 4, stride=1, downsample=False) -> None:
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
        self.rational_inits = rational_inits
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

        Returns
        -------
        out:    torch.Tensor
        """
        out_tensor = torch.zeros_like(out)

        for n in range(self.num_rationals):
            rational = rationals[n]
            rational_out = rational(out.clone())
            out_tensor = out_tensor.clone() + alphas[n].clone() * rational_out.clone()
        out = out_tensor.clone()
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
        out = self.multi_rational(out, self.alpha_1, self.rational_expert_group_1)
        out = self.conv_layer_2(out)
        out = self.batch_norm_2(out)
        out += self.shortcut(x)
        out = self.multi_rational(out, self.alpha_2, self.rational_expert_group_2)

        return out


class RationalResNet(nn.Module):
    """A ResNet as described in the paper above with Rationals as activation functions instead of ReLU."""
    def __init__(self, block: Type[RationalBasicBlock], layers: List[int], rational_inits: List[str], num_rationals: int, num_classes: int = 1000) -> None:
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
        mask: Mask
              The mask that is used for the Lottery Ticket Hypothesis. It sets the pruned weights to zero.
        """
        super(RationalResNet, self).__init__()

        self.norm_layer = nn.BatchNorm2d

        self.planes_in = 64
        self.layers = layers

        self.conv_layer_1 = nn.Conv2d(3, self.planes_in, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm_1 = self.norm_layer(self.planes_in)

        self.num_rationals = num_rationals
        self.rational_inits = rational_inits
        self.expert_group = []

        for n in range(self.num_rationals):
            self.expert_group.append(Rational(cuda=cuda, approx_func=self.rational_inits[n]))

        # expert group with num_rationals Rational Activation Functions
        self.rational_expert_group = nn.Sequential(*self.expert_group)

        # learnable parameter alpha for the weighted sum of the rational experts
        data = utils.initialize_alpha(self.num_rationals)
        self.alpha = torch.nn.parameter.Parameter(data, requires_grad=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block=block, planes_out=64, num_blocks=self.layers[0], stride=1)
        out_size = 64
        if len(self.layers) > 1:
            self.layer2 = self.make_layer(block=block, planes_out=128, num_blocks=self.layers[1], stride=2)
            out_size = 128
        if len(self.layers) > 2:
            self.layer3 = self.make_layer(block=block, planes_out=256, num_blocks=self.layers[2], stride=2)
            out_size = 256
        if len(self.layers) > 3:
            self.layer4 = self.make_layer(block=block, planes_out=512, num_blocks=self.layers[3], stride=2)
            out_size = 512

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_size, num_classes)

        # init model
        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                nn.init.kaiming_normal_(mod.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(mod, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(mod.weight, 1)
                nn.init.constant_(mod.bias, 0)

    def make_layer(self, block: Type[RationalBasicBlock], planes_out: int, num_blocks: int, stride: int) -> nn.Sequential:
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
        out = self.maxpool(out)

        out = self.layer1(out)
        if len(self.layers) > 1:
            out = self.layer2(out)
        if len(self.layers) > 2:
            out = self.layer3(out)
        if len(self.layers) > 3:
            out = self.layer4(out)
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
    mask: Mask

    Returns
    -------
    model: RationalResNet
    """
    model = RationalResNet(block, layers, **kwargs)

    return model


def mix_exp_resnet18(**kwargs: Any) -> RationalResNet:
    """ResNet for ImageNet as mentioned in the paper above"""
    return _resnet('resnet18', RationalBasicBlock, [2, 2, 2, 2], **kwargs)


def mix_exp_resnet18_2_layers(**kwargs: Any) -> RationalResNet:
    """ResNet for ImageNet as mentioned in the paper above"""
    return _resnet('resnet18', RationalBasicBlock, [2, 2], **kwargs)


def mix_exp_resnet18_1_layer(**kwargs: Any) -> RationalResNet:
    """ResNet for ImageNet as mentioned in the paper above"""
    return _resnet('resnet18', RationalBasicBlock, [2], **kwargs)