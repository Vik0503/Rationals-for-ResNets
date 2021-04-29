"""
ResNet Model for CIFAR10 as originally described in: Deep Residual Learning for Image Recognition (arXiv:1512.03385)
by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
"""

from __future__ import print_function, division

from typing import Type, Any, List

import torch
import torch.nn as nn
from torch import Tensor

from LTH_for_Rational_ResNets.Mask import Mask
from LTH_for_Rational_ResNets import argparser

args = argparser.get_arguments()
prune_shortcuts = args.prune_shortcuts

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


class BasicBlock(nn.Module):
    """A Basic Block as described in the paper above, with ReLu as activation function."""
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
        super(BasicBlock, self).__init__()
        self.conv_layer_1 = nn.Conv2d(planes_in, planes_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch_norm_1 = nn.BatchNorm2d(planes_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv_layer_2 = nn.Conv2d(planes_out, planes_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm_2 = nn.BatchNorm2d(planes_out)

        self.shortcut = nn.Sequential()
        if downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(planes_in, self.expansion * planes_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes_out)
            )

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
        out = self.relu(out)
        out = self.conv_layer_2(out)
        out = self.batch_norm_2(out)
        out += self.shortcut(x)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """A ResNet as described in the paper above."""
    def __init__(self, block: Type[BasicBlock], layers: List[int], num_classes: int = 10, mask: Mask = None) -> None:
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
        super(ResNet, self).__init__()

        self.norm_layer = nn.BatchNorm2d

        self.planes_in = 16
        self.layers = layers

        self.conv_layer_1 = nn.Conv2d(3, self.planes_in, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm_1 = self.norm_layer(self.planes_in)
        self.relu = nn.ReLU(inplace=True)

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

        self.mask = mask
        if self.mask is not None:
            self.apply_mask(mask=self.mask)

    def make_layer(self, block: Type[BasicBlock], planes_out: int, num_blocks: int, stride: int) -> nn.Sequential:
        """
        Build ResNet's layers. Each layer contains a number of Basic Blocks.

        Parameters
        ----------
        block: BasicBlock
        planes_out: int
        num_blocks: int
                    The number of BasicBlocks in this layer.
        stride: int

        Returns
        -------
        nn.Sequential
                     A layer build with BasicBlocks.
        """
        downsample = False
        if stride != 1 or planes_out != self.planes_in:
            downsample = True

        layers = [block(self.planes_in, planes_out, stride, downsample=downsample)]

        downsample = False
        stride = 1
        self.planes_in = planes_out * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.planes_in, planes_out, stride, downsample=downsample))
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

    def forward(self, out: Tensor) -> Tensor:
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
        out = self.relu(out)

        out = self.layer1(out)
        if len(self.layers) > 1:
            out = self.layer2(out)
        if len(self.layers) > 2:
            out = self.layer3(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


def _resnet(arch: str, block: Type[BasicBlock], layers: List[int], mask: Mask, **kwargs: Any) -> ResNet:
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
    mask: Mask

    Returns
    -------
    model: RationalResNet
    """
    model = ResNet(block, layers, mask=mask, **kwargs)

    return model


def relu_resnet20(mask: Mask = None, **kwargs: Any) -> ResNet:
    """ResNet20 for CIFAR10 as mentioned in the paper above."""
    return _resnet('resnet20', BasicBlock, [3, 3, 3], mask=mask, **kwargs)


def relu_resnet20_2_BB(mask: Mask = None, **kwargs: Any) -> ResNet:
    """Smaller ResNet for CIFAR10 with only two BasicBlocks in layers 2 and 3."""
    return _resnet('resnet20_2_BB', BasicBlock, [3, 2, 2], mask=mask, **kwargs)


def relu_resnet20_2_layers(mask: Mask = None, **kwargs: Any) -> ResNet:
    """Smaller ResNet for CIFAR10 with only two layers."""
    return _resnet('resnet20_2_layers', BasicBlock, [3, 3], mask=mask, **kwargs)


def relu_resnet20_1_layer(mask: Mask = None, **kwargs: Any) -> ResNet:
    """Smaller ResNet for CIFAR10 with only one layer."""
    return _resnet('resnet20', BasicBlock, [3], mask=mask, **kwargs)
