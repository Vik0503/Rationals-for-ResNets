from __future__ import print_function, division

import time
from typing import Type, Any, List

import torch
import torch.nn as nn
import torchvision
from rational.torch import Rational
from torch import Tensor
from torch import optim
from torch.optim import lr_scheduler
from torchvision import transforms

from LTH_for_Rational_ResNets.Mask import Mask

if torch.cuda.is_available():
    cuda = True
    device = 'cuda'
else:
    cuda = False
    device = 'cpu'


class RationalBasicBlock(nn.Module):
    """A Basic Block as described in the paper above, with Rationals as activation function instead of ReLu"""
    expansion = 1

    def __init__(self, planes_in: int, planes_out: int, stride: int = 1, downsample: bool = False, num_rationals: int = 4):
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

        self.num_rationals = num_rationals
        self.rational_list = []
        for n in range(self.num_rationals):
            self.rational_list.append(Rational(cuda=cuda))

        self.conv_layer_2 = nn.Conv2d(planes_out, planes_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm_2 = nn.BatchNorm2d(planes_out)

        self.shortcut = nn.Sequential()
        if downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(planes_in, self.expansion * planes_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes_out)
            )

    def select_multi_variant_rationals(self, out: Tensor) -> Tensor:
        num_rationals = len(self.rational_list)
        split_size = int(out.shape[1] / num_rationals)
        splitted = torch.split(out.clone(), split_size, dim=1)
        out_list = []

        for n in range(num_rationals):
            rational = self.rational_list[n]
            out_list.append(rational(splitted[n].clone()).clone())

        for i in range(out.shape[0]):
            for n in range(num_rationals):
                out[i][n * split_size:(n + 1) * split_size] = out_list[n][i].clone()

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
        out = self.select_multi_variant_rationals(out)
        out = self.conv_layer_2(out)
        out = self.batch_norm_2(out)
        out += self.shortcut(x)
        out = self.select_multi_variant_rationals(out)

        return out


def initial_state(model):
    """Return the initial initialization before training."""
    initial_state_dict = {}
    for name, param in model.named_parameters():
        initial_state_dict[name] = param.data.clone().detach()
    return initial_state_dict


def reinit(model, mask, initial_state_model):
    """
    Reset pruned model's weights to the initial initialization.
    Parameter
    ---------
    model: RationalResNet
    mask: Mask
          A mask with pruned weights.
    initial_state_model: dict
                         Initially saved state, before the model is trained.
    """
    for name, param in model.named_parameters():
        if 'weight' not in name or 'batch_norm' in name or 'shortcut' in name or 'fc' in name:
            continue
        param.data = param.data.cpu()
        param.data = initial_state_model[name].cpu() * mask[name]


class RationalResNet(nn.Module):
    """A ResNet as described in the paper above."""

    def __init__(self, block: Type[RationalBasicBlock], layers: List[int], num_classes: int = 10, mask: Mask = None, num_rationals: int = 4) -> None:
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

        self.planes_in = 16

        self.conv_layer_1 = nn.Conv2d(in_channels=3, out_channels=self.planes_in, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm_1 = self.norm_layer(self.planes_in)

        self.num_rationals = num_rationals
        self.rational_list = []
        for n in range(self.num_rationals):
            self.rational_list.append(Rational(cuda=cuda))

        block.num_rationals = self.num_rationals
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

        self.mask = mask
        if self.mask is not None:
            self.apply_mask(mask=mask)

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
        if stride != 1 or planes_out != self.planes_in:
            downsample = True

        layers = []
        layers.append(block(self.planes_in, planes_out, stride, downsample=downsample, num_rationals=self.num_rationals))

        downsample = False
        stride = 1
        self.planes_in = planes_out * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.planes_in, planes_out, stride, downsample=downsample, num_rationals=self.num_rationals))
        print(nn.Sequential(*layers))

        return nn.Sequential(*layers)

    def apply_mask(self, mask: Mask):
        """
        Apply a new mask to a net.
        Parameters
        ----------
        self:
               The model to which the mask is applied.
        mask: Mask
        """
        if mask is not None:
            for name, param in self.named_parameters():
                if 'weight' not in name or 'batch_norm' in name or 'shortcut' in name or 'fc' in name:
                    continue
                param.data = param.data.cpu()
                param.data *= mask[name]

    def select_multi_variant_rationals(self, out: Tensor) -> Tensor:
        num_rationals = len(self.rational_list)
        split_size = int(out.shape[1] / num_rationals)
        splitted = torch.split(out.clone(), split_size, dim=1)
        out_list = []

        for n in range(num_rationals):
            rational = self.rational_list[n]
            out_list.append(rational(splitted[n].clone()).clone())

        for i in range(out.shape[0]):
            for n in range(num_rationals):
                out[i][n * split_size:(n + 1) * split_size] = out_list[n][i].clone()

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
        if self.mask is not None:
            self.apply_mask(mask=self.mask)
        self = self.to(device)
        out = out.to(device)
        out = self.conv_layer_1(out)
        out = self.batch_norm_1(out)
        out = self.select_multi_variant_rationals(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

    def prunable_layers(self) -> List:
        """
        Return all layers that are prunable.
        Returns
        -------
        prunable_layer_list: List
                            A list with all layers that are prunable.
        """
        prunable_layer_list = []
        for n, module in self.named_modules():
            if isinstance(module, nn.Conv2d) and 'shortcut' not in n:
                prunable_layer_list.append(n + '.weight')

        return prunable_layer_list


def _resnet(arch: str, block: Type[RationalBasicBlock], layers: List[int], mask: Mask, num_rationals: int, **kwargs: Any) -> RationalResNet:
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
    model = RationalResNet(block, layers, mask=mask, num_rationals=num_rationals, **kwargs)

    return model


def multi_variant_rational_resnet20(mask: Mask = None, num_rationals: int = 4, **kwargs: Any) -> RationalResNet:
    """ResNet for CIFAR10 as mentioned in the paper above"""
    return _resnet('resnet20', RationalBasicBlock, [3, 3, 3], mask=mask, num_rationals=num_rationals, **kwargs)

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_val_set = torchvision.datasets.CIFAR10(root='/home/viktoria/Git/thesis_stuff/data', train=True, download=True, transform=train_transform)
testset = torchvision.datasets.CIFAR10(root='/home/viktoria/Git/thesis_stuff/data', train=False, download=True, transform=test_transform)
trainset, valset = torch.utils.data.random_split(train_val_set, [45000, 5000])

trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=16, batch_size=128, drop_last=True)
valloader = torch.utils.data.DataLoader(valset, shuffle=True, num_workers=16, batch_size=128, drop_last=True)
testloader = torch.utils.data.DataLoader(testset, shuffle=False, num_workers=16, batch_size=128, drop_last=True)

classes = ['bird', 'car', 'cat', 'deer', 'dog', 'frog', 'horse', 'plane', 'ship', 'truck']


def train_val_test_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('*' * 10)
        since_epoch = time.time()

        # Each epoch has a training, a validation and a test phase
        for phase in ['train', 'val', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0

            if phase == 'train':
                dataloader = trainloader
            if phase == 'val':
                dataloader = valloader
            if phase == 'test':
                dataloader = testloader
            # Iterate over data.
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        loss = loss.to(device)
                        optimizer.step()

                # loss + accuracy
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                epoch_loss = running_loss / len(trainset)
                epoch_acc = running_corrects.double() / len(trainset)
                scheduler.step()

            if phase == 'val':
                epoch_loss = running_loss / len(valset)
                epoch_acc = running_corrects.double() / len(valset)

            if phase == 'test':
                epoch_loss = running_loss / len(testset)
                epoch_acc = running_corrects.double() / len(testset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc

        time_elapsed_epoch = time.time() - since_epoch
        print('Epoch finished in {:.0f}m {:.0f}s'.format(time_elapsed_epoch // 60, time_elapsed_epoch % 60))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    return model


model = multi_variant_rational_resnet20(num_rationals=8)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_val_test_model(model, criterion, optimizer, exp_lr_scheduler,
                             num_epochs=25)
