from __future__ import print_function, division
import time
from datetime import datetime
from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
import torchvision
from rational.torch import Rational
# from rtpt import RTPT
from sklearn.metrics import precision_score, multilabel_confusion_matrix, recall_score, confusion_matrix
from torch import Tensor, optim
from torch.hub import load_state_dict_from_url
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import pandas as pd
import seaborn as sns
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

"""Class RationalResNet and RationalBasicBloc based on RESNET by PyTorch (https://pytorch.org/hub/pytorch_vision_resnet/,
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)"""

__all__ = ['RationalResNet', 'rational_resnet18']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class RationalBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(RationalBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        if torch.cuda.is_available():  # use Rationals instead of reLu activation function
            self.rational = Rational().cuda()
        else:
            self.rational = Rational()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.rational(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.rational(out)

        return out


class RationalResNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[RationalBasicBlock]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(RationalResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        if torch.cuda.is_available():
            self.rational = Rational().cuda()
        else:
            self.rational = Rational().cpu()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, RationalBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[RationalBasicBlock]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        print(nn.Sequential(*layers))
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.rational(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
        arch: str,
        block: Type[Union[RationalBasicBlock]],
        layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs: Any
) -> RationalResNet:
    model = RationalResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def rational_resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RationalResNet:
    """ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', RationalBasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


# rtpt = RTPT(name_initials='VS', experiment_name='DebuggingRationalResNet', max_iterations=25, iteration_start=0,
           # update_interval=1)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=16, drop_last=True)
test_data_set = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
testset, valset = torch.utils.data.random_split(test_data_set, [13016, 13016])
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=16, drop_last=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=True, num_workers=16, drop_last=True)

classes = trainset.labels
class_names_str = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

writer = SummaryWriter('Rational_ResNets/runs/rational_resnet18_SVHN_run19')

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

"""Method train_val_test_model based on https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html"""


def train_val_test_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    avg_epoch_time = []
    best_acc = 0.0
    all_train_preds = []
    all_train_labels = []
    all_test_labels = []
    all_test_preds = []
    accuracy_plot_x_vals = []
    train_acc_plot_y_vals = []
    val_acc_plot_y_vals = []
    test_acc_plot_y_vals = []

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
                        all_train_preds.append(preds.cpu().numpy())
                        all_train_labels.append(labels.cpu().numpy())

                if phase == 'test':
                    all_test_preds.append(preds.cpu().numpy())
                    all_test_labels.append(labels.cpu().numpy())

                # loss + accuracy
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                epoch_loss = running_loss / len(trainset)
                writer.add_scalar('training loss', epoch_loss, epoch)
                epoch_acc = running_corrects.double() / len(trainset)
                writer.add_scalar('training accuracy', epoch_acc, epoch)
                train_acc_plot_y_vals.append(epoch_acc.cpu() * 100)
                scheduler.step()

            if phase == 'val':
                epoch_loss = running_loss / len(valset)
                writer.add_scalar('validation loss', epoch_loss, epoch)
                epoch_acc = running_corrects.double() / len(valset)
                val_acc_plot_y_vals.append(epoch_acc.cpu() * 100)
                writer.add_scalar('validation accuracy', epoch_acc, epoch)

            if phase == 'test':
                epoch_loss = running_loss / len(testset)
                writer.add_scalar('test loss', epoch_loss, epoch)
                epoch_acc = running_corrects.double() / len(testset)
                test_acc_plot_y_vals.append(epoch_acc.cpu() * 100)
                writer.add_scalar('test accuracy', epoch_acc, epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc

        accuracy_plot_x_vals.append(epoch)

        cm = torch.tensor(confusion_matrix(labels.to('cpu'), preds.to('cpu')))
        print('confusion matrix: ', cm)

        time_elapsed_epoch = time.time() - since_epoch
        avg_epoch_time.append(time_elapsed_epoch)
        print('Epoch finished in {:.0f}m {:.0f}s'.format(time_elapsed_epoch // 60, time_elapsed_epoch % 60))
        # rtpt._last_iteration_time_start = time_elapsed_epoch * (num_epochs - epoch)
        # rtpt.step()

    summary_plot(accuracy_plot_x_vals, train_acc_plot_y_vals, val_acc_plot_y_vals, test_acc_plot_y_vals)
    train_labels_array = confusion_prepare(all_train_labels)
    train_preds_array = confusion_prepare(all_train_preds)
    cm_train = confusion_matrix(train_labels_array, train_preds_array, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    print('cm_train: ', cm_train)
    # plot_confusion_matrix(cm_train, num_epochs, time_elapsed_epoch, 'train_1.svg')

    test_labels_array = confusion_prepare(all_test_labels)
    test_preds_array = confusion_prepare(all_test_preds)
    cm_test = confusion_matrix(test_labels_array, test_preds_array, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    print('cm_test: ', cm_test)
    time_elapsed_epoch = average_epoch_time(avg_epoch_time)
    plot_confusion_matrix(cm_test, num_epochs, time_elapsed_epoch, 'test_1.svg', best_acc)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    return model


def average_epoch_time(avg_epoch_time):  # calculates average time per epoch
    avg_epoch_int = 0.0
    for i in range(len(avg_epoch_time)):
        avg_epoch_int += avg_epoch_time[i]
    return avg_epoch_int / len(avg_epoch_time)


def confusion_prepare(pl):  # converts pred + acc arrays of arrays with tensors to array
    pl_array = []
    for i in range(len(pl)):
        k = pl[i]
        for j in range(len(k)):
            pl_array.append(k[j])
    return pl_array


def plot_confusion_matrix(cm, num_epochs, epoch_time, title, test_acc):  # plots confusion matrix as heatmap
    plt.subplot(132)
    cm_1 = sns.heatmap(cm, linewidths=1, cmap='plasma')
    props = dict(boxstyle='round', facecolor='grey', alpha=0.5)
    text = 'num epochs: {}, '.format(num_epochs) + 'num params: {}, '.format(num_param) + \
           'batch size: 128, ' + 'lr: 0.01, ' + '\n' + \
           'avg time per epoch: {:.0f}m {:.0f}s, '.format(epoch_time // 60, epoch_time % 60) + \
           'test accuracy: {:4f}, '.format(test_acc) + 'dataset: SVHN'
    plt.text(12, 5, text, bbox=props)
    cm_1.figure.savefig(title)


def summary_plot(acc_x_vals, train_acc_y_vals, val_acc_y_vals, test_acc_y_vals):  # train and val accuracy plot
    plt.figure(figsize=(40, 10))
    plt.subplot(131)
    plt.plot(acc_x_vals, train_acc_y_vals)
    plt.plot(acc_x_vals, val_acc_y_vals)
    plt.plot(acc_x_vals, test_acc_y_vals)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Train Accuracy', 'Validation Accuracy', 'Test Accuracy'])

    plt.savefig('test_plot.svg')


def final_plot():
    plt.savefig('summary_plot_4_SVHN.svg')


def layer_plot():  # plots Rational Activation functions in all layers
    model_tr = torch.load('./model2.pth', map_location=torch.device('cpu')).to('cpu')
    rational_0 = model_tr.relu.cpu()
    rational_1 = model_tr.layer1.__getitem__(0).relu.cpu()
    rational_2 = model_tr.layer1.__getitem__(1).relu.cpu()
    rational_3 = model_tr.layer2.__getitem__(0).relu.cpu()
    rational_4 = model_tr.layer2.__getitem__(1).relu.cpu()
    rational_5 = model_tr.layer3.__getitem__(0).relu.cpu()
    rational_6 = model_tr.layer3.__getitem__(1).relu.cpu()
    rational_7 = model_tr.layer4.__getitem__(0).relu.cpu()
    rational_8 = model_tr.layer4.__getitem__(1).relu.cpu()
    plot = rational_0.show(display=False)
    plot = rational_1.show(display=False)
    plot = rational_2.show(display=False)
    plot = rational_3.show(display=False)
    plot = rational_4.show(display=False)
    plot = rational_5.show(display=False)
    plot = rational_6.show(display=False)
    plot = rational_7.show(display=False)
    plot = rational_8.show(display=False)
    plot.legend(['Layer 0 Rational Function', 'Layer 1 Rational Function 0', 'Layer 1 Rational Function 1',
                 'Layer 2 Rational Function 0',
                 'Layer 2 Rational Function 1', 'Layer 3 Rational Function 0', 'Layer 3 Rational Function 1',
                 'Layer 4 Rational Function 0', 'Layer 4 Rational Function 1'])
    plot.savefig('resnet_plots_3.svg')


RRN18_model = rational_resnet18(pretrained=False)
num_ftrs = RRN18_model.fc.in_features
# Here the size of each output sample is set to nn.Linear(num_ftrs, len(class_names)).
class_names = trainset.labels
RRN18_model.fc = nn.Linear(num_ftrs, len(class_names))

RRN18_model = RRN18_model.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.SGD(RRN18_model.parameters(), lr=0.01, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

num_param = RRN18_model.parameters().__sizeof__()

RRN18_model = train_val_test_model(RRN18_model, criterion, optimizer, exp_lr_scheduler,
                                   num_epochs=25)

final_plot()

torch.save(RRN18_model, './model7.pth')

"""def load_plot_loss(args, save_multi=False):
    # Load the root path of the trained setting
    save_path = get_save_path(args)
    # Load all Tensor Board event files
    events_files_list = glob.glob(save_path + "/events.out.tfevents.*")
    # Create the dictionary for the saving the loss values
    loss_ = {}
    for file_name in events_files_list:
        ea = event_accumulator.EventAccumulator(
            file_name,
            size_guidance={
                event_accumulator.COMPRESSED_HISTOGRAMS: 0,
                event_accumulator.IMAGES: 0,
                event_accumulator.AUDIO: 0,
                event_accumulator.SCALARS: 0,
                event_accumulator.HISTOGRAMS: 0,
            },
        )
        ea.Reload()
        for tag in ea.Tags()["scalars"]:
            tmp = [scalar.__getnewargs__()[-1] for scalar in ea.Scalars(tag)]
            if tag not in loss_.keys():
                loss_[tag] = tmp
            else:
                loss_[tag].extend(tmp)
    # Create a list for each sub-loss function. Each of will entail the
    # values for all three data set parts: training, validation and testing.
    reconstruction = []
    x_z = []
    z_x = []
    regularization = []
    total = []
    for i, key in enumerate(loss_.keys()):
        if key.find("reconstruction") != -1:
            reconstruction.append(loss_[key])
        elif key.find("x_z") != -1:
            x_z.append(loss_[key])
        elif key.find("z_x") != -1:
            z_x.append(loss_[key])
        elif key.find("regularization") != -1:
            regularization.append(loss_[key])
        elif key.find("total") != -1:
            total.append(loss_[key])
    # Pass the lists to the plotting routine...
    plot_loss(reconstruction, x_z, z_x, regularization, total, args, save_multi)"""
