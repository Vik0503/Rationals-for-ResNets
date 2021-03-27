from __future__ import print_function, division

import time
from typing import Type, Any, List

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from rational.torch import Rational
from sklearn.metrics import confusion_matrix
from torch import Tensor
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from LTH_for_Rational_ResNets.Datasets import SVHN
from LTH_for_Rational_ResNets.Mask import Mask

if torch.cuda.is_available():
    cuda = True
    device = 'cuda'
else:
    cuda = False
    device = 'cpu'


class RationalBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, planes_in, planes_out, stride=1, downsample=False):
        super(RationalBasicBlock, self).__init__()

        self.conv_layer_1 = nn.Conv2d(planes_in, planes_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch_norm_1 = nn.BatchNorm2d(planes_out)
        # use Rationals instead of reLu activation function
        self.rational = Rational(cuda=cuda)
        self.conv_layer_2 = nn.Conv2d(planes_out, planes_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm_2 = nn.BatchNorm2d(planes_out)

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
        out = multi_variant_rationals(out)
        out = self.conv_layer_2(out)
        out = self.batch_norm_2(out)
        out += self.shortcut(x)
        out = multi_variant_rationals(out)

        return out


def weights_init(model):
    """
    Initialize weights of model.
    Parameters
    ----------
    model: Model
    """
    if isinstance(model, nn.Conv2d):
        nn.init.kaiming_normal_(model.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(model, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(model.weight, 1)
        nn.init.constant_(model.bias, 0)


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

    def __init__(self,
                 block: Type[RationalBasicBlock],
                 layers: List[int],
                 num_classes: int = 10,
                 mask: Mask = None,
                 ) -> None:
        super(RationalResNet, self).__init__()

        self.norm_layer = nn.BatchNorm2d

        self.planes_in = 16

        self.conv_layer_1 = nn.Conv2d(in_channels=3, out_channels=self.planes_in, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm_1 = self.norm_layer(self.planes_in)

        self.rational = Rational(cuda=cuda)

        self.layer1 = self.make_layer(block=block, planes_out=16, num_blocks=layers[0], stride=1)
        self.layer2 = self.make_layer(block=block, planes_out=32, num_blocks=layers[1], stride=2)
        self.layer3 = self.make_layer(block=block, planes_out=64, num_blocks=layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        self.apply(weights_init)
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

        layers = []
        layers.append(block(self.planes_in, planes_out, stride, downsample=downsample))

        downsample = False
        stride = 1
        self.planes_in = planes_out * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.planes_in, planes_out, stride, downsample=downsample))
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
        out = multi_variant_rationals(out)

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


def _resnet(
        arch: str,
        block: Type[RationalBasicBlock],
        layers: List[int],
        mask: Mask,

        **kwargs: Any
) -> RationalResNet:
    """
    The universal ResNet definition
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
    model = RationalResNet(block, layers, mask=mask, **kwargs)

    return model


def multi_variant_rational_resnet20(mask: Mask = None, **kwargs: Any) -> RationalResNet:
    """ResNet for CIFAR10 as mentioned in the paper above"""
    return _resnet('resnet20', RationalBasicBlock, [3, 3, 3], mask=mask, **kwargs)


def prunable_layer_dict(model):  # shortcuts???
    prune_dict = {}
    for name, param in model.named_parameters():
        if 'weight' not in name:
            continue

        prune_dict[name] = param

    return prune_dict


def multi_variant_rationals(out: Tensor) -> Tensor:
    rational = Rational(cuda=cuda)
    original_shape = out.shape
    img_shape = out.shape[2]
    num_planes = out.shape[1]
    batch_size = out.shape[0]
    rat_planes = int(num_planes/4)

    out = torch.flatten(out, start_dim=0, end_dim=1)
    a_tensor = rational(out[0::4].clone().reshape(batch_size, rat_planes, img_shape, img_shape))
    b_tensor = rational(out[1::4].clone().reshape(batch_size, rat_planes, img_shape, img_shape))
    c_tensor = rational(out[2::4].clone().reshape(batch_size, rat_planes, img_shape, img_shape))
    d_tensor = rational(out[3::4].clone().reshape(batch_size, rat_planes, img_shape, img_shape))

    a_tensor = torch.flatten(a_tensor, start_dim=0, end_dim=1)
    b_tensor = torch.flatten(b_tensor, start_dim=0, end_dim=1)
    c_tensor = torch.flatten(c_tensor, start_dim=0, end_dim=1)
    d_tensor = torch.flatten(d_tensor, start_dim=0, end_dim=1)

    out = out.reshape(batch_size * num_planes, img_shape, img_shape).to(device)
    counter = 0
    counter_2 = 0
    for i in range(out.shape[0]):
        if counter == 0:
            out[i] = a_tensor[counter_2].clone().detach()
            counter += 1
        elif counter == 1:
            out[i] = b_tensor[counter_2].clone().detach()
            counter += 1
        elif counter == 2:
            out[i] = c_tensor[counter_2].clone().detach()
            counter += 1
        elif counter == 3:
            out[i] = d_tensor[counter_2].clone().detach()
            counter_2 += 1
            counter = 0

    out = out.reshape(original_shape)
    return out


writer = SummaryWriter('runs/multi_variant_v4_rational_resnet18_SVHN_run10')

trainset = SVHN.get_trainset()
valset = SVHN.get_validationset()
testset = SVHN.get_testset()
trainloader = SVHN.get_trainloader(bs=128)
valloader = SVHN.get_valloader(bs=128)
testloader = SVHN.get_testloader(bs=128)
classes = SVHN.get_classes()
num_classes = SVHN.get_num_classes()


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
        print('Epoch {}/{}'.format(epoch, num_epochs- 1))
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

    summary_plot(accuracy_plot_x_vals, train_acc_plot_y_vals, val_acc_plot_y_vals, test_acc_plot_y_vals)

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
    text = 'num epochs: {}, '.format(num_epochs) + \
           'batch size: 128, ' + 'lr: 0.01, ' + '\n' + \
           'avg time per epoch: {:.0f}m {:.0f}s, '.format(epoch_time // 60, epoch_time % 60) + \
           'test accuracy: {:4f}, '.format(test_acc) + 'dataset: SVHN'
    plt.text(15, 5, text, bbox=props)
    # cm_1.figure.savefig(title)


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
    plt.savefig('summary_plot_multi_rational_resnet_SVHN_v4.svg')
    plt.show()


model = multi_variant_rational_resnet20().to(device)
num_ftrs = model.fc.in_features
# Here the size of each output sample is set to nn.Linear(num_ftrs, len(class_names)).
model.fc = nn.Linear(num_ftrs, num_classes)


criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


model = train_val_test_model(model, criterion, optimizer, exp_lr_scheduler,
                             num_epochs=50)

final_plot()

torch.save(model, '../../Saved_Models/model13.pth')
