import torch
import torchvision
from rational.torch import Rational
from torch import nn, Tensor
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
#  train_set = torchvision.datasets.SVHN(root='/home/viktoria/Git/thesis_stuff/data/SVHN', split='train', download=True, transform=transform)
train_set = torchvision.datasets.CIFAR10(root='/home/viktoria/Git/thesis_stuff/data/cifar10', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=16, drop_last=True)

if torch.cuda.is_available():
    device = 'cuda'
    cuda = True
else:
    device = 'cpu'
    cuda = False


class TestNet(nn.Module):
    def __init__(self, planes_in, planes_out, stride=1):
        super(TestNet, self).__init__()

        self.conv_layer_1 = nn.Conv2d(planes_in, planes_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch_norm_1 = nn.BatchNorm2d(planes_out)
        # use Rationals instead of reLu activation function
        self.rational = Rational(cuda=cuda)
        self.conv_layer_2 = nn.Conv2d(planes_out, planes_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm_2 = nn.BatchNorm2d(planes_out)
        self.planes_in = planes_in
        self.planes_out = planes_out

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
        shape = (self.planes_out, self.planes_in, 32, 32)  # eventuell planes_in direkt auf 3 setzen?
        neutral_tensor = torch.ones(shape).to(device)
        neutral_tensor = neutral_tensor.squeeze(dim=0)
        print('neutral shape: ', neutral_tensor.shape)
        out = x
        self = self.to(device)
        out = out.to(device)
        print('out shape: ', out.shape)
        out = self.conv_layer_1(out)
        # rgb_tensor = out.mul(neutral_tensor)
        out = self.batch_norm_1(out)
        print('norm shape: ', out.shape)
        out = out.squeeze(dim=0)
        out = out.repeat(3, 1, 1, 1).view(6, 3, 32, 32)  # statt 1 batch size?
        rgb_tensor = (out * neutral_tensor).sum(dim=0)
        rgb_tensor = rgb_tensor
        print(rgb_tensor.shape)
        r_tensor = rgb_tensor[0].clone().detach()
        g_tensor = rgb_tensor[1].clone().detach()
        b_tensor = rgb_tensor[2].clone().detach()
        r_out = self.rational(r_tensor)
        g_out = self.rational(g_tensor)
        b_out = self.rational(b_tensor)
        out = torch.ones(3, 32, 32)
        out[0] = r_out.clone().detach()
        out[1] = g_out.clone().detach()
        out[2] = b_out.clone().detach()

        print(out.shape)
        print(out)
        """out = self.conv_layer_2(out)
        out = self.batch_norm_2(out)
        out = self.rational(out)"""

        return out


testmodel = TestNet(planes_in=3, planes_out=6)
for it, data in enumerate(trainloader, 0):
    inputs, labels = data
    # print(inputs)
    inputs = inputs.to(device)
    labels = labels.to(device)
    test_tensor = torch.ones_like(inputs)
    test_tensor_zero = torch.zeros_like(inputs)
    print('input: ', test_tensor.shape)
    with torch.set_grad_enabled(True):
        outputs = testmodel(test_tensor)
        """outputs_2 = testmodel(inputs)
        print(inputs.shape)
        outputs_3 = testmodel(test_tensor_zero)
        print(test_tensor_zero.shape)"""
    break
