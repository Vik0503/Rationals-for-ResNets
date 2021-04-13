from LTH_for_Rational_ResNets.LTH_Models import select_2_expert_groups_rational_resnet as sel
from LTH_for_Rational_ResNets.LTH_Models import rational_resnet20_cifar10 as rrn20
from LTH_for_Rational_ResNets import Mask
from LTH_for_Rational_ResNets import utils
from copy import deepcopy

model = sel.select_2_expert_groups_rational_resnet20(['leaky_relu', 'gelu', 'swish', 'tanh', 'sigmoid'], num_rationals=5)
# model = rrn20.rational_resnet20()
mask = Mask.make_initial_mask(model)
mask = mask.cuda()
model = model.to('cuda')
model.mask = mask
init_state = deepcopy(model.state_dict())
# sel.reinit(model, mask, init_state)
utils.reinit(model, mask, init_state)

