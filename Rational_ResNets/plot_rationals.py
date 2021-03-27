import torch
import matplotlib.pyplot as plt
from rational.torch import Rational


def layer_plot():  # plots Rational Activation functions in all layers
    plot: plt
    model_tr = torch.load('./Saved_Models_wo_rationals/mv_rational_resnet_no_data_aug.pth')
    # Layer 0
    rational_0_0 = model_tr.r_rational
    rational_0_1 = model_tr.g_rational
    rational_0_2 = model_tr.b_rational
    rational_0_3 = model_tr.c_rational

    # Layer 1
    rational_1_0 = model_tr.layer1.__getitem__(0).r_rational
    rational_1_1 = model_tr.layer1.__getitem__(0).g_rational
    rational_1_2 = model_tr.layer1.__getitem__(0).b_rational
    rational_1_3 = model_tr.layer1.__getitem__(0).c_rational

    rational_2_0 = model_tr.layer1.__getitem__(1).r_rational
    rational_2_1 = model_tr.layer1.__getitem__(1).g_rational
    rational_2_2 = model_tr.layer1.__getitem__(1).b_rational
    rational_2_3 = model_tr.layer1.__getitem__(1).c_rational
    
    rational_3_0 = model_tr.layer1.__getitem__(2).r_rational
    rational_3_1 = model_tr.layer1.__getitem__(2).g_rational
    rational_3_2 = model_tr.layer1.__getitem__(2).b_rational
    rational_3_3 = model_tr.layer1.__getitem__(2).c_rational

    # Layer 2
    rational_4_0 = model_tr.layer2.__getitem__(0).r_rational
    rational_4_1 = model_tr.layer2.__getitem__(0).g_rational
    rational_4_2 = model_tr.layer2.__getitem__(0).b_rational
    rational_4_3 = model_tr.layer2.__getitem__(0).c_rational

    rational_5_0 = model_tr.layer2.__getitem__(1).r_rational
    rational_5_1 = model_tr.layer2.__getitem__(1).g_rational
    rational_5_2 = model_tr.layer2.__getitem__(1).b_rational
    rational_5_3 = model_tr.layer2.__getitem__(1).c_rational

    rational_6_0 = model_tr.layer2.__getitem__(2).r_rational
    rational_6_1 = model_tr.layer2.__getitem__(2).g_rational
    rational_6_2 = model_tr.layer2.__getitem__(2).b_rational
    rational_6_3 = model_tr.layer2.__getitem__(2).c_rational

    # Layer 3
    rational_7_0 = model_tr.layer3.__getitem__(0).r_rational
    rational_7_1 = model_tr.layer3.__getitem__(0).g_rational
    rational_7_2 = model_tr.layer3.__getitem__(0).b_rational
    rational_7_3 = model_tr.layer3.__getitem__(0).c_rational

    rational_8_0 = model_tr.layer3.__getitem__(1).r_rational
    rational_8_1 = model_tr.layer3.__getitem__(1).g_rational
    rational_8_2 = model_tr.layer3.__getitem__(1).b_rational
    rational_8_3 = model_tr.layer3.__getitem__(1).c_rational

    rational_9_0 = model_tr.layer3.__getitem__(2).r_rational
    rational_9_1 = model_tr.layer3.__getitem__(2).g_rational
    rational_9_2 = model_tr.layer3.__getitem__(2).b_rational
    rational_9_3 = model_tr.layer3.__getitem__(2).c_rational

    plot = rational_0_0.show(display=False)
    plot = rational_0_1.show(display=False)
    plot = rational_0_2.show(display=False)
    plot = rational_0_3.show(display=False)

    plot.legend(['Layer 0 Rational Function', 'Layer 1 Rational Function 0', 'Layer 1 Rational Function 1',
                 'Layer 2 Rational Function 0',
                 'Layer 2 Rational Function 1', 'Layer 3 Rational Function 0', 'Layer 3 Rational Function 1',
                 'Layer 4 Rational Function 0', 'Layer 4 Rational Function 1'])
    plot.savefig('resnet_plots_3.svg')
    plot.show()

layer_plot()