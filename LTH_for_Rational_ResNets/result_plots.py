import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use(["science", "grid"])
matplotlib.rcParams.update({
    "font.family": "serif",
    "text.usetex": False,
})

relu_data_ns = pd.read_csv('/Users/viktoriasartor/git/Rationals-for-ResNets/LTH_for_Rational_ResNets/CSV/' +
                           'relu_resnet20/test_accs.csv')
univ_rat_data_ns = pd.read_csv('/Users/viktoriasartor/git/Rationals-for-ResNets/LTH_for_Rational_ResNets/CSV/' +
                               'univ_rational_resnet20/test_accs.csv')
mix_exp_data_ns = pd.read_csv('/Users/viktoriasartor/git/Rationals-for-ResNets/LTH_for_Rational_ResNets/CSV/' +
                              'mix_experts_resnet20/test_accs.csv')

relu_data_s = pd.read_csv('/Users/viktoriasartor/git/Rationals-for-ResNets/LTH_for_Rational_ResNets/CSV/' +
                          'relu_resnet20/test_accs_shortcuts.csv')
univ_rat_data_s = pd.read_csv('/Users/viktoriasartor/git/Rationals-for-ResNets/LTH_for_Rational_ResNets/CSV/' +
                              'univ_rational_resnet20/test_accs_shortcuts.csv')
mix_exp_data_s = pd.read_csv('/Users/viktoriasartor/git/Rationals-for-ResNets/LTH_for_Rational_ResNets/CSV/' +
                             'mix_experts_resnet20/test_accs_shortcuts.csv')

test_accs_ns = [relu_data_ns['test_acc'], univ_rat_data_ns['test_acc'],
                mix_exp_data_ns['test_acc']]
test_accs_s = [relu_data_s['test_acc'], univ_rat_data_s['test_acc'],
               mix_exp_data_s['test_acc']]
x_vals_ns = [relu_data_ns['pruning_percentage'], univ_rat_data_ns['pruning_percentage'],
             mix_exp_data_ns['pruning_percentage']]
x_vals_s = [relu_data_s['pruning_percentage'], univ_rat_data_s['pruning_percentage'],
            mix_exp_data_s['pruning_percentage']]

plt.figure(figsize=(10, 6))

plt.subplots_adjust(bottom=0.3)
plt.subplot(121)
ax = plt.gca()
# plt.xticks(data['pruning_percentage'])
for i in range(len(test_accs_ns)):
    sns.lineplot(x=x_vals_ns[i], y=test_accs_ns[i])
plt.xlabel('Percentage of Pruned Weights')
plt.ylabel('Test Accuracy in Percent')

plt.subplot(122)
for i in range(len(test_accs_s)):
    sns.lineplot(x=x_vals_s[i], y=test_accs_s[i])
plt.xlabel('Percentage of Pruned Weights')
plt.ylabel('Test Accuracy in Percent')

ax = plt.gca()

plt.figlegend(['ReLU ResNet20', 'Univariate Rational ResNet20', 'Mixture of Experts ResNet20'],
              loc='upper center', bbox_to_anchor=(0.2, -0.375, 0.45, 0.5), ncol=3)
PATH = '/Users/viktoriasartor/git/Rationals-for-ResNets/LTH_for_Rational_ResNets/Plots/all.svg'
plt.savefig(PATH)
plt.show()

plt.figure(figsize=(10, 6))

plt.subplots_adjust(bottom=0.3)
plt.subplot(121)
# plt.xticks(data['pruning_percentage'])
for i in range(len(test_accs_ns)):
    sns.lineplot(x=x_vals_ns[i], y=test_accs_ns[i])
plt.xlabel('Percentage of Pruned Weights')
plt.ylabel('Test Accuracy in Percent')


def forward(x):
    return 1.1 ** x


def inverse(x):
    return np.log(x)


def forward_x(x):
    return 1.02 ** abs(x)


def inverse_x(x):
    return np.log(abs(x))


ax = plt.gca()
ax.set_xscale('function', functions=(forward_x, inverse_x))
plt.xticks([0, 20, 40, 60, 80, 100])
ax.set_yscale('function', functions=(forward, inverse))

plt.subplot(122)
for i in range(len(test_accs_s)):
    sns.lineplot(x=x_vals_s[i], y=test_accs_s[i])
plt.xlabel('Percentage of Pruned Weights')
plt.ylabel('Test Accuracy in Percent')

ax = plt.gca()
ax.set_xscale('function', functions=(forward_x, inverse_x))
plt.xticks([0, 20, 40, 60, 80, 100])
ax.set_yscale('function', functions=(forward, inverse))

plt.figlegend(['ReLU ResNet20', 'Univariate Rational ResNet20', 'Mixture of Experts ResNet20'],
              loc='upper center', bbox_to_anchor=(0.2, -0.375, 0.45, 0.5), ncol=3)
PATH = '/Users/viktoriasartor/git/Rationals-for-ResNets/LTH_for_Rational_ResNets/Plots/all_exp.svg'
plt.savefig(PATH)
plt.show()

plt.figure(figsize=(10, 6))

plt.subplots_adjust(bottom=0.3)
plt.subplot(121)
ax = plt.gca()
# plt.xticks(data['pruning_percentage'])
for i in range(len(test_accs_ns)):
    sns.lineplot(x=x_vals_ns[i], y=test_accs_ns[i])
plt.xlabel('Percentage of Pruned Weights')
plt.ylabel('Test Accuracy in Percent')


def forward(x):
    return 1.1 ** x


def inverse(x):
    return np.log(x)


def forward_x(x):
    return 1.2 ** abs(x)


def inverse_x(x):
    return np.log(abs(x))


ax = plt.gca()
ax.set_xscale('function', functions=(forward_x, inverse_x))
ax.set_yscale('function', functions=(forward, inverse))
ax.set_xlim(90, 100)

plt.subplot(122)
for i in range(len(test_accs_s)):
    sns.lineplot(x=x_vals_s[i], y=test_accs_s[i])
plt.xlabel('Percentage of Pruned Weights')
plt.ylabel('Test Accuracy in Percent')

ax = plt.gca()
ax.set_xlim(90, 100)
ax.set_xscale('function', functions=(forward_x, inverse_x))
ax.set_yscale('function', functions=(forward, inverse))


plt.figlegend(['ReLU ResNet20', 'Univariate Rational ResNet20', 'Mixture of Experts ResNet20'],
              loc='upper center', bbox_to_anchor=(0.2, -0.375, 0.45, 0.5), ncol=3)
PATH = '/Users/viktoriasartor/git/Rationals-for-ResNets/LTH_for_Rational_ResNets/Plots/small_exp.svg'
plt.savefig(PATH)
plt.show()
