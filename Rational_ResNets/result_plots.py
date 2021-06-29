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

# ResNet20
relu = pd.read_csv('./CSV/relu_resnet20/final_plot.csv')

univ = relu_data_ns = pd.read_csv(
    './CSV/univ_rational_resnet20/final_plot.csv')

mix = pd.read_csv(
    './CSV/mix_experts_resnet20/final_plot.csv')

mix_grad = pd.read_csv(
    './CSV/mix_experts_resnet20/final_plot_gradient.csv')

x_vals = relu['Epoch']

train_accs = [relu['Train Accuracy'], univ['Train Accuracy'], mix['Train Accuracy'], mix_grad['Train Accuracy']]
val_accs = [relu['Validation Accuracy'], univ['Validation Accuracy'], mix['Validation Accuracy'], mix_grad['Validation Accuracy']]
test_accs = [relu['Test Accuracy'], univ['Test Accuracy'], mix['Test Accuracy'], mix_grad['Test Accuracy']]

plt.figure(figsize=(15, 6))
plt.subplots_adjust(bottom=0.3)



def forward(x):
    return 1.04 ** x


def inverse(x):
    return np.log(x)


plt.subplot(131)
for i in range(len(train_accs)):
    sns.lineplot(x=x_vals, y=train_accs[i])
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy in Percent')

plt.subplot(132)
for i in range(len(val_accs)):
    sns.lineplot(x_vals, val_accs[i])
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy in Percent')

plt.subplot(133)
for i in range(len(test_accs)):
    sns.lineplot(x_vals, test_accs[i])
plt.xlabel('Epochs')
plt.ylabel('Test Accuracy in Percent')

plt.figlegend(['ReLU ResNet20', 'Univariate Rational ResNet20', 'Mixture of Experts ResNet20',
               'Mixture of Experts with Gradient Clipping'],
              bbox_to_anchor=(0, -0.776, 1, 1), loc='upper center', ncol=4, bbox_transform=plt.gcf().transFigure)

PATH = './Plots/all' + '/' + 'ResNet20_grad' + '.svg'
plt.savefig(PATH)
plt.show()

plt.figure(figsize=(15, 6))
plt.subplots_adjust(bottom=0.3)

plt.subplot(131)
for i in range(len(train_accs)):
    sns.lineplot(x=x_vals, y=train_accs[i])
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy in Percent')

ax = plt.gca()
ax.set_yscale('function', functions=(forward, inverse))

plt.subplot(132)
for i in range(len(val_accs)):
    sns.lineplot(x_vals, val_accs[i])
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy in Percent')

ax = plt.gca()
ax.set_yscale('function', functions=(forward, inverse))

plt.subplot(133)
for i in range(len(test_accs)):
    sns.lineplot(x_vals, test_accs[i])
plt.xlabel('Epochs')
plt.ylabel('Test Accuracy in Percent')

ax = plt.gca()
ax.set_yscale('function', functions=(forward, inverse))

plt.figlegend(['ReLU ResNet20 ', 'Univariate Rational ResNet20', 'Mixture of Experts ResNet20', 'Mixture of Experts with Gradient Clipping'],
              bbox_to_anchor=(0, -0.776, 1, 1), loc='upper center', ncol=4, bbox_transform=plt.gcf().transFigure)

PATH = './Plots/all' + '/' + 'ResNet20_exp_grad' + '.svg'
plt.savefig(PATH)
plt.show()


# ResNet14 A
relu = pd.read_csv('./CSV/relu_resnet14_A/final_plot.csv')

univ = relu_data_ns = pd.read_csv(
    './CSV/univ_rational_resnet14_A/final_plot.csv')

mix = pd.read_csv(
    './CSV/mix_experts_resnet14_A/final_plot.csv')

mix_grad = pd.read_csv(
    './CSV/mix_experts_resnet14_A/final_plot_gradient.csv')

x_vals = relu['Epoch']
train_accs = [relu['Train Accuracy'], univ['Train Accuracy'], mix['Train Accuracy'], mix_grad['Train Accuracy']]
val_accs = [relu['Validation Accuracy'], univ['Validation Accuracy'], mix['Validation Accuracy'], mix_grad['Validation Accuracy']]
test_accs = [relu['Test Accuracy'], univ['Test Accuracy'], mix['Test Accuracy'], mix_grad['Test Accuracy']]

plt.figure(figsize=(15, 6))
plt.subplots_adjust(bottom=0.3)


plt.subplot(131)
for i in range(len(train_accs)):
    sns.lineplot(x=x_vals, y=train_accs[i])
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy in Percent')

plt.subplot(132)
for i in range(len(val_accs)):
    sns.lineplot(x_vals, val_accs[i])
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy in Percent')

plt.subplot(133)
for i in range(len(test_accs)):
    sns.lineplot(x_vals, test_accs[i])
plt.xlabel('Epochs')
plt.ylabel('Test Accuracy in Percent')

plt.figlegend(['ReLU ResNet14 A', 'Univariate Rational ResNet14 A', 'Mixture of Experts ResNet14 A',
               'Mixture of Experts with Gradient Clipping'],
              bbox_to_anchor=(0, -0.776, 1, 1), loc='upper center', ncol=4, bbox_transform=plt.gcf().transFigure)

PATH = './Plots/all' + '/' + 'ResNet14_A_grad' + '.svg'
plt.savefig(PATH)
plt.show()

plt.figure(figsize=(15, 6))
plt.subplots_adjust(bottom=0.3)

plt.subplot(131)
for i in range(len(train_accs)):
    sns.lineplot(x=x_vals, y=train_accs[i])
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy in Percent')

ax = plt.gca()
ax.set_yscale('function', functions=(forward, inverse))

plt.subplot(132)
for i in range(len(val_accs)):
    sns.lineplot(x_vals, val_accs[i])
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy in Percent')

ax = plt.gca()
ax.set_yscale('function', functions=(forward, inverse))

plt.subplot(133)
for i in range(len(test_accs)):
    sns.lineplot(x_vals, test_accs[i])
plt.xlabel('Epochs')
plt.ylabel('Test Accuracy in Percent')

ax = plt.gca()
ax.set_yscale('function', functions=(forward, inverse))

plt.figlegend(['ReLU ResNet14 A', 'Univariate Rational ResNet14 A', 'Mixture of Experts ResNet14 A',
               'Mixture of Experts with Gradient Clipping'],
              bbox_to_anchor=(0, -0.776, 1, 1), loc='upper center', ncol=4, bbox_transform=plt.gcf().transFigure)

PATH = './Plots/all' + '/' + 'ResNet14_A_exp_grad' + '.svg'
plt.savefig(PATH)
plt.show()


# ResNet14 B
relu = pd.read_csv('./CSV/relu_resnet14_B/final_plot.csv')

univ = relu_data_ns = pd.read_csv(
    './CSV/univ_rational_resnet14_B/final_plot.csv')

mix = pd.read_csv(
    './CSV/mix_experts_resnet14_B/final_plot.csv')

mix_grad = pd.read_csv(
    './CSV/mix_experts_resnet14_B/final_plot_gradient.csv')

x_vals = relu['Epoch']
train_accs = [relu['Train Accuracy'], univ['Train Accuracy'], mix['Train Accuracy'], mix_grad['Train Accuracy']]
val_accs = [relu['Validation Accuracy'], univ['Validation Accuracy'], mix['Validation Accuracy'], mix_grad['Validation Accuracy']]
test_accs = [relu['Test Accuracy'], univ['Test Accuracy'], mix['Test Accuracy'], mix_grad['Test Accuracy']]

plt.figure(figsize=(15, 6))
plt.subplots_adjust(bottom=0.3)


plt.subplot(131)
for i in range(len(train_accs)):
    sns.lineplot(x=x_vals, y=train_accs[i])
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy in Percent')

plt.subplot(132)
for i in range(len(val_accs)):
    sns.lineplot(x_vals, val_accs[i])
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy in Percent')

plt.subplot(133)
for i in range(len(test_accs)):
    sns.lineplot(x_vals, test_accs[i])
plt.xlabel('Epochs')
plt.ylabel('Test Accuracy in Percent')

plt.figlegend(['ReLU ResNet14 B', 'Univariate Rational ResNet14 B', 'Mixture of Experts ResNet14 B',
               'Mixture of Experts with Gradient Clipping'],
              bbox_to_anchor=(0, -0.776, 1, 1), loc='upper center', ncol=4, bbox_transform=plt.gcf().transFigure)

PATH = './Plots/all' + '/' + 'ResNet14_B_grad' + '.svg'
plt.savefig(PATH)
plt.show()

plt.figure(figsize=(15, 6))
plt.subplots_adjust(bottom=0.3)

plt.subplot(131)
for i in range(len(train_accs)):
    sns.lineplot(x=x_vals, y=train_accs[i])
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy in Percent')

ax = plt.gca()
ax.set_yscale('function', functions=(forward, inverse))

plt.subplot(132)
for i in range(len(val_accs)):
    sns.lineplot(x_vals, val_accs[i])
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy in Percent')

ax = plt.gca()
ax.set_yscale('function', functions=(forward, inverse))

plt.subplot(133)
for i in range(len(test_accs)):
    sns.lineplot(x_vals, test_accs[i])
plt.xlabel('Epochs')
plt.ylabel('Test Accuracy in Percent')

ax = plt.gca()
ax.set_yscale('function', functions=(forward, inverse))

plt.figlegend(['ReLU ResNet14 B', 'Univariate Rational ResNet14 B', 'Mixture of Experts ResNet14 B',
               'Mixture of Experts with Gradient Clipping'],
              bbox_to_anchor=(0, -0.776, 1, 1), loc='upper center', ncol=4, bbox_transform=plt.gcf().transFigure)

PATH = './Plots/all' + '/' + 'ResNet14_B_exp_grad' + '.svg'
plt.savefig(PATH)
plt.show()

# ResNet8
relu = pd.read_csv('./CSV/relu_resnet8/final_plot.csv')

univ = relu_data_ns = pd.read_csv(
    './CSV/univ_rational_resnet8/final_plot.csv')

mix = pd.read_csv(
    './CSV/mix_experts_resnet8/final_plot.csv')

mix_grad = pd.read_csv(
    './CSV/mix_experts_resnet8/final_plot_gradient.csv')
x_vals = relu['Epoch']
train_accs = [relu['Train Accuracy'], univ['Train Accuracy'], mix['Train Accuracy'], mix_grad['Train Accuracy']]
val_accs = [relu['Validation Accuracy'], univ['Validation Accuracy'], mix['Validation Accuracy'], mix_grad['Validation Accuracy']]
test_accs = [relu['Test Accuracy'], univ['Test Accuracy'], mix['Test Accuracy'], mix_grad['Test Accuracy']]

plt.figure(figsize=(15, 6))
plt.subplots_adjust(bottom=0.3)

plt.subplot(131)
for i in range(len(train_accs)):
    sns.lineplot(x=x_vals, y=train_accs[i])
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy in Percent')

plt.subplot(132)
for i in range(len(val_accs)):
    sns.lineplot(x_vals, val_accs[i])
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy in Percent')

plt.subplot(133)
for i in range(len(test_accs)):
    sns.lineplot(x_vals, test_accs[i])
plt.xlabel('Epochs')
plt.ylabel('Test Accuracy in Percent')

plt.figlegend(['ReLU ResNet8', 'Univariate Rational ResNet8', 'Mixture of Experts ResNet8',
               'Mixture of Experts with Gradient Clipping'],
              bbox_to_anchor=(0, -0.776, 1, 1), loc='upper center', ncol=4, bbox_transform=plt.gcf().transFigure)

PATH = './Plots/all' + '/' + 'ResNet8_grad' + '.svg'
plt.savefig(PATH)
plt.show()

plt.figure(figsize=(15, 6))
plt.subplots_adjust(bottom=0.3)

plt.subplot(131)
for i in range(len(train_accs)):
    sns.lineplot(x=x_vals, y=train_accs[i])
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy in Percent')

ax = plt.gca()
ax.set_yscale('function', functions=(forward, inverse))

plt.subplot(132)
for i in range(len(val_accs)):
    sns.lineplot(x_vals, val_accs[i])
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy in Percent')

ax = plt.gca()
ax.set_yscale('function', functions=(forward, inverse))

plt.subplot(133)
for i in range(len(test_accs)):
    sns.lineplot(x_vals, test_accs[i])
plt.xlabel('Epochs')
plt.ylabel('Test Accuracy in Percent')

ax = plt.gca()
ax.set_yscale('function', functions=(forward, inverse))

plt.figlegend(['ReLU ResNet8', 'Univariate Rational ResNet8', 'Mixture of Experts ResNet8',
               'Mixture of Experts with Gradient Clipping'],
              bbox_to_anchor=(0, -0.776, 1, 1), loc='upper center', ncol=4, bbox_transform=plt.gcf().transFigure)

PATH = './Plots/all' + '/' + 'ResNet8_exp_grad' + '.svg'
plt.savefig(PATH)
plt.show()

# ResNet8 no data augmentation
relu = pd.read_csv('./CSV/relu_resnet8/final_plot_no_aug.csv')

univ = relu_data_ns = pd.read_csv(
    './CSV/univ_rational_resnet8/final_plot_no_aug.csv')

mix = pd.read_csv(
    './CSV/mix_experts_resnet8/final_plot_no_aug.csv')


mix_grad = pd.read_csv(
    './CSV/mix_experts_resnet8/final_plot_no_aug_grad.csv')
x_vals = relu['Epoch']
train_accs = [relu['Train Accuracy'], univ['Train Accuracy'], mix['Train Accuracy'], mix_grad['Train Accuracy']]
val_accs = [relu['Validation Accuracy'], univ['Validation Accuracy'], mix['Validation Accuracy'], mix_grad['Validation Accuracy']]
test_accs = [relu['Test Accuracy'], univ['Test Accuracy'], mix['Test Accuracy'], mix_grad['Test Accuracy']]

plt.figure(figsize=(15, 6))
plt.subplots_adjust(bottom=0.3)

plt.subplot(131)
for i in range(len(train_accs)):
    sns.lineplot(x=x_vals, y=train_accs[i])
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy in Percent')

plt.subplot(132)
for i in range(len(val_accs)):
    sns.lineplot(x_vals, val_accs[i])
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy in Percent')

plt.subplot(133)
for i in range(len(test_accs)):
    sns.lineplot(x_vals, test_accs[i])
plt.xlabel('Epochs')
plt.ylabel('Test Accuracy in Percent')

plt.figlegend(['ReLU ResNet8', 'Univariate Rational ResNet8', 'Mixture of Experts ResNet8',
               'Mixture of Experts with Gradient Clipping'],
              bbox_to_anchor=(0, -0.776, 1, 1), loc='upper center', ncol=4, bbox_transform=plt.gcf().transFigure)

PATH = './Plots/all' + '/' + 'ResNet8_grad_no_aug' + '.svg'
plt.savefig(PATH)
plt.show()

plt.figure(figsize=(15, 6))
plt.subplots_adjust(bottom=0.3)

plt.subplot(131)
for i in range(len(train_accs)):
    sns.lineplot(x=x_vals, y=train_accs[i])
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy in Percent')

ax = plt.gca()
ax.set_yscale('function', functions=(forward, inverse))

plt.subplot(132)
for i in range(len(val_accs)):
    sns.lineplot(x_vals, val_accs[i])
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy in Percent')

ax = plt.gca()
ax.set_yscale('function', functions=(forward, inverse))

plt.subplot(133)
for i in range(len(test_accs)):
    sns.lineplot(x_vals, test_accs[i])
plt.xlabel('Epochs')
plt.ylabel('Test Accuracy in Percent')

ax = plt.gca()
ax.set_yscale('function', functions=(forward, inverse))

plt.figlegend(['ReLU ResNet8', 'Univariate Rational ResNet8', 'Mixture of Experts ResNet8',
               'Mixture of Experts with Gradient Clipping'],
              bbox_to_anchor=(0, -0.776, 1, 1), loc='upper center', ncol=4, bbox_transform=plt.gcf().transFigure)

PATH = './Plots/all' + '/' + 'ResNet8_exp_grad_no_aug' + '.svg'
plt.savefig(PATH)
plt.show()
