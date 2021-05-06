import torch
from rational.torch import Rational
import matplotlib
import matplotlib.pyplot as plt

means = torch.ones((50, 50)) * 2.
stds = torch.ones((50, 50)) * 3.
rational_function = Rational()
rational_function.input_retrieve_mode()
for _ in range(1500):
    input = torch.normal(means, stds).to(rational_function.device)
    rational_function(input)

print(rational_function.show(display=False))
bins = rational_function.show(display=False)['hist']['bins']
freq = rational_function.show(display=False)['hist']['freq']
ax = plt.gca()
ax2 = ax.twinx()
ax2.set_yticks([])
grey_color = (0.5, 0.5, 0.5, 0.6)
ax2.bar(bins, freq, width=bins[1] - bins[0], color=grey_color, edgecolor=grey_color)
ax.plot(rational_function.show(display=False)['line']['x'], rational_function.show(display=False)['line']['y'])
plt.show()
rational_function.show()