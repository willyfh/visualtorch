"""Custom Color
=======================================

Visualization of custom color
"""  # noqa: D205

from collections import defaultdict

import matplotlib.pyplot as plt
import visualtorch
from torch import nn

# Example of a simple CNN model using nn.Sequential
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(64 * 28 * 28, 256),  # Adjusted the input size for the Linear layer
    nn.ReLU(),
    nn.Linear(256, 10),  # Assuming 10 output classes
)

color_map: dict = defaultdict(dict)
color_map[nn.Conv2d]["fill"] = "#E69F00"  # orange
color_map[nn.ReLU]["fill"] = "#56B4E9"  # sky blue
color_map[nn.MaxPool2d]["fill"] = "#CC79A7"  # reddish purple
color_map[nn.Flatten]["fill"] = "#009E73"  # bluish green
color_map[nn.Linear]["fill"] = "#0072B2"  # blue

input_shape = (1, 3, 224, 224)
img = visualtorch.render(model, input_shape=input_shape, style="flow", color_map=color_map)

plt.axis("off")
plt.tight_layout()
plt.imshow(img)
plt.show()
