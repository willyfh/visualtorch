"""Dark Background
=======================================

``background_fill`` isn't limited to plain white - it also accepts a transparent color
(e.g. ``(0, 0, 0, 0)``), useful for dropping a figure onto a paper/slide without a white box
around it, or an opaque dark color for a nicer look on dark-mode pages.

Note: when using a non-white background, also set an ``outline`` per layer type in
``color_map``. Box borders and funnel connector lines default to plain black, which becomes
invisible against a dark or black background.
"""  # noqa: D205

from collections import defaultdict

import matplotlib.pyplot as plt
import visualtorch
from torch import nn

# Example of a simple CNN model using nn.Sequential
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
)

color_map: dict = defaultdict(dict)
color_map[nn.Conv2d]["fill"] = "#00F5FF"
color_map[nn.Conv2d]["outline"] = "#E0FFFF"
color_map[nn.BatchNorm2d]["fill"] = "#FF10F0"
color_map[nn.BatchNorm2d]["outline"] = "#FFD1FA"
color_map[nn.ReLU]["fill"] = "#FCEE09"
color_map[nn.ReLU]["outline"] = "#FFFACD"

input_shape = (1, 3, 32, 32)
img = visualtorch.render(
    model,
    input_shape=input_shape,
    style="flow",
    color_map=color_map,
    background_fill="black",
)

plt.axis("off")
plt.tight_layout()
plt.imshow(img)
plt.show()
