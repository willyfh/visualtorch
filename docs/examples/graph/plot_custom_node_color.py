"""
Custom Color
=======================================

Visualization of custom color
"""

import torch.nn as nn
import visualtorch
import matplotlib.pyplot as plt
from collections import defaultdict


class SimpleDense(nn.Module):
    def __init__(self):
        super(SimpleDense, self).__init__()
        self.h0 = nn.Linear(4, 8)
        self.h1 = nn.Linear(8, 8)
        self.h2 = nn.Linear(8, 4)
        self.out = nn.Linear(4, 2)

    def forward(self, x):
        x = self.h0(x)
        x = self.h1(x)
        x = self.h2(x)
        x = self.out(x)
        return x


model = SimpleDense()

input_shape = (1, 4)

color_map: dict = defaultdict(dict)
color_map[nn.Linear]["fill"] = "#98FB98"

img = visualtorch.graph_view(model, input_shape, color_map=color_map)

plt.axis("off")
plt.tight_layout()
plt.imshow(img)
plt.show()