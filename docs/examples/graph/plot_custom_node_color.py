"""Custom Color
=======================================

Visualization of custom color
"""  # noqa: D205

from collections import defaultdict

import matplotlib.pyplot as plt
import torch
import visualtorch
from torch import nn


class SimpleDense(nn.Module):
    """Simple Dense Model."""

    def __init__(self) -> None:
        super().__init__()
        self.h0 = nn.Linear(4, 8)
        self.h1 = nn.Linear(8, 8)
        self.h2 = nn.Linear(8, 4)
        self.out = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the forward pass."""
        x = self.h0(x)
        x = self.h1(x)
        x = self.h2(x)
        return self.out(x)


model = SimpleDense()

input_shape = (1, 4)

color_map: dict = defaultdict(dict)
color_map[nn.Linear]["fill"] = "#009E73"  # bluish green

img = visualtorch.render(model, input_shape, style="graph", color_map=color_map)

dpi = 150  # rendered at 2x this in the final doc build (savefig.dpi=300 in conf.py)
plt.figure(figsize=(img.width / dpi, img.height / dpi), dpi=dpi)
plt.imshow(img)
plt.axis("off")
plt.tight_layout()
plt.show()
