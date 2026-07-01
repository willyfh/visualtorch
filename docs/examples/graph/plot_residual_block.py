"""Residual Block
=======================================

Visualization of a classic ResNet-style residual block: Conv2d + BatchNorm2d, twice, with a
plain identity shortcut around them and a final ReLU.

Conv2d is orange, BatchNorm2d is green, and ReLU is salmon.
"""  # noqa: D205

from collections import defaultdict

import matplotlib.pyplot as plt
import torch
import visualtorch
from torch import nn


class ResidualBlock(nn.Module):
    """A classic ResNet-style block with a plain identity shortcut."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the forward pass, with a skip connection around conv1/bn1/relu/conv2/bn2."""
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


model = ResidualBlock(channels=8)

input_shape = (1, 8, 16, 16)

color_map: dict = defaultdict(dict)
color_map[nn.Conv2d]["fill"] = "#FFE4B5"
color_map[nn.BatchNorm2d]["fill"] = "#98FB98"
color_map[nn.ReLU]["fill"] = "#FFA07A"

img = visualtorch.graph_view(model, input_shape, show_neurons=False, color_map=color_map, layer_spacing=60)

plt.axis("off")
plt.tight_layout()
plt.imshow(img)
plt.show()
