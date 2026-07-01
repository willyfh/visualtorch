"""Residual Block
=======================================

Visualization of a real ResNet-style residual block: two Conv2d+BatchNorm2d branches (a main
path and a 1x1 "downsample" shortcut) that converge and pass through a final ReLU. graph_view
previously only recognized nn.Linear and nn.Conv2d and had no way to represent branching at
all - this is exactly the kind of architecture that became possible once that was fixed.

Conv2d is orange, BatchNorm2d is green, and ReLU is salmon.
"""  # noqa: D205

from collections import defaultdict

import matplotlib.pyplot as plt
import torch
import visualtorch
from torch import nn


class ResidualBlock(nn.Module):
    """A ResNet-style block with a projection ("downsample") shortcut for the skip connection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.downsample_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the forward pass, with a skip connection around conv1/bn1/relu/conv2/bn2."""
        identity = self.downsample_bn(self.downsample_conv(x))
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


model = ResidualBlock(in_channels=4, out_channels=8)

input_shape = (1, 4, 16, 16)

color_map: dict = defaultdict(dict)
color_map[nn.Conv2d]["fill"] = "#FFE4B5"
color_map[nn.BatchNorm2d]["fill"] = "#98FB98"
color_map[nn.ReLU]["fill"] = "#FFA07A"

img = visualtorch.graph_view(model, input_shape, show_neurons=False, color_map=color_map)

plt.axis("off")
plt.tight_layout()
plt.imshow(img)
plt.show()
