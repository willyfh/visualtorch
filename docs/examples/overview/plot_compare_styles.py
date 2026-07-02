"""Comparing Styles
=======================================

VisualTorch traces a model's real structure once, then renders that same structure three
different ways via ``style="graph"|"flow"|"lenet"`` - so a branching model like this
ResNet-style residual block (the same one used in each style's own residual-block example)
renders with a correctly routed skip connection in every style, not just one.

Conv2d is orange, BatchNorm2d is green, and ReLU is sky blue.
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
color_map[nn.Conv2d]["fill"] = "#E69F00"
color_map[nn.BatchNorm2d]["fill"] = "#009E73"
color_map[nn.ReLU]["fill"] = "#56B4E9"

img_graph = visualtorch.render(
    model,
    input_shape,
    style="graph",
    show_neurons=False,
    color_map=color_map,
    layer_spacing=60,
)
img_flow = visualtorch.render(model, input_shape, style="flow", color_map=color_map, scale_xy=3, spacing=20)
img_lenet = visualtorch.render(model, input_shape, style="lenet", color_map=color_map, scale_xy=1.5, padding=80)

fig, axes = plt.subplots(3, 1, figsize=(9, 9))
for ax, img, title in zip(
    axes,
    [img_graph, img_flow, img_lenet],
    ['style="graph"', 'style="flow"', 'style="lenet"'],
    strict=True,
):
    ax.imshow(img)
    ax.set_title(title, fontsize=11, loc="left")
    ax.axis("off")
plt.tight_layout()
plt.show()
