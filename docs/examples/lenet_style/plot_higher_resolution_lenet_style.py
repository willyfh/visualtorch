"""Higher Resolution
=======================================

``lenet_view``'s box size is mainly controlled by ``scale_xy`` (each stacked plane's spatial
width/height). Unlike ``flow_view``, its ``scale_z`` already defaults to ``1`` rather than
``0.1``, so the per-slice depth is rarely the bottleneck here - ``scale_xy`` is the lever that
actually makes each plane bigger and easier to read. ``scale_z`` is instead capped by
``max_channels`` (default ``100``): a layer with more channels than that gets its slice count
clamped, not scaled further.

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

dpi = 150  # rendered at 2x this in the final doc build (savefig.dpi=300 in conf.py)

# %%
# Default
# -------
# At the default ``scale_xy=1``, each stacked plane is fairly small.

img_default = visualtorch.render(model, input_shape, style="lenet", color_map=color_map)

plt.figure(figsize=(img_default.width / dpi, img_default.height / dpi), dpi=dpi)
plt.imshow(img_default)
plt.axis("off")
plt.tight_layout()
plt.show()

# %%
# Higher Resolution
# -----------------
# Raising ``scale_xy`` grows each plane's spatial extent for a clearer, more substantial result.

img_hires = visualtorch.render(model, input_shape, style="lenet", color_map=color_map, scale_xy=3)

plt.figure(figsize=(img_hires.width / dpi, img_hires.height / dpi), dpi=dpi)
plt.imshow(img_hires)
plt.axis("off")
plt.tight_layout()
plt.show()
