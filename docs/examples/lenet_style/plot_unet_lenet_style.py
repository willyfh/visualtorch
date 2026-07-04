"""U-Net
=======================================

The same small U-Net as the ``graph``/``flow`` styles' examples, rendered in ``lenet`` style
instead - the contracting-then-expanding channel/spatial shape naturally produces the classic
U-Net silhouette.

Conv2d is orange, ConvTranspose2d is green, ReLU is sky blue, and MaxPool2d is reddish purple.
"""  # noqa: D205

from collections import defaultdict

import matplotlib.pyplot as plt
import torch
import visualtorch
from torch import nn


class UNet(nn.Module):
    """A small U-Net with 2 encoder/decoder stages and skip connections."""

    def __init__(self) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU())
        self.final = nn.Conv2d(16, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample through 2 encoder stages, then upsample back, concatenating each
        encoder stage's output into its corresponding decoder stage.
        """  # noqa: D205
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


model = UNet()

input_shape = (1, 3, 64, 64)

color_map: dict = defaultdict(dict)
color_map[nn.Conv2d]["fill"] = "#E69F00"
color_map[nn.ConvTranspose2d]["fill"] = "#009E73"
color_map[nn.ReLU]["fill"] = "#56B4E9"
color_map[nn.MaxPool2d]["fill"] = "#CC79A7"

img = visualtorch.render(model, input_shape, style="lenet", color_map=color_map, spacing=25)

dpi = 150  # rendered at 2x this in the final doc build (savefig.dpi=300 in conf.py)
plt.figure(figsize=(img.width / dpi, img.height / dpi), dpi=dpi)
plt.imshow(img)
plt.axis("off")
plt.tight_layout()
plt.show()
