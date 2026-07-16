"""Skip-Connection Level Spacing
=======================================

``level_gap`` controls the vertical distance in pixels between overlapping skip-connection routes. This
nested residual model creates two detour levels: an inner shortcut around one convolution and an
outer shortcut around the whole block. Compare a compact setting with a more widely spaced one.
The additional rendering options separate the boxes and keep attention on the routed lines.
"""  # noqa: D205

import matplotlib.pyplot as plt
import torch
import visualtorch
from torch import nn


class NestedResidualBlock(nn.Module):
    """A residual block with one shortcut nested inside another."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.inner_relu = nn.ReLU()
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.outer_relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block with overlapping inner and outer shortcuts."""
        outer_identity = x
        x = self.conv1(x)
        inner_identity = x
        x = self.inner_relu(self.conv2(x) + inner_identity)
        x = self.conv3(x)
        return self.outer_relu(x + outer_identity)


model = NestedResidualBlock(channels=4)
input_shape = (1, 4, 16, 16)

level_gaps = (20, 80)
images = [
    visualtorch.render(
        model,
        input_shape,
        style="lenet",
        level_gap=level_gap,
        spacing=25,
        show_dimension=False,
        connector_width=2,
    )
    for level_gap in level_gaps
]

dpi = 150  # rendered at 2x this in the final doc build (savefig.dpi=300 in conf.py)
figure_width = min(12, max(8, sum(img.width for img in images) / dpi))
figure_height = min(4, max(3, max(img.height for img in images) / dpi))
_, axes = plt.subplots(1, 2, figsize=(figure_width, figure_height), dpi=dpi)
for ax, img, level_gap in zip(axes, images, level_gaps, strict=True):
    ax.imshow(img)
    ax.set_title(f"level_gap={level_gap}")
    ax.axis("off")

plt.tight_layout()
plt.show()
