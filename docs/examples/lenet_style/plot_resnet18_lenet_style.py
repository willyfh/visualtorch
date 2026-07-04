"""ResNet-18
=======================================

The same real, torchvision-provided `resnet18` architecture as the ``graph``/``flow`` styles'
examples - 8 residual blocks across 4 stages, with a projection shortcut where channels/spatial
size change - rendered in ``lenet`` style instead.

Note: ``offset_z`` (the per-slice depth offset used for the stacked-plane 3D look) defaults to
``10``, tuned for small toy models - for a real model with channel counts in the hundreds, that
default multiplies into an enormous image, so it's lowered here to ``1``.

Conv2d is orange, BatchNorm2d is green, and ReLU is sky blue.
"""  # noqa: D205

from collections import defaultdict

import matplotlib.pyplot as plt
import visualtorch
from torch import nn
from torchvision.models import resnet18

model = resnet18(weights=None, num_classes=10)

input_shape = (1, 3, 64, 64)

color_map: dict = defaultdict(dict)
color_map[nn.Conv2d]["fill"] = "#E69F00"
color_map[nn.BatchNorm2d]["fill"] = "#009E73"
color_map[nn.ReLU]["fill"] = "#56B4E9"

img = visualtorch.render(model, input_shape, style="lenet", color_map=color_map, offset_z=1, spacing=10)

dpi = 150  # rendered at 2x this in the final doc build (savefig.dpi=300 in conf.py)
plt.figure(figsize=(img.width / dpi, img.height / dpi), dpi=dpi)
plt.imshow(img)
plt.axis("off")
plt.tight_layout()
plt.show()
