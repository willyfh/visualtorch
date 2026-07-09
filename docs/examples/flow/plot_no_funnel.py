"""Disable Funnel Connectors
=======================================

Compare the default tapered connectors with plain box-to-box connectors by setting
``draw_funnel=False``.
"""  # noqa: D205

import matplotlib.pyplot as plt
import visualtorch
from torch import nn

# Example of a simple CNN model using nn.Sequential
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(64 * 28 * 28, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

input_shape = (1, 3, 224, 224)

img_default = visualtorch.render(model, input_shape=input_shape, style="flow")
img_no_funnel = visualtorch.render(model, input_shape=input_shape, style="flow", draw_funnel=False)

dpi = 150  # rendered at 2x this in the final doc build (savefig.dpi=300 in conf.py)
_, axes = plt.subplots(
    1,
    2,
    figsize=((img_default.width + img_no_funnel.width) / dpi, max(img_default.height, img_no_funnel.height) / dpi),
    dpi=dpi,
)

for ax, img, title in zip(axes, [img_default, img_no_funnel], ["Default", "draw_funnel=False"], strict=True):
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()
