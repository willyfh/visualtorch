"""Layer Legend
=======================================

Add a legend below a LeNet-style diagram with ``legend=True``. The legend uses
stacked-plane swatches that match the colors and visual style of the layers.
"""  # noqa: D205

import matplotlib.pyplot as plt
import visualtorch
from torch import nn

model = nn.Sequential(
    nn.Conv2d(3, 8, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(8, 16, kernel_size=3, padding=1),
    nn.ReLU(),
)
input_shape = (1, 3, 32, 32)

img = visualtorch.render(
    model,
    input_shape=input_shape,
    style="lenet",
    legend=True,
)

dpi = 150  # rendered at 2x this in the final doc build (savefig.dpi=300 in conf.py)
plt.figure(figsize=(img.width / dpi, img.height / dpi), dpi=dpi)
plt.imshow(img)
plt.axis("off")
plt.tight_layout()
plt.show()
