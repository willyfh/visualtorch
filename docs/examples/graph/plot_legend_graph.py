"""Layer Legend
=======================================

Add a legend below a graph diagram with ``legend=True``. Each circular swatch
uses the same color as its layer type in the diagram.
"""  # noqa: D205

import matplotlib.pyplot as plt
import visualtorch
from torch import nn

model = nn.Sequential(
    nn.Linear(4, 6),
    nn.ReLU(),
    nn.Linear(6, 3),
)
input_shape = (1, 4)

img = visualtorch.render(
    model,
    input_shape=input_shape,
    style="graph",
    show_neurons=True,
    legend=True,
)

dpi = 150  # rendered at 2x this in the final doc build (savefig.dpi=300 in conf.py)
plt.figure(figsize=(img.width / dpi, img.height / dpi), dpi=dpi)
plt.imshow(img)
plt.axis("off")
plt.tight_layout()
plt.show()
