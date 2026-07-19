"""Legend Position
=======================================

Place the graph legend above or below the diagram, aligned left, center, or right.
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
dpi = 150  # rendered at 2x this in the final doc build (savefig.dpi=300 in conf.py)


def _show(position: str) -> None:
    img = visualtorch.render(
        model,
        input_shape=input_shape,
        style="graph",
        show_neurons=True,
        legend=True,
        legend_position=position,
    )
    plt.figure(figsize=(img.width / dpi, img.height / dpi), dpi=dpi)
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# %%
# Top-left
# --------

_show("top-left")

# %%
# Top-center
# ----------

_show("top-center")

# %%
# Bottom-right
# ------------

_show("bottom-right")
