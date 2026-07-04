"""Color Palettes
=======================================

Instead of hand-building a ``color_map`` entry for every layer type, pick a named ``palette`` -
it's used as the fallback fill color for any layer type not given an explicit ``color_map``
override. A handful of the built-in palettes are shown below; see ``visualtorch.PALETTES.keys()``
for the full list.
"""  # noqa: D205

import matplotlib.pyplot as plt
import visualtorch
from torch import nn

model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(32 * 112 * 112, 10),
)

input_shape = (1, 3, 224, 224)

dpi = 150  # rendered at 2x this in the final doc build (savefig.dpi=300 in conf.py)


def _show(palette: str) -> None:
    img = visualtorch.render(model, input_shape, style="flow", palette=palette)
    plt.figure(figsize=(img.width / dpi, img.height / dpi), dpi=dpi)
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# %%
# Okabe-Ito (default)
# --------------------
# Colorblind-safe, widely recommended for scientific visualization.

_show("okabe_ito")

# %%
# Nord
# ----
# Cool, arctic-toned - a popular editor/terminal theme.

_show("nord")

# %%
# Dracula
# -------
# High-contrast purple/pink/green/cyan - punchy and playful.

_show("dracula")

# %%
# Catppuccin
# ----------
# Soft pastel-but-vibrant - the current darling of the dev-theme world.

_show("catppuccin")
