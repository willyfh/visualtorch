"""Regenerate the docs gallery's per-style animated-reveal demo GIFs + thumbnails.

Renders one small model with a skip connection through `visualtorch.animate()` for each style, so
each style's `plot_animated_*.py` gallery example has a real, versioned GIF to embed
(sphinx-gallery can't capture a GIF via its default matplotlib scraper, so these are pre-generated
and committed rather than produced during the doc build). Also saves each demo's last frame (the
fully-revealed diagram) as a static PNG, used as the gallery thumbnail via that example's
`# sphinx_gallery_thumbnail_path` directive.

Usage: `python scripts/generate_animation_demos.py` from anywhere - writes directly to
`docs/source/_static/images/animations/`.
"""

# Copyright (C) 2024 VisualTorch Contributors
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import visualtorch
from torch import nn

if TYPE_CHECKING:
    from visualtorch.render import Style

REPO_ROOT = Path(__file__).resolve().parent.parent
ANIMATION_DIR = REPO_ROOT / "docs" / "source" / "_static" / "images" / "animations"


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


def main() -> None:
    """Render and save each style's demo GIF + last-frame thumbnail PNG."""
    ANIMATION_DIR.mkdir(parents=True, exist_ok=True)

    model = ResidualBlock(channels=8)
    input_shape = (1, 8, 16, 16)

    demo_kwargs: dict[Style, dict[str, Any]] = {
        "graph": {"show_neurons": False, "layer_spacing": 60},
        "flow": {"scale_xy": 3},
        "lenet": {"scale_xy": 3},
    }

    for style, kwargs in demo_kwargs.items():
        frames = visualtorch.animate(model, input_shape, style=style, **kwargs)
        assert frames is not None  # to_file wasn't passed, so a frame list is always returned

        gif_path = ANIMATION_DIR / f"{style}_animated_demo.gif"
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=600, loop=0)
        print(gif_path, frames[0].size)

        # The last frame (fully revealed diagram) makes a far more informative gallery
        # thumbnail than the first (which is just the input node, mostly blank).
        thumbnail_path = ANIMATION_DIR / f"{style}_animated_demo_thumbnail.png"
        frames[-1].convert("RGB").save(thumbnail_path)
        print(thumbnail_path)


if __name__ == "__main__":
    main()
