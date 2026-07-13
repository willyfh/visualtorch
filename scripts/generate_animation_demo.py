"""Regenerate the README's hero animated-reveal demo GIF.

Renders the same colorful sequential CNN used in the "Custom Color" docs example through
`visualtorch.animate(style="flow")`, at a wide spacing so the classic funnel taper from wide conv
layers down to the final classifier reads clearly. Run this after any change that affects how
this specific example renders, then update the README's `<img>` src to the new commit's SHA
(matching the existing static banner's pinning convention).

Note: earlier versions of this script tried branching models (an Inception-style 4-branch block,
then a ResNet-style residual block) to showcase branch/merge reveal behavior, upscaled afterward
since their native renders were small. This colorful sequential model renders at a large enough
native resolution on its own (no branches to stack vertically eating into width) that no
upscaling step is needed at all - avoiding any upscale-filter tradeoffs entirely.

Usage: `python scripts/generate_animation_demo.py` from anywhere - writes directly to
`docs/source/_static/images/banners/readme-animated-demo.gif`.
"""

# Copyright (C) 2024 VisualTorch Contributors
# SPDX-License-Identifier: MIT

from collections import defaultdict
from pathlib import Path

import visualtorch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
BANNER_DIR = REPO_ROOT / "docs" / "source" / "_static" / "images" / "banners"

_FRAME_DURATION_MS = 400
_FINAL_HOLD_DURATION_MS = 1500


def main() -> None:
    """Render and save the README's hero animated demo GIF."""
    BANNER_DIR.mkdir(parents=True, exist_ok=True)

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

    color_map: dict = defaultdict(dict)
    color_map[visualtorch.Input]["fill"] = "#D55E00"  # vermillion
    color_map[nn.Conv2d]["fill"] = "#E69F00"  # orange
    color_map[nn.ReLU]["fill"] = "#56B4E9"  # sky blue
    color_map[nn.MaxPool2d]["fill"] = "#CC79A7"  # reddish purple
    color_map[nn.Flatten]["fill"] = "#009E73"  # bluish green
    color_map[nn.Linear]["fill"] = "#0072B2"  # blue

    frames = visualtorch.animate(model, input_shape, style="flow", color_map=color_map, scale_xy=1, spacing=40)
    assert frames is not None  # to_file wasn't passed, so a frame list is always returned

    durations = [_FRAME_DURATION_MS] * (len(frames) - 1) + [_FINAL_HOLD_DURATION_MS]

    gif_path = BANNER_DIR / "readme-animated-demo.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    print(gif_path, frames[0].size)


if __name__ == "__main__":
    main()
