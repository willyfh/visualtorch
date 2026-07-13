"""Regenerate the README's hero animated-reveal demo GIF.

Renders the same residual block used in the docs gallery examples through `flow_view_animate`,
showing the skip connection stay hidden until its merge point is revealed. Run this after any
change that affects how this specific example renders, then update the README's `<img>` src to
the new commit's SHA (matching the existing static banner's pinning convention).

Note: an earlier version of this script used an Inception-style 4-branch model, but 4 branches
stacking vertically in one column made the render tall and narrow regardless of style settings -
a single skip connection (one extra branch, not four) renders landscape instead, which is a much
better fit for a README hero image.

Usage: `python scripts/generate_animation_demo.py` from anywhere - writes directly to
`docs/source/_static/images/banners/readme-animated-demo.gif`.
"""

# Copyright (C) 2024 VisualTorch Contributors
# SPDX-License-Identifier: MIT

from pathlib import Path

import torch
from PIL import Image
from torch import nn
from visualtorch.flow import flow_view_animate

REPO_ROOT = Path(__file__).resolve().parent.parent
BANNER_DIR = REPO_ROOT / "docs" / "source" / "_static" / "images" / "banners"

# This model's small channel counts keep flow_view's native render tiny (well under 200px wide),
# which looks blurry once GitHub displays it at normal README width - upscaling the frames
# directly gives a crisp result at a predictable final size. NEAREST, not a smoothing filter like
# LANCZOS: this is flat-color vector-like content (solid fills, thin outlines), and a GIF's
# 256-color palette dithers LANCZOS's soft anti-aliased edge pixels into visible noise once
# quantized - NEAREST introduces no new intermediate colors, so it survives palette reduction
# cleanly (verified by comparing both as actual encoded GIFs, not just the pre-quantization PNGs).
_UPSCALE_FACTOR = 4
_FRAME_DURATION_MS = 600
_FINAL_HOLD_DURATION_MS = 1500


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
    """Render and save the README's hero animated demo GIF."""
    BANNER_DIR.mkdir(parents=True, exist_ok=True)

    model = ResidualBlock(channels=8)
    input_shape = (1, 8, 16, 16)

    frames = flow_view_animate(model, input_shape, scale_xy=3)
    assert frames is not None  # to_file wasn't passed, so a frame list is always returned

    upscaled = [
        frame.resize((frame.width * _UPSCALE_FACTOR, frame.height * _UPSCALE_FACTOR), Image.NEAREST) for frame in frames
    ]
    durations = [_FRAME_DURATION_MS] * (len(upscaled) - 1) + [_FINAL_HOLD_DURATION_MS]

    gif_path = BANNER_DIR / "readme-animated-demo.gif"
    upscaled[0].save(gif_path, save_all=True, append_images=upscaled[1:], duration=durations, loop=0)
    print(gif_path, upscaled[0].size)


if __name__ == "__main__":
    main()
