"""Regenerate the README's hero animated-reveal demo GIF.

Renders a small Inception-style multi-branch model through `flow_view_animate`, showing the
parallel branches fanning out and merging as the animation reveals column by column. Run this
after any change that affects how this specific example renders, then update the README's
`<img>` src to the new commit's SHA (matching the existing static banner's pinning convention).

Usage: `python scripts/generate_animation_demo.py` from anywhere - writes directly to
`docs/source/_static/images/banners/readme-animated-demo.gif`.
"""

# Copyright (C) 2024 VisualTorch Contributors
# SPDX-License-Identifier: MIT

from pathlib import Path

import torch
from torch import nn
from visualtorch.flow import flow_view_animate

REPO_ROOT = Path(__file__).resolve().parent.parent
BANNER_DIR = REPO_ROOT / "docs" / "source" / "_static" / "images" / "banners"


class InceptionBlock(nn.Module):
    """A simplified Inception-style block with four parallel branches, merged by concatenation."""

    def __init__(self, in_channels: int, out_1x1: int, out_3x3: int, out_5x5: int, out_pool: int) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, out_1x1, kernel_size=1), nn.BatchNorm2d(out_1x1))
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_3x3, kernel_size=1),
            nn.Conv2d(out_3x3, out_3x3, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_5x5, kernel_size=1),
            nn.Conv2d(out_5x5, out_5x5, kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool, kernel_size=1),
        )
        total_channels = out_1x1 + out_3x3 + out_5x5 + out_pool
        self.project = nn.Conv2d(total_channels, total_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run every branch on the same input, then concatenate and project."""
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        merged = torch.cat([b1, b2, b3, b4], dim=1)
        return self.project(merged)


def main() -> None:
    """Render and save the README's hero animated demo GIF."""
    BANNER_DIR.mkdir(parents=True, exist_ok=True)

    model = InceptionBlock(16, 8, 8, 8, 8)
    input_shape = (1, 16, 16, 16)

    gif_path = BANNER_DIR / "readme-animated-demo.gif"
    flow_view_animate(
        model,
        input_shape,
        scale_xy=3,
        draw_volume=False,
        to_file=str(gif_path),
    )
    print(gif_path)


if __name__ == "__main__":
    main()
