"""Regenerate the README/docs banner images.

Renders a fixed set of small example models through `visualtorch.render()`, then composes them
into labeled-card grids. Run this after any change that affects how these specific examples
render (a new default, a renamed parameter, a layout tweak) so the banners stay in sync with
actual current behavior instead of going stale.

Usage: `python scripts/generate_banners.py` from anywhere - writes directly to
`docs/source/_static/images/banners/{readme-examples,visualizations-preview}.png`.
"""

# Copyright (C) 2024 VisualTorch Contributors
# SPDX-License-Identifier: MIT

from collections import defaultdict
from pathlib import Path

import matplotlib.font_manager
import torch
import visualtorch
from PIL import Image, ImageDraw, ImageFont
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
BANNER_DIR = REPO_ROOT / "docs" / "source" / "_static" / "images" / "banners"
FONT_PATH = matplotlib.font_manager.findfont("DejaVu Sans")


class SimpleDense(nn.Module):
    """A tiny fully-connected model, for the neuron-level-detail panel."""

    def __init__(self) -> None:
        super().__init__()
        self.h0 = nn.Linear(4, 8)
        self.h1 = nn.Linear(8, 8)
        self.h2 = nn.Linear(8, 4)
        self.out = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the forward pass."""
        x = self.h0(x)
        x = self.h1(x)
        x = self.h2(x)
        return self.out(x)


class ResidualBlock(nn.Module):
    """A classic ResNet-style block with a plain identity shortcut, for the skip-connections panel."""

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


class InceptionBlock(nn.Module):
    """A simplified Inception-style block with four parallel branches, for the multi-branch-merge panel."""

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


class SiameseNet(nn.Module):
    """A two-branch model: an image branch and a tabular-vector branch, for the multi-input panel."""

    def __init__(self) -> None:
        super().__init__()
        self.image_branch = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.vector_branch = nn.Sequential(nn.Linear(10, 8), nn.ReLU(), nn.Linear(8, 8), nn.ReLU())
        self.head = nn.Linear(16, 4)

    def forward(self, image: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        """Run each branch on its own input tensor, then concatenate and project."""
        image_features = self.image_branch(image)
        vector_features = self.vector_branch(vector)
        merged = torch.cat([image_features, vector_features], dim=1)
        return self.head(merged)


class UNet(nn.Module):
    """A small U-Net with 2 encoder/decoder stages and skip connections, for the volumetric-style panel."""

    def __init__(self) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU())
        self.final = nn.Conv2d(16, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample through 2 encoder stages, then upsample back, concatenating each
        encoder stage's output into its corresponding decoder stage.
        """  # noqa: D205
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


class LegendDemoNet(nn.Module):
    """A small, standalone model - not reused elsewhere in the grid - just to showcase legend=True."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(2, 8)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(2)
        self.out = nn.Conv2d(8, 16, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the forward pass."""
        x = self.relu(self.norm(self.conv(x)))
        x = self.pool(x)
        return self.out(x)


def cnn_sequential() -> nn.Sequential:
    """A moderately deep CNN with a Linear head, reused across several panels."""
    return nn.Sequential(
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


def small_cnn() -> nn.Sequential:
    """A tiny conv-only CNN, reused across several panels."""
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.MaxPool2d(2, 2),
    )


def rounded_rect_mask(size: tuple[int, int], radius: int) -> Image.Image:
    """Build an L-mode mask for a rounded-rectangle card."""
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([0, 0, size[0] - 1, size[1] - 1], radius=radius, fill=255)
    return mask


def make_card(
    panel_img: Image.Image,
    card_w: int,
    card_h: int,
    img_area_h: int,
    caption: str,
    font: ImageFont.FreeTypeFont,
    radius: int = 18,
) -> Image.Image:
    """Compose one panel image, its caption, and a rounded card border into a single card image."""
    bg = (255, 255, 255)
    border_color = (225, 225, 228)
    caption_color = (60, 60, 65)

    card = Image.new("RGB", (card_w, card_h), bg)
    draw = ImageDraw.Draw(card)

    # Fit panel image within (card_w - 2*pad) x img_area_h, centered - allow slight upscale for
    # tiny panels (capped by the same max_w/max_h bound) so they don't look lost on the card.
    pad = 24
    max_w = card_w - 2 * pad
    max_h = img_area_h - pad
    scale = min(max_w / panel_img.width, max_h / panel_img.height, 1.0)
    if scale < 1.0 or panel_img.width < max_w * 0.6:
        scale = min(max_w / panel_img.width, max_h / panel_img.height)
    new_size = (max(1, int(panel_img.width * scale)), max(1, int(panel_img.height * scale)))
    resized = panel_img.convert("RGBA").resize(new_size, Image.LANCZOS)

    px = pad + (max_w - new_size[0]) // 2
    py = pad // 2 + (max_h - new_size[1]) // 2
    card.paste(resized, (px, py), resized)

    bbox = draw.textbbox((0, 0), caption, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (card_w - tw) // 2
    ty = img_area_h + (card_h - img_area_h - th) // 2 - bbox[1]
    draw.text((tx, ty), caption, font=font, fill=caption_color)

    mask = rounded_rect_mask((card_w, card_h), radius)
    out = Image.new("RGBA", (card_w, card_h), (0, 0, 0, 0))
    out.paste(card, (0, 0), mask)
    border_draw = ImageDraw.Draw(out)
    border_draw.rounded_rectangle([0, 0, card_w - 1, card_h - 1], radius=radius, outline=border_color, width=2)
    return out


def build_grid(
    entries: list[tuple[Image.Image, str]],
    cols: int,
    card_w: int,
    card_h: int,
    img_area_h: int,
    margin: int,
    gap: int,
    font_size: int,
) -> Image.Image:
    """Lay out a list of (panel image, caption) pairs into a card grid."""
    canvas_bg = (247, 247, 250)
    font = ImageFont.truetype(FONT_PATH, font_size)
    rows = (len(entries) + cols - 1) // cols
    canvas_w = margin * 2 + cols * card_w + (cols - 1) * gap
    canvas_h = margin * 2 + rows * card_h + (rows - 1) * gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), canvas_bg)

    for i, (panel_img, caption) in enumerate(entries):
        card = make_card(panel_img, card_w, card_h, img_area_h, caption, font)
        col = i % cols
        row = i // cols
        x = margin + col * (card_w + gap)
        y = margin + row * (card_h + gap)
        canvas.paste(card, (x, y), card)

    return canvas


def render_readme_panels() -> list[tuple[Image.Image, str]]:
    """Render every panel shown in the README's example grid."""
    color_map: dict = defaultdict(dict)
    color_map[visualtorch.Input]["fill"] = "#D55E00"
    color_map[nn.Conv2d]["fill"] = "#E69F00"
    color_map[nn.ReLU]["fill"] = "#56B4E9"
    color_map[nn.MaxPool2d]["fill"] = "#CC79A7"
    color_map[nn.Flatten]["fill"] = "#009E73"
    color_map[nn.Linear]["fill"] = "#0072B2"

    return [
        (visualtorch.render(SimpleDense(), (1, 4), style="graph"), "Neuron-Level Detail"),
        (visualtorch.render(small_cnn(), (1, 3, 128, 128), style="lenet"), "LeNet Style"),
        (
            visualtorch.render(cnn_sequential(), (1, 3, 224, 224), style="flow", draw_volume=False),
            "Flat 2D Style",
        ),
        (
            visualtorch.render(cnn_sequential(), (1, 3, 224, 224), style="flow", color_map=color_map),
            "Custom Colors",
        ),
        (
            visualtorch.render(ResidualBlock(8), (1, 8, 16, 16), style="graph", show_neurons=False, layer_spacing=60),
            "Skip Connections",
        ),
        (
            visualtorch.render(LegendDemoNet(), (1, 3, 32, 32), style="flow", legend=True, scale_xy=4, spacing=15),
            "Legend",
        ),
        (
            visualtorch.render(
                InceptionBlock(16, 8, 8, 8, 8),
                (1, 16, 16, 16),
                style="graph",
                show_neurons=False,
                layer_spacing=60,
            ),
            "Multi-Branch Merge",
        ),
        (
            visualtorch.render(UNet(), (1, 3, 64, 64), style="flow", scale_xy=3, spacing=15),
            "3D Volumetric Style",
        ),
        (
            visualtorch.render(
                cnn_sequential(),
                (1, 3, 224, 224),
                style="flow",
                type_ignore=[nn.ReLU, nn.Flatten],
                show_input=False,
            ),
            "Hide Layers",
        ),
        (
            visualtorch.render(
                small_cnn(),
                (1, 3, 16, 16),
                style="graph",
                show_neurons=False,
                show_dimension=True,
                layer_spacing=60,
            ),
            "Output Shapes",
        ),
        (
            visualtorch.render(cnn_sequential(), (1, 3, 224, 224), style="flow", low_dim_orientation="x", spacing=40),
            "Low-Dim Layer Orientation",
        ),
        (
            visualtorch.render(SiameseNet(), ((1, 3, 16, 16), (1, 10)), style="flow", scale_xy=3, spacing=15),
            "Multi-Input Model",
        ),
    ]


def render_preview_panels() -> list[tuple[Image.Image, str]]:
    """Render the 4 panels shown in the standalone visualizations-preview.png banner."""
    return [
        (visualtorch.render(SimpleDense(), (1, 4), style="graph"), "Neuron-Level Detail"),
        (
            visualtorch.render(
                InceptionBlock(16, 8, 8, 8, 8),
                (1, 16, 16, 16),
                style="graph",
                show_neurons=False,
                layer_spacing=60,
            ),
            "Block Style",
        ),
        (visualtorch.render(UNet(), (1, 3, 64, 64), style="flow", scale_xy=3, spacing=15), "U-Net (Flow Style)"),
        (visualtorch.render(small_cnn(), (1, 3, 128, 128), style="lenet"), "LeNet Style"),
    ]


def main() -> None:
    """Render every panel and compose the 2 banner images into their final destination."""
    BANNER_DIR.mkdir(parents=True, exist_ok=True)

    readme_banner = build_grid(
        render_readme_panels(),
        cols=4,
        card_w=380,
        card_h=320,
        img_area_h=250,
        margin=24,
        gap=22,
        font_size=20,
    )
    readme_path = BANNER_DIR / "readme-examples.png"
    readme_banner.save(readme_path)
    print(readme_path, readme_banner.size)

    preview_banner = build_grid(
        render_preview_panels(),
        cols=4,
        card_w=380,
        card_h=380,
        img_area_h=380,
        margin=24,
        gap=22,
        font_size=1,
    )
    preview_path = BANNER_DIR / "visualizations-preview.png"
    preview_banner.save(preview_path)
    print(preview_path, preview_banner.size)


if __name__ == "__main__":
    main()
