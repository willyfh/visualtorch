"""Shared legend placement helpers."""

# Copyright (C) 2024 VisualTorch Contributors
# SPDX-License-Identifier: MIT

from typing import Literal

from PIL import Image

LegendPosition = Literal["top-left", "top-right", "top-center", "bottom-left", "bottom-right", "bottom-center"]
LEGEND_POSITIONS: tuple[LegendPosition, ...] = (
    "top-left",
    "top-right",
    "top-center",
    "bottom-left",
    "bottom-right",
    "bottom-center",
)


def validate_legend_position(legend_position: str) -> None:
    """Raise a targeted error for unsupported legend placement values."""
    if legend_position not in LEGEND_POSITIONS:
        supported = ", ".join(f"{position!r}" for position in LEGEND_POSITIONS)
        error_msg = f"unsupported legend_position: {legend_position!r}. Supported positions: {supported}."
        raise ValueError(error_msg)


def place_legend(
    img: Image.Image,
    legend_image: Image.Image,
    legend_position: LegendPosition,
    background_fill: str | tuple[int, ...],
) -> Image.Image:
    """Place the legend outside the diagram while aligning it within the final canvas."""
    vertical_position, horizontal_position = legend_position.split("-", 1)
    canvas = Image.new(
        "RGBA",
        (max(img.width, legend_image.width), img.height + legend_image.height),
        background_fill,
    )
    legend_x = _horizontal_legend_offset(canvas.width, legend_image.width, horizontal_position)

    if vertical_position == "top":
        canvas.paste(legend_image, (legend_x, 0))
        canvas.paste(img, (0, legend_image.height))
    else:
        canvas.paste(img, (0, 0))
        canvas.paste(legend_image, (legend_x, img.height))
    return canvas


def _horizontal_legend_offset(canvas_width: int, legend_width: int, horizontal_position: str) -> int:
    """Return the x offset for the legend row inside the final canvas."""
    if horizontal_position == "right":
        return canvas_width - legend_width
    if horizontal_position == "center":
        return (canvas_width - legend_width) // 2
    return 0
