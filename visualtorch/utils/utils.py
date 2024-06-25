"""Utils module for pytorch model visualization."""

# Copyright (C) 2020 Paul Gavrikov
# Copyright (C) 2024 Willy Fitra Hendria
# SPDX-License-Identifier: MIT

from collections.abc import Generator
from typing import Any

import aggdraw
import PIL
from PIL import Image, ImageColor, ImageDraw


class Shape:
    """Base class for shapes"""

    def __init__(self) -> None:
        self.x1 = 0
        self.x2 = 0
        self.y1 = 0
        self.y2 = 0
        self._fill = ()
        self._outline = ()

    @property
    def fill(self) -> tuple:
        """Return the fill color."""
        return self._fill

    def set_fill(self, color: str | tuple, opacity: int) -> None:
        """Set color and opacity.

        Args:
            color (Any): The color representation, i.e., rbg, hex, or color name.
            opacity (int): The transparency of the color (0 ~ 255).

        Returns:
            None
        """
        self._fill = get_rgba_tuple(color, opacity)

    @property
    def outline(self) -> tuple:
        """Return the outline/border color."""
        return self._outline

    @outline.setter
    def outline(self, v: str | tuple) -> None:
        self._outline = get_rgba_tuple(v)

    def _get_pen_brush(self) -> tuple:
        """Get aggdraw pen and brush"""
        pen = aggdraw.Pen(self._outline)
        brush = aggdraw.Brush(self._fill)
        return pen, brush


class StackedBox(Shape):
    """Stacked Box shape class."""

    de: int
    shade: int
    offset_z: int
    label: str
    output_shape: tuple

    def draw(self, draw: ImageDraw) -> None:
        """Draw box shape."""
        pen, brush = self._get_pen_brush()

        if hasattr(self, "de") and self.de > 0:
            brush_s2 = aggdraw.Brush(_fade_color(self.fill, 4 * self.shade))

            # Calculate initial offset
            offset = -self.offset_z * self.de // 2

            # Define initial brush
            brush_choice = brush_s2 if self.de % 2 == 0 else brush

            # Loop through each iteration
            for _ in range(self.de):
                draw.rectangle(
                    [self.x1 + offset, self.y1 + offset, self.x2 + offset, self.y2 + offset],
                    pen,
                    brush_choice,
                )
                offset += self.offset_z
                # Switch brush
                brush_choice = brush_s2 if brush_choice is brush else brush

        else:
            draw.rectangle([self.x1, self.y1, self.x2, self.y2], pen, brush)


class Box(Shape):
    """Box shape class."""

    de: int
    shade: int

    def draw(self, draw: ImageDraw) -> None:
        """Draw box shape."""
        pen, brush = self._get_pen_brush()

        if hasattr(self, "de") and self.de > 0:
            brush_s1 = aggdraw.Brush(_fade_color(self.fill, self.shade))
            brush_s2 = aggdraw.Brush(_fade_color(self.fill, 2 * self.shade))

            draw.line(
                [
                    self.x1 + self.de,
                    self.y1 - self.de,
                    self.x1 + self.de,
                    self.y2 - self.de,
                ],
                pen,
            )
            draw.line([self.x1 + self.de, self.y2 - self.de, self.x1, self.y2], pen)
            draw.line(
                [
                    self.x1 + self.de,
                    self.y2 - self.de,
                    self.x2 + self.de,
                    self.y2 - self.de,
                ],
                pen,
            )

            draw.polygon(
                [
                    self.x1,
                    self.y1,
                    self.x1 + self.de,
                    self.y1 - self.de,
                    self.x2 + self.de,
                    self.y1 - self.de,
                    self.x2,
                    self.y1,
                ],
                pen,
                brush_s1,
            )

            draw.polygon(
                [
                    self.x2 + self.de,
                    self.y1 - self.de,
                    self.x2,
                    self.y1,
                    self.x2,
                    self.y2,
                    self.x2 + self.de,
                    self.y2 - self.de,
                ],
                pen,
                brush_s2,
            )

        draw.rectangle([self.x1, self.y1, self.x2, self.y2], pen, brush)


class Circle(Shape):
    """Circle shape class."""

    def draw(self, draw: ImageDraw) -> None:
        """Draw circle shape."""
        pen, brush = self._get_pen_brush()
        draw.ellipse([self.x1, self.y1, self.x2, self.y2], pen, brush)


class Ellipses(Shape):
    """Ellipses shape class."""

    def draw(self, draw: ImageDraw) -> None:
        """Draw ellipses shape."""
        pen, brush = self._get_pen_brush()
        w = self.x2 - self.x1
        d = int(w / 7)
        draw.ellipse(
            [
                self.x1 + (w - d) / 2,
                self.y1 + 1 * d,
                self.x1 + (w + d) / 2,
                self.y1 + 2 * d,
            ],
            pen,
            brush,
        )
        draw.ellipse(
            [
                self.x1 + (w - d) / 2,
                self.y1 + 3 * d,
                self.x1 + (w + d) / 2,
                self.y1 + 4 * d,
            ],
            pen,
            brush,
        )
        draw.ellipse(
            [
                self.x1 + (w - d) / 2,
                self.y1 + 5 * d,
                self.x1 + (w + d) / 2,
                self.y1 + 6 * d,
            ],
            pen,
            brush,
        )


class ColorWheel:
    """Default colors for the shapes."""

    def __init__(self, colors: list | None = None) -> None:
        self._cache: dict[type, Any] = {}
        self.colors = colors if colors is not None else ["#FFE4B5", "#ADD8E6", "#98FB98", "#FFA07A", "#D8BFD8"]

    def get_color(self, class_type: type) -> tuple | None:
        """Return color from cache if exist, if not, get from the list and store it to the cache."""
        if class_type not in self._cache:
            index = len(self._cache.keys()) % len(self.colors)
            self._cache[class_type] = self.colors[index]
        return self._cache.get(class_type)


def _fade_color(color: tuple, fade_amount: int) -> tuple:
    """To create shadow effect."""
    r = max(0, color[0] - fade_amount)
    g = max(0, color[1] - fade_amount)
    b = max(0, color[2] - fade_amount)
    return r, g, b, color[3]


def get_rgba_tuple(color: str | int | tuple, opacity: int = 255) -> tuple:
    """Converts a color representation to an RGBA tuple.

    Args:
        color (Any): The color representation to be converted.
        opacity (int): The transparency of the color (0 ~ 255).

    Returns:
        tuple: A tuple representing the color in RGBA format, with values for R, G, B, and A.
    """
    if isinstance(color, tuple):
        rgba = color
    elif isinstance(color, int):
        rgba = (color >> 16 & 0xFF, color >> 8 & 0xFF, color & 0xFF, color >> 24 & 0xFF)
    else:
        rgba = ImageColor.getrgb(color)

    if len(rgba) == 3:
        rgba = (rgba[0], rgba[1], rgba[2], opacity)
    return rgba


def get_keys_by_value(d: dict, v: int) -> Generator:
    """Get keys from dictionary given the value."""
    for key in d:  # reverse search the dict for the value
        if d[key] == v:
            yield key


def self_multiply(tensor_tuple: tuple) -> int | float:
    """Multiplies all elements in the tuple together.

    Args:
        tensor_tuple (tuple): A tuple containing tensors.

    Returns:
        int or float: The result of multiplying all elements together.
    """
    tensor_list = list(tensor_tuple)
    if None in tensor_list:
        tensor_list.remove(None)
    if len(tensor_list) == 0:
        return 0
    s = tensor_list[0]
    for i in range(1, len(tensor_list)):
        s *= tensor_list[i]
    return s


def vertical_image_concat(
    im1: Image,
    im2: Image,
    background_fill: str | tuple = "white",
) -> PIL.Image:
    """Concatenates two PIL images vertically.

    Args:
        im1 (PIL.Image): The top image.
        im2 (PIL.Image): The bottom image.
        background_fill (str or tuple, optional): Color for the background. A string or a tuple (R, G, B, A).

    Returns:
        PIL.Image: A new image resulting from the vertical concatenation of the two input images.
    """
    dst = Image.new(
        "RGBA",
        (max(im1.width, im2.width), im1.height + im2.height),
        background_fill,
    )
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def linear_layout(
    images: list,
    max_width: int = -1,
    max_height: int = -1,
    horizontal: bool = True,
    padding: int = 0,
    spacing: int = 0,
    background_fill: str | tuple = "white",
) -> PIL.Image:
    """Creates a linear layout of a passed list of images in horizontal or vertical orientation.

    The layout will wrap in x or y dimension if a maximum value is exceeded.

    Args:
        images (list): List of PIL images.
        max_width (int, optional): Maximum width of the image. Only enforced in horizontal orientation.
        max_height (int, optional): Maximum height of the image. Only enforced in vertical orientation.
        horizontal (bool, optional): If True, will draw images horizontally, else vertically.
        padding (int, optional): Top, bottom, left, right border distance in pixels.
        spacing (int, optional): Spacing in pixels between elements.
        background_fill (str or tuple, optional): Color for the background. A string or a tuple (R, G, B, A).

    Returns:
        PIL.Image: An Image object representing the linear layout of the passed list of images.
    """
    coords = []
    width = 0
    height = 0

    x, y = padding, padding

    for img in images:
        if horizontal:
            if max_width != -1 and x + img.width > max_width:
                # make a new row
                x = padding
                y = height - padding + spacing
            coords.append((x, y))

            width = max(x + img.width + padding, width)
            height = max(y + img.height + padding, height)

            x += img.width + spacing
        else:
            if max_height != -1 and y + img.height > max_height:
                # make a new column
                x = width - padding + spacing
                y = padding
            coords.append((x, y))

            width = max(x + img.width + padding, width)
            height = max(y + img.height + padding, height)

            y += img.height + spacing

    layout = Image.new("RGBA", (width, height), background_fill)
    for img, coord in zip(images, coords, strict=False):
        layout.paste(img, coord)

    return layout
