"""Utils module for pytorch model visualization."""

# Copyright (C) 2020 Paul Gavrikov
# Copyright (C) 2024 Willy Fitra Hendria
# SPDX-License-Identifier: MIT

from typing import Any

import aggdraw
import PIL
from PIL import Image, ImageColor, ImageDraw

InputShape = tuple[int, ...] | tuple[tuple[int, ...], ...]
"""A single flat shape (one input tensor) or a tuple of per-tensor shapes (multiple inputs)."""


class Shape:
    """Base class for shapes"""

    def __init__(self) -> None:
        self.x1: float = 0
        self.x2: float = 0
        self.y1: float = 0
        self.y2: float = 0
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
    extra_output_shapes: tuple[tuple[int, ...], ...] = ()

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

    def front_offset(self) -> float:
        """The offset of the last (frontmost, fully unoccluded) slice drawn by `draw`.

        Every other slice is either drawn before it (and thus at least partly covered by it or
        by a slice drawn later still) or shares this same offset when there's only one slice, so
        this is the only offset a connector can attach to without appearing to run through a
        slice drawn on top of it.
        """
        if getattr(self, "de", 0) <= 0:
            return 0.0
        initial_offset = -self.offset_z * self.de // 2
        return initial_offset + self.offset_z * (self.de - 1)


class Box(Shape):
    """Box shape class."""

    de: int
    shade: int
    output_shape: tuple[int, ...]
    extra_output_shapes: tuple[tuple[int, ...], ...] = ()

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
        # Okabe-Ito: a colorblind-safe palette (Okabe & Ito, 2008) widely recommended for
        # scientific visualization, e.g. in Nature's figure guidelines.
        self.colors = (
            colors
            if colors is not None
            else [
                "#E69F00",  # orange
                "#56B4E9",  # sky blue
                "#009E73",  # bluish green
                "#F0E442",  # yellow
                "#0072B2",  # blue
                "#D55E00",  # vermillion
                "#CC79A7",  # reddish purple
            ]
        )

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


def validate_input_shape(input_shape: tuple) -> tuple[tuple[int, ...], ...]:
    """Validate input_shape and normalize it to a tuple of per-tensor shapes.

    Accepts either a single flat tuple of ints (one input tensor, e.g. (1, 3, 224, 224)) or a
    tuple of such tuples (multiple separate input tensors, e.g. ((1, 3, 224, 224), (1, 10))),
    for models whose forward() takes more than one positional tensor argument.

    Args:
        input_shape (tuple): The shape(s) to validate.

    Returns:
        tuple: A tuple of per-tensor shapes - always length 1 for a single-input model, length N
            for an N-input model - so every caller can treat both cases uniformly.

    Raises:
        ValueError: If input_shape is neither a flat tuple of ints nor a tuple of tuples of ints.
    """
    error_msg = (
        "input_shape must be either a single tuple of ints, e.g. (1, 3, 224, 224), or - for "
        "models whose forward() takes multiple separate input tensors - a tuple of per-tensor "
        f"shape tuples, e.g. ((1, 3, 224, 224), (1, 10)). Got {input_shape!r} instead."
    )
    if not isinstance(input_shape, tuple) or len(input_shape) == 0:
        raise ValueError(error_msg)

    if all(isinstance(dim, int) for dim in input_shape):
        return (input_shape,)

    is_well_formed_multi = all(
        isinstance(shape, tuple) and len(shape) > 0 and all(isinstance(dim, int) for dim in shape)
        for shape in input_shape
    )
    if not is_well_formed_multi:
        raise ValueError(error_msg)

    return tuple(input_shape)


def self_multiply(tensor_tuple: tuple | list) -> int | float:
    """Multiplies all elements in the tuple together.

    Elements that are themselves a tuple/list (e.g. a nested torch.Size, which can end up here
    when a layer's captured output shape wasn't a plain flat shape) are flattened by multiplying
    their own elements together first, so the result is always a scalar.

    Args:
        tensor_tuple (tuple or list): A tuple containing tensors.

    Returns:
        int or float: The result of multiplying all elements together.
    """
    tensor_list = [v for v in tensor_tuple if v is not None]
    if len(tensor_list) == 0:
        return 0
    s = 1
    for v in tensor_list:
        s *= self_multiply(v) if isinstance(v, tuple | list) else v
    return s


def format_shape_label(output_shape: tuple[int, ...], extra_output_shapes: tuple[tuple[int, ...], ...]) -> str:
    """Format an output shape for display, appending any extra output shapes if present.

    A module that returns more than one meaningful tensor (e.g. `nn.LSTM`'s `(output, (h_n,
    c_n))`) would otherwise only ever show `output_shape` (the first tensor found, also the one
    driving box size) with no indication the other tensors exist at all. `+` is used rather than
    `visualtorch`'s existing `/` convention (already used to join sibling branches within one
    column) so this doesn't read as an alternative/branch - these are all real outputs of this
    same node, not one of several options.
    """
    label = str(output_shape)
    if extra_output_shapes:
        label += " + " + " + ".join(str(shape) for shape in extra_output_shapes)
    return label


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
