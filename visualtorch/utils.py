from typing import Any, Dict
from PIL import ImageColor, ImageDraw, Image
import aggdraw
import PIL
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import List, Tuple


class RectShape:
    def __init__(self):
        self.x1 = 0
        self.x2 = 0
        self.y1 = 0
        self.y2 = 0
        self._fill = None
        self._outline = None

    @property
    def fill(self):
        return self._fill

    @fill.setter
    def fill(self, v):
        self._fill = get_rgba_tuple(v)

    @property
    def outline(self):
        return self._outline

    @outline.setter
    def outline(self, v):
        self._outline = get_rgba_tuple(v)

    def _get_pen_brush(self):
        pen = aggdraw.Pen(self._outline)
        brush = aggdraw.Brush(self._fill)
        return pen, brush


class Box(RectShape):
    de: int
    shade: int

    def draw(self, draw: ImageDraw):
        pen, brush = self._get_pen_brush()

        if hasattr(self, "de") and self.de > 0:
            brush_s1 = aggdraw.Brush(fade_color(self.fill, self.shade))
            brush_s2 = aggdraw.Brush(fade_color(self.fill, 2 * self.shade))

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


class Circle(RectShape):
    def draw(self, draw: ImageDraw):
        pen, brush = self._get_pen_brush()
        draw.ellipse([self.x1, self.y1, self.x2, self.y2], pen, brush)


class Ellipses(RectShape):
    def draw(self, draw: ImageDraw):
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
    def __init__(self, colors: list | None = None):
        self._cache: Dict[type, Any] = dict()
        self.colors = (
            colors
            if colors is not None
            else ["#FFE4B5", "#ADD8E6", "#98FB98", "#FFA07A", "#D8BFD8"]
        )

    def get_color(self, class_type: type):
        if class_type not in self._cache.keys():
            index = len(self._cache.keys()) % len(self.colors)
            self._cache[class_type] = self.colors[index]
        return self._cache.get(class_type)


def fade_color(color: tuple, fade_amount: int) -> tuple:
    r = max(0, color[0] - fade_amount)
    g = max(0, color[1] - fade_amount)
    b = max(0, color[2] - fade_amount)
    return r, g, b, color[3]


def get_rgba_tuple(color: Any) -> tuple:
    """
    Converts a color representation to an RGBA tuple.

    Args:
        color (Any): The color representation to be converted.

    Returns:
        tuple: A tuple representing the color in RGBA format, with values for red (R), green (G), blue (B), and alpha (A).
    """
    if isinstance(color, tuple):
        rgba = color
    elif isinstance(color, int):
        rgba = (color >> 16 & 0xFF, color >> 8 & 0xFF, color & 0xFF, color >> 24 & 0xFF)
    else:
        rgba = ImageColor.getrgb(color)

    if len(rgba) == 3:
        rgba = (rgba[0], rgba[1], rgba[2], 255)
    return rgba


def get_keys_by_value(d, v):
    for key in d.keys():  # reverse search the dict for the value
        if d[key] == v:
            yield key


def self_multiply(tensor_tuple: tuple) -> int | float:
    """
    Multiplies all elements in the tuple together.

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
    im1: Image, im2: Image, background_fill: Any = "white"
) -> PIL.Image:
    """
    Concatenates two PIL images vertically.

    Args:
        im1 (PIL.Image): The top image.
        im2 (PIL.Image): The bottom image.
        background_fill (str or tuple, optional): Color for the image background. Can be a string or a tuple (R, G, B, A).

    Returns:
        PIL.Image: A new image resulting from the vertical concatenation of the two input images.
    """
    dst = Image.new(
        "RGBA", (max(im1.width, im2.width), im1.height + im2.height), background_fill
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
    background_fill: Any = "white",
) -> PIL.Image:
    """
    Creates a linear layout of a passed list of images in horizontal or vertical orientation. The layout will wrap in x
    or y dimension if a maximum value is exceeded.

    Args:
        images (list): List of PIL images.
        max_width (int, optional): Maximum width of the image. Only enforced in horizontal orientation.
        max_height (int, optional): Maximum height of the image. Only enforced in vertical orientation.
        horizontal (bool, optional): If True, will draw images horizontally, else vertically.
        padding (int, optional): Top, bottom, left, right border distance in pixels.
        spacing (int, optional): Spacing in pixels between elements.
        background_fill (str or tuple, optional): Color for the image background. Can be a string or a tuple (R, G, B, A).

    Returns:
        PIL.Image: An Image object representing the linear layout of the passed list of images.
    """
    coords = list()
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
    for img, coord in zip(images, coords):
        layout.paste(img, coord)

    return layout


def register_hook(
    model: nn.Module, module: nn.Module, hooks: List, layers: OrderedDict
) -> None:
    """
    Registers a forward hook on the specified module and collects the module and the output shapes.

    Args:
        model (nn.Module): The parent model.
        module (nn.Module): The module to register the hook on.
        hooks (List): A list to store the registered hooks.
        layers (OrderedDict): An OrderedDict to store information about the registered modules and output shapes.

    Returns:
        None
    """

    def hook(
        module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor
    ) -> None:
        class_name = str(module.__class__).split(".")[-1].split("'")[0]
        module_idx = len(layers)

        m_key = "%s-%i" % (class_name, module_idx + 1)
        layers[m_key] = OrderedDict()
        layers[m_key]["module"] = module
        if isinstance(output, (list, tuple)):
            layers[m_key]["output_shape"] = tuple((-1,) + o.size()[1:] for o in output)
        else:
            layers[m_key]["output_shape"] = output.size()

    if (
        not isinstance(module, nn.Sequential)
        and not isinstance(module, nn.ModuleList)
        and module is not model
    ):
        hooks.append(module.register_forward_hook(hook))
