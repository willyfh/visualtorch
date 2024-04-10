"""Layered View module for pytorch model visualization."""

# Copyright (C) 2024 Willy Fitra Hendria
# SPDX-License-Identifier: MIT

from collections import OrderedDict, defaultdict
from math import ceil
from typing import Any

import aggdraw
import PIL
import torch
from PIL import Image, ImageFont
from torch import nn

from .utils.layer_utils import InputDummyLayer, SpacingDummyLayer, register_hook
from .utils.utils import (
    ImageDraw,
    StackedBox,
    get_rgba_tuple,
    self_multiply,
)


def lenet_view(
    model: nn.Module | nn.Sequential | nn.ModuleList,
    input_shape: tuple[int, ...],
    to_file: str | None = None,
    min_z: int = 1,
    min_xy: int = 10,
    max_xy: int = 2000,
    scale_z: float = 1,
    scale_xy: float = 1,
    type_ignore: list | None = None,
    index_ignore: list | None = None,
    color_map: dict | None = None,
    one_dim_orientation: str = "z",
    background_fill: str | tuple[int, ...] = "white",
    padding: int = 10,
    spacing: int = 10,
    draw_funnel: bool = True,
    shade_step: int = 10,
    font: ImageFont = None,
    font_color: str | tuple[int, ...] = "black",
    opacity: int = 255,
    max_channels: int = 100,
    offset_z: int = 10,
) -> PIL.Image:
    """Generate a LeNet style architecture visualization for a given torch model.

    TODO: remove unnecessary arguments for this LeNet style architecture.

    Args:
        model (torch.nn.Module): A torch model that will be visualized.
        input_shape (tuple): The shape of the input tensor (default: (1, 3, 224, 224)).
        to_file (str, optional): Path to the file to write the created image. Overwrite if exist.
            Image type is inferred from the file extension. Providing None will disable writing.
        min_z (int, optional): Minimum size in pixels that a layer will have along the z-axis.
        min_xy (int, optional): Minimum size in pixels that a layer will have along the x and y axes.
        max_channels (int, optional): Maximum number of channels.
        max_xy (int, optional): Maximum size in pixels that a layer will have along the x and y axes.
        scale_z (float, optional): Scalar multiplier for the size of each layer along the z-axis.
        scale_xy (float, optional): Scalar multiplier for the size of each layer along the x and y axes.
        type_ignore (list, optional): List of layer types in the torch model to ignore during drawing.
        index_ignore (list, optional): List of layer indexes in the torch model to ignore during drawing.
        color_map (dict, optional): Dictionary defining fill and outline colors for each layer by class type.
            Will fallback to default values for unspecified classes.
        one_dim_orientation (str, optional): Axis on which one-dim layers should be drawn. E.g., 'x', 'y', or 'z'.
        background_fill (str or tuple, optional): Background color for the image. A string or a tuple (R, G, B, A).
        draw_volume (bool, optional): Flag to switch between 3D volumetric view and 2D box view.
        padding (int, optional): Distance in pixels before the first and after the last layer.
        spacing (int, optional): Spacing in pixels between two layers.
        draw_funnel (bool, optional): If True, a funnel will be drawn between consecutive layers.
        shade_step (int, optional): Deviation in lightness for drawing shades (only in volumetric view).
        font (PIL.ImageFont, optional): Font that will be used for the legend. If None, default font will be used.
        font_color (str or tuple, optional): Color for the font if used. Can be a string or a tuple (R, G, B, A).
        opacity (int): Transparency of the color (0 ~ 255).
        offset_z (int): control the offset of overlapping between channels.

    Returns:
        PIL.Image: An Image object representing the generated architecture visualization.
    """
    # Iterate over the model to compute bounds and generate boxes

    x_off = -1

    img_height = 0
    max_right = 0

    if type_ignore is None:
        type_ignore = []

    if index_ignore is None:
        index_ignore = []

    _color_map: dict = {}
    if color_map is not None:
        _color_map = defaultdict(dict, color_map)

    dummy_input = torch.rand(*input_shape)

    # Get the list of layers

    layers: OrderedDict[str, Any] = OrderedDict()
    layers["inputdummy"] = {"module": InputDummyLayer("Input"), "output_shape": dummy_input.shape}
    hooks: list[Any] = []

    model.apply(lambda module: register_hook(model, module, hooks, layers))

    with torch.no_grad():
        if isinstance(model, nn.ModuleList):
            output = dummy_input
            for layer in model:
                output = layer(output)
        else:
            output = model(dummy_input)

    # remove these hooks
    for h in hooks:
        h.remove()

    layer_y, _, boxes, x_off, img_height, max_right = _create_architecture(
        layers,
        x_off,
        img_height,
        max_right,
        type_ignore,
        index_ignore,
        min_xy,
        min_z,
        one_dim_orientation,
        scale_xy,
        max_xy,
        scale_z,
        shade_step,
        spacing,
        _color_map,
        opacity,
        padding,
        max_channels,
        offset_z,
    )

    # Generate image
    img_width = max_right + x_off + padding
    img = Image.new(
        "RGBA",
        (int(ceil(img_width)), int(ceil(img_height)) + 100),
        background_fill,
    )
    draw = aggdraw.Draw(img)

    # Create a new RGBA image for drawing the text
    text_img = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw_text = ImageDraw.Draw(text_img)

    # x, y correction (centering)
    for i, node in enumerate(boxes):
        y_off = (img.height - layer_y[i]) / 2
        node.y1 += y_off
        node.y2 += y_off

        node.x1 += x_off
        node.x2 += x_off

    # Draw created boxes

    last_box = None
    if font is None:
        font = ImageFont.load_default()

    for box in boxes:
        pen = aggdraw.Pen(get_rgba_tuple(box.outline))

        if last_box is not None and draw_funnel:
            last_off = -last_box.offset_z * last_box.de // 2
            off = -box.offset_z * box.de // 2

            draw.line(
                [
                    last_box.x2 + last_off,
                    last_box.y1 + last_off,
                    box.x1 + off,
                    box.y1 + off,
                ],
                pen,
            )
            last_off += last_box.offset_z * (last_box.de - 1)
            off += box.offset_z * (box.de - 1)
            draw.line(
                [
                    last_box.x2 + last_off,
                    last_box.y2 + last_off,
                    box.x1 + off,
                    box.y2 + off,
                ],
                pen,
            )

        # Draw the box
        box.draw(draw)

        # Draw the text
        loc_x = box.x1 + (box.x2 - box.x1) // 4
        draw_text.text((loc_x, img.height - 50), f"{box.label} {box.output_shape}", font=font, fill=font_color)

        last_box = box

    draw.flush()

    # Merge the text image onto the original image
    img = Image.alpha_composite(img, text_img)

    if to_file is not None:
        img.save(to_file)
    return img


def _create_architecture(
    layers: OrderedDict[str, Any],
    x_off: int,
    img_height: int,
    max_right: int,
    type_ignore: list,
    index_ignore: list,
    min_xy: int,
    min_z: int,
    one_dim_orientation: str,
    scale_xy: float,
    max_xy: int,
    scale_z: float,
    shade_step: int,
    spacing: int,
    color_map: dict,
    opacity: int,
    padding: int,
    max_channels: int,
    offset_z: int,
) -> tuple:
    boxes = []
    layer_types = []
    layer_y = []
    current_y = padding
    current_x = padding

    for index, key in enumerate(layers):
        layer = layers[key]["module"]
        shape = layers[key]["output_shape"]
        # Do no render the SpacingDummyLayer, just increase the pointer
        if type(layer) == SpacingDummyLayer:
            current_x += layer.spacing
            continue

        # Ignore layers that the use has opted out to
        if type(layer) in type_ignore or index in index_ignore:
            continue

        layer_type = type(layer)

        if layer_type not in layer_types:
            layer_types.append(layer_type)

        x = min_xy
        y = min_xy
        z = min_z

        shape = shape[1:]  # drop batch size

        if len(shape) == 1:
            if one_dim_orientation in ["x", "y", "z"]:
                shape = (1,) * "cxyz".index(one_dim_orientation) + shape
            else:
                error_msg = f"unsupported orientation: {one_dim_orientation}"
                raise ValueError(error_msg)
        ori_shape = shape
        shape = shape + (1,) * (4 - len(shape))  # expand 4D.

        x = min(max(shape[1] * scale_xy, x), max_xy)
        y = min(max(shape[2] * scale_xy, y), max_xy)
        z = min(max(int(self_multiply(shape[0:1] + shape[3:]) * scale_z), z), max_channels)

        box = StackedBox()
        box.offset_z = offset_z

        box.label = layer.name() if layer_type == InputDummyLayer else layer_type.__name__
        box.output_shape = tuple(ori_shape)

        box.de = z

        x_off = 0

        # top left coordinate
        current_x += box.de * offset_z // 2
        box.x1 = current_x
        box.y1 = current_y

        # bottom right coordinate
        box.x2 = box.x1 + x
        box.y2 = box.y1 + y

        box.set_fill(
            color_map.get(layer_type, {}).get(
                "fill",
                "#cccccc",
            ),
            opacity,
        )
        box.outline = color_map.get(layer_type, {}).get("outline", "black")
        color_map[layer_type] = {"fill": box.fill, "outline": box.outline}

        box.shade = shade_step
        boxes.append(box)
        layer_y.append(box.y2 - box.y1)

        # Update image bounds
        hh = box.y2 - box.y1 + (box.de * offset_z) + offset_z * 2
        if hh > img_height:
            img_height = hh

        if box.x2 + box.de > max_right:
            max_right = box.x2 + (box.de * offset_z // 2)

        current_y = padding
        current_x += x + spacing + box.de * offset_z // 2

    return layer_y, layer_types, boxes, x_off, img_height, max_right
