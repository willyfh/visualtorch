"""Graph View module for pytorch model visualization."""

# Copyright (C) 2020 Paul Gavrikov
# Copyright (C) 2024 Willy Fitra Hendria
# SPDX-License-Identifier: MIT

from collections import defaultdict
from math import ceil
from typing import Any

import aggdraw
import numpy as np
import torch
from PIL import Image, ImageFont

from .utils.layer_utils import add_input_dummy_layer, model_to_adj_matrix
from .utils.traced_layer import TracedLayer
from .utils.utils import Box, Circle, Ellipses, ImageDraw, get_keys_by_value


def graph_view(
    model: torch.nn.Module,
    input_shape: tuple[int, ...],
    to_file: str | None = None,
    color_map: dict[Any, Any] | None = None,
    node_size: int = 50,
    background_fill: str | tuple[int, ...] = "white",
    padding: int = 10,
    layer_spacing: int = 250,
    node_spacing: int = 10,
    connector_fill: str | tuple[int, ...] = "gray",
    connector_width: int = 1,
    ellipsize_after: int = 10,
    show_neurons: bool = True,
    opacity: int = 255,
    show_dimension: bool = False,
    font: ImageFont = None,
    font_color: str | tuple[int, ...] = "black",
) -> Image.Image:
    """Generates an architecture visualization for a given linear PyTorch model in a graph style.

    Args:
        model (torch.nn.Module): A PyTorch model that will be visualized.
        input_shape (tuple): The shape of the input tensor.
        to_file (str, optional): Path to the file to write the created image to. If the image does not exist yet,
            it will be created, else overwritten. Image type is inferred from the file ending. Providing None
            will disable writing.
        color_map (dict, optional): Dict defining fill and outline for each layer by class type. Will fallback
            to default values for not specified classes.
        node_size (int, optional): Size in pixels each node will have.
        background_fill (Any, optional): Color for the image background. Can be str or (R,G,B,A).
        padding (int, optional): Distance in pixels before the first and after the last layer.
        layer_spacing (int, optional): Spacing in pixels between two layers.
        node_spacing (int, optional): Spacing in pixels between nodes.
        connector_fill (Any, optional): Color for the connectors. Can be str or (R,G,B,A).
        connector_width (int, optional): Line-width of the connectors in pixels.
        ellipsize_after (int, optional): Maximum number of neurons per layer to draw. If a layer is exceeding this,
            the remaining neurons will be drawn as ellipses.
        show_neurons (bool, optional): If True a node for each neuron in supported layers is created (constrained by
            ellipsize_after), else each layer is represented by a node.
        opacity (int, optional): Transparency of the color (0 ~ 255).
        show_dimension (bool, optional): If True, print each layer's output shape below it.
        font (PIL.ImageFont, optional): Font used for the shape labels. If None, default font will be used.
        font_color (str or tuple, optional): Color for the font if used. Can be a string or a tuple (R, G, B, A).

    Returns:
        Image.Image: Generated architecture image.
    """
    _color_map: dict = {}
    if color_map is not None:
        _color_map = defaultdict(dict, color_map)

    # Iterate over the model to compute bounds and generate boxes

    # Attach helper layers

    id_to_num_mapping, adj_matrix, model_layers, direct_input_node_ids = model_to_adj_matrix(
        model,
        input_shape,
    )

    # Add fake input layers

    id_to_num_mapping, adj_matrix, model_layers = add_input_dummy_layer(
        input_shape,
        id_to_num_mapping,
        adj_matrix,
        model_layers,
        direct_input_node_ids,
    )

    # Create architecture

    current_x = padding  # + input_label_size[0] + text_padding

    layers, layer_y, id_to_node_list_map, label_info = _create_architecture(
        model_layers,
        current_x,
        show_neurons,
        ellipsize_after,
        node_size,
        node_spacing,
        _color_map,
        opacity,
        layer_spacing,
    )

    if font is None and show_dimension:
        font = ImageFont.load_default()

    # Reserve a fixed-height strip below each layer's own content for its shape label,
    # measured up front so labels never clip the bottom of the canvas.
    label_row_height = 0
    if show_dimension:
        label_row_height = font.getbbox("Ag")[3] + 5

    # Generate image

    img_width: float = len(layers) * node_size + (len(layers) - 1) * layer_spacing + 2 * padding
    img_height = max(*layer_y) + 2 * padding + label_row_height

    if show_dimension:
        label_info, img_width = _fit_dimension_labels(layers, label_info, font, img_width)

    img = Image.new(
        "RGBA",
        (int(ceil(img_width)), int(ceil(img_height))),
        background_fill,
    )

    draw = aggdraw.Draw(img)

    # y correction (centering)
    for i, layer in enumerate(layers):
        y_off = (img.height - label_row_height - layer_y[i]) / 2
        node: Any
        for node in layer:
            node.y1 += y_off
            node.y2 += y_off
        label_info[i] = [(label, center_x, y + y_off) for label, center_x, y in label_info[i]]

    for start_idx, end_idx in zip(*np.where(adj_matrix > 0), strict=False):
        start_id = next(get_keys_by_value(id_to_num_mapping, start_idx))
        end_id = next(get_keys_by_value(id_to_num_mapping, end_idx))

        start_layer_list = id_to_node_list_map[start_id]
        end_layer_list = id_to_node_list_map[end_id]

        # draw connectors
        for start_node in start_layer_list:
            for end_node in end_layer_list:
                if not isinstance(start_node, Ellipses) and not isinstance(
                    end_node,
                    Ellipses,
                ):
                    _draw_connector(
                        draw,
                        start_node,
                        end_node,
                        color=connector_fill,
                        width=connector_width,
                    )

    for layer in layers:
        for node in layer:
            node.draw(draw)

    draw.flush()

    # Print each layer's output shape below its own block of nodes
    if show_dimension:
        img = _draw_dimension_labels(img, label_info, font, font_color)

    if to_file is not None:
        img.save(to_file)

    return img


def _fit_dimension_labels(
    layers: list[list[Any]],
    label_info: list[list[tuple[str, float, float]]],
    font: ImageFont,
    img_width: float,
) -> tuple[list[list[tuple[str, float, float]]], float]:
    """Extend the canvas and shift nodes so the outermost labels never clip an edge.

    Extend the canvas and shift nodes right if the outermost labels are wider than their
    column, so they never clip past the left or right edge of the canvas.

    Returns:
        tuple: The updated `label_info` and `img_width`.
    """
    extra_left = 0.0
    extra_right = 0.0
    for column_labels in label_info:
        for label, center_x, _y in column_labels:
            label_width = font.getbbox(label)[2]
            extra_left = max(extra_left, label_width / 2 - center_x)
            extra_right = max(extra_right, center_x + label_width / 2 - img_width)

    if extra_left > 0:
        for layer in layers:
            for node in layer:
                node.x1 += extra_left
                node.x2 += extra_left
        label_info = [
            [(label, center_x + extra_left, y) for label, center_x, y in column_labels] for column_labels in label_info
        ]
        img_width += extra_left

    return label_info, img_width + max(extra_right, 0.0)


def _draw_dimension_labels(
    img: Image.Image,
    label_info: list[list[tuple[str, float, float]]],
    font: ImageFont,
    font_color: str | tuple[int, ...],
) -> Image.Image:
    """Draw each layer's output shape centered beneath its own block of nodes."""
    text_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw_text = ImageDraw.Draw(text_img)

    for column_labels in label_info:
        for label, center_x, y_bottom in column_labels:
            text_width = font.getbbox(label)[2]
            draw_text.text((center_x - text_width / 2, y_bottom + 2), label, font=font, fill=font_color)

    return Image.alpha_composite(img, text_img)


def _draw_connector(
    draw: aggdraw.Draw,
    start_node: Box | Circle | Ellipses,
    end_node: Box | Circle | Ellipses,
    color: str | tuple[int, ...],
    width: int,
) -> None:
    """Draw the line connector between nodes."""
    pen = aggdraw.Pen(color, width)
    x1 = start_node.x2
    y1 = start_node.y1 + (start_node.y2 - start_node.y1) / 2
    x2 = end_node.x1
    y2 = end_node.y1 + (end_node.y2 - end_node.y1) / 2
    draw.line([x1, y1, x2, y2], pen)


def _retrieve_isbox_units(layer: TracedLayer, show_neurons: bool) -> tuple[bool, int]:
    """Return the number of units and the flag whether to visualize using a box or not.

    Units are derived generically from the layer's already-known output shape (dropping the
    batch dim) rather than from private autograd attributes tied to a hardcoded op list:
    a 1D output (e.g. Linear) uses its only dim as the unit count, a 3D output (e.g. Conv,
    channels-first) uses the channel dim, and anything else (e.g. an RNN's (seq, hidden))
    falls back to its last dim.
    """
    is_box = True
    units = 1
    if show_neurons:
        dims = layer.output_shape[1:]
        if len(dims) in (1, 3):
            is_box = False
            units = dims[0]
        elif len(dims) >= 2:
            is_box = False
            units = dims[-1]
    return is_box, units


def _create_architecture(
    model_layers: list[list[TracedLayer]],
    current_x: int,
    show_neurons: bool,
    ellipsize_after: int,
    node_size: int,
    node_spacing: int,
    color_map: dict[Any, Any],
    opacity: int,
    layer_spacing: int,
) -> tuple[list, list, dict, list[list[tuple[str, float, float]]]]:
    """Create nodes of architecture for each layers."""
    id_to_node_list_map = {}
    layers = []
    layer_y = []
    # (label text, x center, y bottom) per layer, grouped by column - a column can hold more
    # than one layer if multiple leaf modules share the same depth (parallel branches).
    label_info: list[list[tuple[str, float, float]]] = []
    for layer_list in model_layers:
        current_y = 0
        nodes = []
        column_labels: list[tuple[str, float, float]] = []
        layer: TracedLayer
        for layer in layer_list:
            is_box, units = _retrieve_isbox_units(layer, show_neurons)
            layer_type = type(layer.module)

            n = min(units, ellipsize_after)
            layer_nodes = []

            for i in range(n):
                scale = 1
                c: Box | Circle | Ellipses
                if not is_box:
                    c = Circle() if i != ellipsize_after - 2 else Ellipses()
                else:
                    c = Box()
                    scale = 3

                c.x1 = current_x
                c.y1 = current_y
                c.x2 = c.x1 + node_size
                c.y2 = c.y1 + node_size * scale

                current_y = c.y2 + node_spacing

                c.set_fill(
                    color_map.get(layer_type, {}).get("fill", "#ADD8E6"),
                    opacity,
                )
                c.outline = color_map.get(layer_type, {}).get(
                    "outline",
                    "black",
                )

                layer_nodes.append(c)

            id_to_node_list_map[layer.node_id] = layer_nodes
            nodes.extend(layer_nodes)
            column_labels.append((str(layer.output_shape), current_x + node_size / 2, current_y))
            current_y += 2 * node_size

        layer_y.append(current_y - node_spacing - 2 * node_size)
        layers.append(nodes)
        label_info.append(column_labels)
        current_x += node_size + layer_spacing
    return layers, layer_y, id_to_node_list_map, label_info
