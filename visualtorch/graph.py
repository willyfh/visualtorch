"""Graph View module for pytorch model visualization."""

# Copyright (C) 2020 Paul Gavrikov
# Copyright (C) 2024 VisualTorch Contributors
# SPDX-License-Identifier: MIT

from collections import defaultdict
from math import ceil
from typing import Any

import aggdraw
import torch
from PIL import Image, ImageFont

from .backend import extract_architecture
from .connectors import compute_skip_levels, draw_connector
from .utils.traced_layer import TracedLayer
from .utils.utils import (
    Box,
    Circle,
    ColorWheel,
    Ellipses,
    ImageDraw,
    InputDtype,
    InputShape,
    format_shape_label,
    resolve_palette,
)


def graph_view(
    model: torch.nn.Module,
    input_shape: InputShape,
    input_dtype: InputDtype | None = None,
    to_file: str | None = None,
    color_map: dict[Any, Any] | None = None,
    palette: str = "okabe_ito",
    node_size: int = 50,
    background_fill: str | tuple[int, ...] = "white",
    padding: int = 10,
    layer_spacing: int = 250,
    node_spacing: int = 10,
    type_ignore: list | None = None,
    outline_width: int = 1,
    connector_fill: str | tuple[int, ...] = "gray",
    connector_width: int = 1,
    ellipsize_after: int = 10,
    show_neurons: bool = True,
    opacity: int = 255,
    show_dimension: bool = False,
    font: ImageFont = None,
    font_color: str | tuple[int, ...] = "black",
    level_gap: int | None = None,
    show_input: bool = True,
    show_arrows: bool = False,
) -> Image.Image:
    """Generates an architecture visualization for a given linear PyTorch model in a graph style.

    Args:
        model (torch.nn.Module): A PyTorch model that will be visualized.
        input_shape (tuple): The shape of the input tensor. For a model whose forward() takes
            multiple separate input tensors, pass a tuple of per-tensor shapes instead, one per
            positional argument in order, e.g. ((1, 3, 224, 224), (1, 10)).
        input_dtype (torch.dtype, optional): The dtype of the dummy input tensor(s) used to trace
            the model. Defaults to `None` for every input, giving a uniformly random float
            tensor. Needed for a model that starts with an integer/bool-input layer (e.g.
            `nn.Embedding`, which rejects a float index tensor): pass `torch.long`, or - for a
            model with multiple input tensors of different kinds - a tuple of per-tensor dtypes,
            e.g. `(torch.long, None)`.
        to_file (str, optional): Path to the file to write the created image to. If the image does not exist yet,
            it will be created, else overwritten. Image type is inferred from the file ending. Providing None
            will disable writing.
        color_map (dict, optional): Dict defining fill and outline for each layer by class type. Will fallback
            to default values for not specified classes.
        palette (str, optional): Named color palette used as the fallback for any layer type not
            given an explicit override via `color_map`. One of `"okabe_ito"` (default,
            colorblind-safe), `"tol_bright"`, `"tol_muted"`, `"tab10"`, `"grayscale"`, `"nord"`,
            `"dracula"`, `"gruvbox"`, `"solarized"`, `"material"`, `"catppuccin"`.
        node_size (int, optional): Size in pixels each node will have.
        background_fill (Any, optional): Color for the image background. Can be str or (R,G,B,A).
        padding (int, optional): Distance in pixels before the first and after the last layer.
        layer_spacing (int, optional): Spacing in pixels between two layers.
        node_spacing (int, optional): Spacing in pixels between nodes.
        type_ignore (list, optional): List of layer types in the torch model to ignore during drawing.
        outline_width (int, optional): Line width in pixels for the shape borders. Defaults to 1.
        connector_fill (Any, optional): Color for the connectors. Can be str or (R,G,B,A).
        connector_width (int, optional): Line-width of the connectors in pixels.
        show_arrows (bool, optional): If True, draw a small arrowhead at each connector's
            downstream endpoint to indicate data-flow direction.
        ellipsize_after (int, optional): Maximum number of neurons per layer to draw. If a layer is exceeding this,
            the remaining neurons will be drawn as ellipses.
        show_neurons (bool, optional): If True a node for each neuron in supported layers is created (constrained by
            ellipsize_after), else each layer is represented by a node.
        opacity (int, optional): Transparency of the color (0 ~ 255).
        show_dimension (bool, optional): If True, print each layer's output shape below it.
        font (PIL.ImageFont, optional): Font used for the shape labels. If None, default font will be used.
        font_color (str or tuple, optional): Color for the font if used. Can be a string or a tuple (R, G, B, A).
        level_gap (int, optional): Vertical spacing in pixels between stacked skip-connection detour
            routes. If None, defaults to `node_size`.
        show_input (bool, optional): For a single-input model, whether to draw the synthetic
            "Input" node. Defaults to True. Set False to hide it - e.g. if you're overlaying your
            own custom input illustration instead. Has no effect on a multi-input model, where
            every input is always shown (omitting any of them would make it ambiguous which
            arrow belongs to which named input). Ignored (input always kept) when the input feeds
            more than one consumer, e.g. a residual shortcut, since dropping it would silently
            discard that edge.

    Returns:
        Image.Image: Generated architecture image.
    """
    if type_ignore is None:
        type_ignore = []

    _color_map: dict = {}
    if color_map is not None:
        _color_map = defaultdict(dict, color_map)

    architecture = extract_architecture(model, input_shape, input_dtype)

    # Hiding is only honored when it's safe to: an input feeding more than one consumer (e.g. a
    # residual block's raw-input shortcut) must stay visible regardless of show_input, since
    # dropping it would silently discard that edge. Has no effect on a multi-input model (2+
    # nodes in the input column), which is always shown in full.
    input_column = architecture.columns[0]
    if len(input_column) == 1 and not show_input:
        input_node_id = input_column[0].node_id
        input_out_degree = int(architecture.adjacency[architecture.id_to_index[input_node_id]].sum())
        raw_columns = architecture.columns if input_out_degree > 1 else architecture.columns[1:]
    else:
        raw_columns = architecture.columns

    # Create architecture

    filtered_columns = [[layer for layer in column if type(layer.module) not in type_ignore] for column in raw_columns]
    filtered_columns = [column for column in filtered_columns if column]

    current_x = padding  # + input_label_size[0] + text_padding

    layers, layer_y, id_to_node_list_map, label_info = _create_architecture(
        filtered_columns,
        current_x,
        show_neurons,
        ellipsize_after,
        node_size,
        node_spacing,
        _color_map,
        opacity,
        layer_spacing,
        ColorWheel(colors=resolve_palette(palette)),
        outline_width,
    )

    # An edge whose endpoint was just dropped above (a hidden input's own edges) can no longer
    # be drawn - filter those out rather than leaving a dangling id_to_node_list_map lookup.
    edges = [edge for edge in architecture.edges() if edge[0] in id_to_node_list_map and edge[1] in id_to_node_list_map]

    # Skip connections (edges spanning more than one column) need to be routed above the
    # diagram, or they'd draw collinear with - and hidden under - the ordinary adjacent-column
    # edges beneath them. Compute this before img_height so the canvas can reserve the room.
    edge_to_level, num_levels = compute_skip_levels(
        edges,
        architecture.id_to_column,
        lambda start_id, end_id: _edge_draws_visible_content(id_to_node_list_map, start_id, end_id),
        lambda node_id: _union_bbox(id_to_node_list_map.get(node_id, [])),
    )
    resolved_level_gap = level_gap if level_gap is not None else node_size
    top_margin_for_skips = num_levels * resolved_level_gap

    if font is None and show_dimension:
        font = ImageFont.load_default()

    # Reserve a fixed-height strip below each layer's own content for its shape label,
    # measured up front so labels never clip the bottom of the canvas.
    label_row_height = 0
    if show_dimension:
        label_row_height = font.getbbox("Ag")[3] + 5

    # Generate image

    img_width: float = len(layers) * node_size + (len(layers) - 1) * layer_spacing + 2 * padding
    img_height = max(*layer_y) + 2 * padding + label_row_height + top_margin_for_skips

    if show_dimension:
        label_info, img_width = _fit_dimension_labels(layers, label_info, font, img_width)

    img = Image.new(
        "RGBA",
        (int(ceil(img_width)), int(ceil(img_height))),
        background_fill,
    )

    draw = aggdraw.Draw(img)

    # y correction (centering) - content is centered within the band below the reserved
    # top strip, so existing (no-skip) layouts are unaffected when top_margin_for_skips == 0.
    for i, layer in enumerate(layers):
        y_off = top_margin_for_skips + (img.height - label_row_height - top_margin_for_skips - layer_y[i]) / 2
        node: Any
        for node in layer:
            node.y1 += y_off
            node.y2 += y_off
        label_info[i] = [(label, center_x, y + y_off) for label, center_x, y in label_info[i]]

    _draw_connectors(
        draw,
        edges,
        id_to_node_list_map,
        edge_to_level,
        num_levels,
        show_neurons,
        padding,
        resolved_level_gap,
        connector_fill,
        connector_width,
        show_arrows,
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


def _edge_draws_visible_content(id_to_node_list_map: dict[str, list], start_id: str, end_id: str) -> bool:
    """Whether this edge has at least one non-Ellipses node pair, i.e. would draw something."""
    return any(
        not isinstance(start_node, Ellipses) and not isinstance(end_node, Ellipses)
        for start_node in id_to_node_list_map[start_id]
        for end_node in id_to_node_list_map[end_id]
    )


def _union_bbox(nodes: list[Circle | Box | Ellipses]) -> tuple[float, float, float, float] | None:
    """The union bounding box over a list of shapes, or None if the list is empty.

    Includes Ellipses nodes (unlike `_edge_draws_visible_content`) - an ellipsis marker still
    occupies screen space a connector line could cross, so it's a real obstacle for collision
    purposes even though it wouldn't itself be a meaningful connector endpoint.
    """
    if not nodes:
        return None
    return (
        min(node.x1 for node in nodes),
        min(node.y1 for node in nodes),
        max(node.x2 for node in nodes),
        max(node.y2 for node in nodes),
    )


def _draw_connectors(
    draw: aggdraw.Draw,
    edges: list[tuple[str, str]],
    id_to_node_list_map: dict[str, list],
    edge_to_level: dict[tuple[str, str], int],
    num_levels: int,
    show_neurons: bool,
    padding: int,
    resolved_level_gap: int,
    connector_fill: str | tuple[int, ...],
    connector_width: int,
    show_arrows: bool,
) -> None:
    """Draw every connector, routing skip-connection edges above the diagram."""
    for start_id, end_id in edges:
        start_layer_list = id_to_node_list_map[start_id]
        end_layer_list = id_to_node_list_map[end_id]

        level = edge_to_level.get((start_id, end_id))
        detour_y = None if level is None else padding + (num_levels - 1 - level) * resolved_level_gap

        if level is not None and show_neurons:
            # A skip edge under show_neurons=True would otherwise fan out to a dense
            # neuron-to-neuron mesh; collapse it to one representative box-to-box-style
            # connector instead, or it renders as a solid overlapping band.
            start_candidates = [node for node in start_layer_list if not isinstance(node, Ellipses)]
            end_candidates = [node for node in end_layer_list if not isinstance(node, Ellipses)]
            if start_candidates and end_candidates:
                draw_connector(
                    draw,
                    start_candidates[0].x2,
                    (min(node.y1 for node in start_candidates) + max(node.y2 for node in start_candidates)) / 2,
                    end_candidates[0].x1,
                    (min(node.y1 for node in end_candidates) + max(node.y2 for node in end_candidates)) / 2,
                    color=connector_fill,
                    width=connector_width,
                    detour_y=detour_y,
                    show_arrows=show_arrows,
                )
            continue

        for start_node in start_layer_list:
            for end_node in end_layer_list:
                if not isinstance(start_node, Ellipses) and not isinstance(end_node, Ellipses):
                    draw_connector(
                        draw,
                        start_node.x2,
                        start_node.y1 + (start_node.y2 - start_node.y1) / 2,
                        end_node.x1,
                        end_node.y1 + (end_node.y2 - end_node.y1) / 2,
                        color=connector_fill,
                        width=connector_width,
                        detour_y=detour_y,
                        show_arrows=show_arrows,
                    )


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
    columns: list[list[TracedLayer]],
    current_x: int,
    show_neurons: bool,
    ellipsize_after: int,
    node_size: int,
    node_spacing: int,
    color_map: dict[Any, Any],
    opacity: int,
    layer_spacing: int,
    color_wheel: ColorWheel,
    outline_width: int = 1,
) -> tuple[list, list, dict, list[list[tuple[str, float, float]]]]:
    """Create nodes of architecture for each layers."""
    id_to_node_list_map = {}
    layers = []
    layer_y = []
    # (label text, x center, y bottom) per layer, grouped by column - a column can hold more
    # than one layer if multiple leaf modules share the same depth (parallel branches).
    label_info: list[list[tuple[str, float, float]]] = []
    for layer_list in columns:
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
                    color_map.get(layer_type, {}).get("fill", color_wheel.get_color(layer_type)),
                    opacity,
                )
                c.outline = color_map.get(layer_type, {}).get(
                    "outline",
                    "black",
                )
                c.outline_width = outline_width

                layer_nodes.append(c)

            id_to_node_list_map[layer.node_id] = layer_nodes
            nodes.extend(layer_nodes)
            label = format_shape_label(layer.output_shape, layer.extra_output_shapes)
            column_labels.append((label, current_x + node_size / 2, current_y))
            current_y += 2 * node_size

        layer_y.append(current_y - node_spacing - 2 * node_size)
        layers.append(nodes)
        label_info.append(column_labels)
        current_x += node_size + layer_spacing
    return layers, layer_y, id_to_node_list_map, label_info
