"""LeNet Style View module for pytorch model visualization."""

# Copyright (C) 2024 Willy Fitra Hendria
# SPDX-License-Identifier: MIT

import warnings
from collections import defaultdict
from collections.abc import Callable
from math import ceil

import aggdraw
import PIL
from PIL import Image, ImageFont
from torch import nn

from ._volumetric_layout import ColumnLayout, VolumetricBox, layout_columns
from .backend import Architecture, extract_architecture
from .connectors import compute_skip_levels, draw_connector
from .utils.layer_utils import Input
from .utils.traced_layer import TracedLayer
from .utils.utils import (
    ColorWheel,
    ImageDraw,
    InputShape,
    StackedBox,
    format_shape_label,
    get_rgba_tuple,
    resolve_palette,
    self_multiply,
)

_LABEL_ROW_HEIGHT = 100


def lenet_view(
    model: nn.Module | nn.Sequential | nn.ModuleList,
    input_shape: InputShape,
    to_file: str | None = None,
    min_z: int = 1,
    min_xy: int = 10,
    max_xy: int = 2000,
    scale_z: float = 1,
    scale_xy: float = 1,
    type_ignore: list | None = None,
    color_map: dict | None = None,
    palette: str = "okabe_ito",
    low_dim_orientation: str = "z",
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
    level_gap: int | None = None,
    show_dimension: bool = True,
    show_input: bool = True,
    outline_width: int = 1,
    connector_fill: str | tuple[int, ...] | None = None,
    connector_width: int = 1,
    one_dim_orientation: str | None = None,
) -> PIL.Image:
    """Generate a LeNet style architecture visualization for a given torch model.

    TODO: remove unnecessary arguments for this LeNet style architecture.

    Args:
        model (torch.nn.Module): A torch model that will be visualized.
        input_shape (tuple): The shape of the input tensor (default: (1, 3, 224, 224)). For a
            model whose forward() takes multiple separate input tensors, pass a tuple of
            per-tensor shapes instead, one per positional argument in order, e.g.
            ((1, 3, 224, 224), (1, 10)).
        to_file (str, optional): Path to the file to write the created image. Overwrite if exist.
            Image type is inferred from the file extension. Providing None will disable writing.
        min_z (int, optional): Minimum size in pixels that a layer will have along the z-axis.
        min_xy (int, optional): Minimum size in pixels that a layer will have along the x and y axes.
        max_channels (int, optional): Maximum number of channels.
        max_xy (int, optional): Maximum size in pixels that a layer will have along the x and y axes.
        scale_z (float, optional): Scalar multiplier for the size of each layer along the z-axis.
        scale_xy (float, optional): Scalar multiplier for the size of each layer along the x and y axes.
        type_ignore (list, optional): List of layer types in the torch model to ignore during drawing.
        color_map (dict, optional): Dictionary defining fill and outline colors for each layer by class type.
            Will fallback to default values for unspecified classes.
        palette (str, optional): Named color palette used as the fallback for any layer type not
            given an explicit override via `color_map`. One of `"okabe_ito"` (default,
            colorblind-safe), `"tol_bright"`, `"tol_muted"`, `"tab10"`, `"grayscale"`, `"nord"`,
            `"dracula"`, `"gruvbox"`, `"solarized"`, `"material"`, `"catppuccin"`.
        low_dim_orientation (str, optional): Axis on which a layer without real spatial/channel
            structure (a 1D shape, or a 2D shape like an RNN/attention layer's
            `(seq_len, hidden_size)`) should be drawn. One of `'x'`, `'y'`, or `'z'`.
        background_fill (str or tuple, optional): Background color for the image. A string or a tuple (R, G, B, A).
        padding (int, optional): Distance in pixels before the first and after the last layer.
        spacing (int, optional): Spacing in pixels between two layers.
        draw_funnel (bool, optional): If True, a funnel will be drawn between consecutive layers.
        shade_step (int, optional): Deviation in lightness for drawing shades (only in volumetric view).
        font (PIL.ImageFont, optional): Font that will be used for the legend. If None, default font will be used.
        font_color (str or tuple, optional): Color for the font if used. Can be a string or a tuple (R, G, B, A).
        opacity (int): Transparency of the color (0 ~ 255).
        offset_z (int): control the offset of overlapping between channels.
        level_gap (int, optional): Vertical spacing in pixels between stacked skip-connection detour
            routes. If None, defaults to 50.
        show_dimension (bool, optional): If True (the default), print each layer's output shape
            below it. For a model with parallel branches (e.g. multi-branch merges or multiple
            input tensors), several boxes can share a column and their labels may overlap - set
            this to False to drop the labels entirely in that case.
        show_input (bool, optional): For a single-input model, whether to draw the synthetic
            "Input" box. Defaults to True. Set False to hide it - e.g. if you're overlaying your
            own custom input illustration instead. Has no effect on a multi-input model, where
            every input is always shown (omitting any of them would make it ambiguous which
            arrow belongs to which named input). Ignored (input always kept) when the input feeds
            more than one consumer, e.g. a residual shortcut, since dropping it would silently
            discard that edge.
        outline_width (int, optional): Line width in pixels for the shape borders. Defaults to 1.
        connector_fill (str or tuple, optional): Color for skip-connection lines. Can be a string
            or a tuple (R, G, B, A). If None, inherits the target box's outline color.
        connector_width (int, optional): Line width in pixels for skip-connection lines. Defaults to 1.
        one_dim_orientation (str, optional): Deprecated, use `low_dim_orientation` instead.

    Returns:
        PIL.Image: An Image object representing the generated architecture visualization.
    """
    if one_dim_orientation is not None:
        warnings.warn(
            "`one_dim_orientation` is deprecated and will be removed in a future release, "
            "use `low_dim_orientation` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        low_dim_orientation = one_dim_orientation

    if type_ignore is None:
        type_ignore = []

    _color_map: dict = {}
    if color_map is not None:
        _color_map = defaultdict(dict, color_map)

    architecture = extract_architecture(model, input_shape)

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

    filtered_columns = [[layer for layer in column if type(layer.module) not in type_ignore] for column in raw_columns]
    filtered_columns = [column for column in filtered_columns if column]

    layer_types: list[type] = []
    make_box = _box_factory(
        low_dim_orientation,
        scale_xy,
        min_xy,
        max_xy,
        scale_z,
        min_z,
        max_channels,
        shade_step,
        _color_map,
        opacity,
        offset_z,
        layer_types,
        ColorWheel(colors=resolve_palette(palette)),
        outline_width,
    )
    column_layout = layout_columns(
        filtered_columns,
        make_box,
        _gap_for(spacing),
        spacing,
        padding,
        top_offset_for=lambda _box: float(padding),
        x_shift_for=_x_shift_for,
        x_extra_for=_x_extra_for,
        right_extent_for=_right_extent_for,
    )

    edge_to_level, num_levels = compute_skip_levels(
        architecture.edges(),
        architecture.id_to_column,
        lambda *_: True,
        column_layout.bbox_for,
    )
    resolved_level_gap = level_gap if level_gap is not None else 50
    top_margin_for_skips = num_levels * resolved_level_gap

    if font is None:
        font = ImageFont.load_default()

    # StackedBox draws de*offset_z worth of offset copies both above and below its nominal
    # y1/y2, which the generic column layout doesn't account for (it only tracks y1/y2 span),
    # so extra headroom is reserved here, split evenly above and below each column's stack.
    spread_margin = max(
        (
            getattr(box, "de", 0) * getattr(box, "offset_z", 0)
            for column in column_layout.boxes_by_column
            for box in column
        ),
        default=0,
    )

    label_row_height = _LABEL_ROW_HEIGHT if show_dimension else 0
    img_width = column_layout.img_width
    diagram_height = top_margin_for_skips + spread_margin + column_layout.diagram_height
    img = Image.new(
        "RGBA",
        (int(ceil(img_width)), ceil(diagram_height) + label_row_height),
        background_fill,
    )
    draw = aggdraw.Draw(img)

    _apply_centering(column_layout, top_margin_for_skips + spread_margin / 2)

    _draw_funnels_and_boxes(draw, architecture, column_layout, edge_to_level, draw_funnel)
    _draw_skip_connectors(
        draw,
        architecture,
        column_layout,
        edge_to_level,
        num_levels,
        padding,
        resolved_level_gap,
        connector_fill,
        connector_width,
    )

    draw.flush()

    if show_dimension:
        img = _draw_labels(img, column_layout.boxes_by_column, font, font_color)

    if to_file is not None:
        img.save(to_file)

    return img


def _gap_for(spacing: int) -> Callable[[VolumetricBox], float]:
    def gap(box: VolumetricBox) -> float:
        return max(spacing, getattr(box, "de", 0) * getattr(box, "offset_z", 0))

    return gap


def _x_shift_for(box: VolumetricBox) -> float:
    return getattr(box, "de", 0) * getattr(box, "offset_z", 0) // 2


def _x_extra_for(box: VolumetricBox) -> float:
    return getattr(box, "de", 0) * getattr(box, "offset_z", 0)


def _right_extent_for(box: VolumetricBox) -> float:
    return getattr(box, "de", 0) * getattr(box, "offset_z", 0) // 2


def _box_factory(
    low_dim_orientation: str,
    scale_xy: float,
    min_xy: int,
    max_xy: int,
    scale_z: float,
    min_z: int,
    max_channels: int,
    shade_step: int,
    color_map: dict,
    opacity: int,
    offset_z: int,
    layer_types: list[type],
    color_wheel: ColorWheel,
    outline_width: int = 1,
) -> Callable[[TracedLayer], StackedBox]:
    """Build a `make_box` callback: given a traced layer, return a sized, unpositioned `StackedBox`."""

    def make_box(layer: TracedLayer) -> StackedBox:
        shape = layer.output_shape[1:]  # drop batch size

        if len(shape) in (1, 2):
            # Neither a 1D nor a 2D shape has real spatial/channel structure - there's nothing
            # to distinguish "channel" from "spatial" the way a genuine (C, H, W) feature map
            # does. Take the last value (for 2D, e.g. an RNN/attention layer's
            # (seq_len, hidden_size), this is the feature/channel-like one, matching PyTorch's
            # (..., seq, feature) convention for sequence data; for 1D it's the only value) and
            # let the user place it on whichever axis they choose, same as any 1D value - the
            # positional-like dim, if any, is discarded either way.
            value = shape[-1]
            if low_dim_orientation in ("x", "y", "z"):
                shape = (1,) * "cxyz".index(low_dim_orientation) + (value,)
            else:
                error_msg = f"unsupported orientation: {low_dim_orientation}"
                raise ValueError(error_msg)

        ori_shape = shape
        shape = shape + (1,) * (4 - len(shape))  # expand 4D.

        x = min(max(shape[1] * scale_xy, min_xy), max_xy)
        y = min(max(shape[2] * scale_xy, min_xy), max_xy)
        z = min(max(int(self_multiply(shape[0:1] + shape[3:]) * scale_z), min_z), max_channels)

        layer_type = type(layer.module)
        if layer_type not in layer_types:
            layer_types.append(layer_type)

        box = StackedBox()
        box.offset_z = offset_z
        box.label = layer.module.name() if isinstance(layer.module, Input) else layer_type.__name__
        box.output_shape = tuple(ori_shape)
        box.extra_output_shapes = layer.extra_output_shapes
        box.de = z

        box.x1 = 0
        box.y1 = 0
        box.x2 = x
        box.y2 = y

        box.set_fill(
            color_map.get(layer_type, {}).get("fill", color_wheel.get_color(layer_type)),
            opacity,
        )
        box.outline = color_map.get(layer_type, {}).get("outline", "black")
        box.outline_width = outline_width
        color_map[layer_type] = {"fill": box.fill, "outline": box.outline}

        box.shade = shade_step
        return box

    return make_box


def _apply_centering(column_layout: ColumnLayout, top_margin: float) -> None:
    """Center each column vertically within the band below the reserved top margin."""
    for column, column_height in zip(column_layout.boxes_by_column, column_layout.column_heights, strict=True):
        y_off = top_margin + (column_layout.diagram_height - column_height) / 2
        for box in column:
            box.y1 += y_off
            box.y2 += y_off
            box.x1 += column_layout.x_off
            box.x2 += column_layout.x_off


def _draw_stacked_funnel(draw: aggdraw.Draw, start_box: VolumetricBox, end_box: VolumetricBox) -> None:
    """Draw a tapered funnel connecting the outer bounds of two boxes' offset-copy stacks."""
    pen = aggdraw.Pen(get_rgba_tuple(end_box.outline))
    start_de, start_offset_z = getattr(start_box, "de", 0), getattr(start_box, "offset_z", 0)
    end_de, end_offset_z = getattr(end_box, "de", 0), getattr(end_box, "offset_z", 0)
    start_off = -start_offset_z * start_de // 2
    end_off = -end_offset_z * end_de // 2

    draw.line([start_box.x2 + start_off, start_box.y1 + start_off, end_box.x1 + end_off, end_box.y1 + end_off], pen)

    start_off += start_offset_z * (start_de - 1)
    end_off += end_offset_z * (end_de - 1)
    draw.line([start_box.x2 + start_off, start_box.y2 + start_off, end_box.x1 + end_off, end_box.y2 + end_off], pen)


def _draw_funnels_and_boxes(
    draw: aggdraw.Draw,
    architecture: Architecture,
    column_layout: ColumnLayout,
    edge_to_level: dict[tuple[str, str], int],
    draw_funnel_flag: bool,
) -> None:
    """Draw each column's incoming funnels, then that column's own boxes, column by column.

    This interleaving matters: a funnel is drawn *before* the box it points to (so the box's own
    fill covers the funnel's far end) but *after* the column before it. Drawing every connector
    first and every box second - simpler, but wrong - would let each box's fill blot out large
    parts of its own incoming funnel whenever neighboring layers have a very different `de`/
    `offset_z` spread, which is the common case for a real CNN.
    """
    if draw_funnel_flag:
        incoming_funnels: dict[str, list[str]] = {}
        for start_id, end_id in architecture.edges():
            if edge_to_level.get((start_id, end_id)) is None and end_id in column_layout.id_to_box:
                incoming_funnels.setdefault(end_id, []).append(start_id)
        layer_id_by_box: dict[int, str] = {id(box): layer_id for layer_id, box in column_layout.id_to_box.items()}

    for column in column_layout.boxes_by_column:
        if draw_funnel_flag:
            for box in column:
                layer_id = layer_id_by_box[id(box)]
                for start_id in incoming_funnels.get(layer_id, []):
                    if start_id in column_layout.id_to_box:
                        _draw_stacked_funnel(draw, column_layout.id_to_box[start_id], box)

        for box in column:
            box.draw(draw)


def _draw_skip_connectors(
    draw: aggdraw.Draw,
    architecture: Architecture,
    column_layout: ColumnLayout,
    edge_to_level: dict[tuple[str, str], int],
    num_levels: int,
    padding: int,
    resolved_level_gap: int,
    connector_fill: str | tuple[int, ...] | None = None,
    connector_width: int = 1,
) -> None:
    """Draw every skip-connection edge, routed above the diagram.

    Drawn in a separate pass after every funnel and box, so a routed line is always visible on
    top rather than potentially hidden under a box it happens to cross over. Always drawn
    regardless of `draw_funnel` - a funnel implies a continuous volume flowing between two
    layers, which a skip connection isn't, and it should never become invisible just because
    funnels are toggled off.

    Each endpoint is anchored at the top corner of the frontmost of the box's stacked slices
    (`front_offset`), not the unshifted, centered reference coordinates - the center slice sits
    behind several slices stacked in front of it, so a line anchored there would appear to run
    through them. The top corner (rather than that slice's mid-height) matches the line's own
    always-straight-up-first routing and mirrors flow_view's equivalent ridge-point anchor.
    """
    for start_id, end_id in architecture.edges():
        level = edge_to_level.get((start_id, end_id))
        if level is None or start_id not in column_layout.id_to_box or end_id not in column_layout.id_to_box:
            continue

        start_box = column_layout.id_to_box[start_id]
        end_box = column_layout.id_to_box[end_id]
        detour_y = padding + (num_levels - 1 - level) * resolved_level_gap
        start_front = start_box.front_offset() if isinstance(start_box, StackedBox) else 0
        end_front = end_box.front_offset() if isinstance(end_box, StackedBox) else 0
        resolved_color = connector_fill if connector_fill is not None else end_box.outline
        draw_connector(
            draw,
            start_box.x2 + start_front,
            start_box.y1 + start_front,
            end_box.x1 + end_front,
            end_box.y1 + end_front,
            color=resolved_color,
            width=connector_width,
            detour_y=detour_y,
        )


def _draw_labels(
    img: PIL.Image,
    boxes_by_column: list[list[VolumetricBox]],
    font: ImageFont,
    font_color: str | tuple[int, ...],
) -> PIL.Image:
    """Draw every box's label (on by default, unlike flow_view's opt-in show_dimension)."""
    text_img = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw_text = ImageDraw.Draw(text_img)

    for column in boxes_by_column:
        for box in column:
            loc_x = box.x1 + (box.x2 - box.x1) // 4
            label = getattr(box, "label", type(box).__name__)
            shape_label = format_shape_label(box.output_shape, box.extra_output_shapes)
            draw_text.text((loc_x, img.height - 50), f"{label} {shape_label}", font=font, fill=font_color)

    return Image.alpha_composite(img, text_img)
