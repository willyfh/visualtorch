"""Flow View module for pytorch model visualization."""

# Copyright (C) 2020 Paul Gavrikov
# Copyright (C) 2024 VisualTorch Contributors
# SPDX-License-Identifier: MIT

import warnings
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from math import ceil
from typing import Literal

import aggdraw
import PIL
from PIL import Image, ImageFont
from torch import nn

from ._animation import animate_frames
from ._volumetric_layout import ColumnLayout, VolumetricBox, layout_columns
from .backend import Architecture, extract_architecture
from .connectors import compute_skip_levels, draw_connector
from .utils.traced_layer import TracedLayer
from .utils.utils import (
    Box,
    ColorWheel,
    ImageDraw,
    InputDtype,
    InputShape,
    format_shape_label,
    get_rgba_tuple,
    linear_layout,
    resolve_palette,
    self_multiply,
)

LegendPosition = Literal["top-left", "top-right", "top-center", "bottom-left", "bottom-right", "bottom-center"]
_LEGEND_POSITIONS: tuple[LegendPosition, ...] = (
    "top-left",
    "top-right",
    "top-center",
    "bottom-left",
    "bottom-right",
    "bottom-center",
)


def flow_view(
    model: nn.Module | nn.Sequential | nn.ModuleList,
    input_shape: InputShape,
    input_dtype: InputDtype | None = None,
    to_file: str | None = None,
    min_z: int = 10,
    min_xy: int = 10,
    max_z: int = 400,
    max_xy: int = 2000,
    scale_z: float = 0.1,
    scale_xy: float = 1,
    type_ignore: list | None = None,
    color_map: dict | None = None,
    palette: str = "okabe_ito",
    low_dim_orientation: str = "z",
    background_fill: str | tuple[int, ...] = "white",
    draw_volume: bool = True,
    padding: int = 10,
    spacing: int = 10,
    draw_funnel: bool = True,
    shade_step: int = 10,
    legend: bool = False,
    font: ImageFont = None,
    font_color: str | tuple[int, ...] = "black",
    opacity: int = 255,
    show_dimension: bool = False,
    level_gap: int | None = None,
    show_input: bool = True,
    outline_width: int = 1,
    connector_fill: str | tuple[int, ...] | None = None,
    connector_width: int = 1,
    one_dim_orientation: str | None = None,
    legend_position: LegendPosition = "bottom-left",
) -> PIL.Image:
    """Generate a flow-style architecture visualization for a given torch model.

    Args:
        model (torch.nn.Module): A torch model that will be visualized.
        input_shape (tuple): The shape of the input tensor (default: (1, 3, 224, 224)). For a
            model whose forward() takes multiple separate input tensors, pass a tuple of
            per-tensor shapes instead, one per positional argument in order, e.g.
            ((1, 3, 224, 224), (1, 10)).
        input_dtype (torch.dtype, optional): The dtype of the dummy input tensor(s) used to trace
            the model. Defaults to `None` for every input, giving a uniformly random float
            tensor. Needed for a model that starts with an integer/bool-input layer (e.g.
            `nn.Embedding`, which rejects a float index tensor): pass `torch.long`, or - for a
            model with multiple input tensors of different kinds - a tuple of per-tensor dtypes,
            e.g. `(torch.long, None)`.
        to_file (str, optional): Path to the file to write the created image. Overwrite if exist.
            Image type is inferred from the file extension. Providing None will disable writing.
        min_z (int, optional): Minimum size in pixels that a layer will have along the z-axis.
        min_xy (int, optional): Minimum size in pixels that a layer will have along the x and y axes.
        max_z (int, optional): Maximum size in pixels that a layer will have along the z-axis.
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
        draw_volume (bool, optional): Flag to switch between 3D volumetric view and 2D box view.
        padding (int, optional): Distance in pixels before the first and after the last layer.
        spacing (int, optional): Spacing in pixels between two layers.
        draw_funnel (bool, optional): If True, a funnel will be drawn between consecutive layers.
        shade_step (int, optional): Deviation in lightness for drawing shades (only in volumetric view).
        legend (bool, optional): Add a legend of the layers to the image.
        font (PIL.ImageFont, optional): Font that will be used for the legend. If None, default font will be used.
        font_color (str or tuple, optional): Color for the font if used. Can be a string or a tuple (R, G, B, A).
        opacity (int): Transparency of the color (0 ~ 255).
        show_dimension (bool, optional): If True, print each layer's output shape below it.
        level_gap (int, optional): Vertical spacing in pixels between stacked skip-connection detour
            routes. If None, defaults to 50.
        show_input (bool, optional): For a single-input model, whether to draw the synthetic
            "Input" box. Defaults to True - e.g. so an autoencoder's or U-Net's output shape has
            a visible size-matched partner on the input side. Set False to hide it - e.g. if
            you're overlaying your own custom input illustration instead. Has no effect on a
            multi-input model, where every input is always shown (omitting any of them would
            make it ambiguous which arrow belongs to which named input).
        outline_width (int, optional): Line width in pixels for the shape borders. Defaults to 1.
        connector_fill (str or tuple, optional): Color for funnel and skip-connection lines. Can be a string
            or a tuple (R, G, B, A). If None, inherits the target box's outline color.
        connector_width (int, optional): Line width in pixels for funnel and skip-connection lines. Defaults to 1.
        one_dim_orientation (str, optional): Deprecated, use `low_dim_orientation` instead.
        legend_position (str, optional): Where to place the legend when `legend=True`. One of
            `"top-left"`, `"top-right"`, `"top-center"`, `"bottom-left"`, `"bottom-right"`, or
            `"bottom-center"`. Defaults to `"bottom-left"`, matching the previous layout.

    Returns:
        PIL.Image: An Image object representing the generated architecture visualization.
    """
    setup = _prepare_flow_render(
        model,
        input_shape,
        input_dtype,
        min_z,
        min_xy,
        max_z,
        max_xy,
        scale_z,
        scale_xy,
        type_ignore,
        color_map,
        palette,
        low_dim_orientation,
        draw_volume,
        padding,
        spacing,
        shade_step,
        font,
        opacity,
        show_dimension,
        legend,
        level_gap,
        show_input,
        outline_width,
        one_dim_orientation,
        legend_position,
        background_fill,
    )

    img = _draw_diagram_frame(
        setup,
        reveal_up_to=len(setup.column_layout.boxes_by_column) - 1,
        draw_funnel_flag=draw_funnel,
        connector_fill=connector_fill,
        connector_width=connector_width,
        padding=padding,
        show_dimension=show_dimension,
        font_color=font_color,
        legend=legend,
        draw_volume=draw_volume,
        shade_step=shade_step,
        opacity=opacity,
        spacing=spacing,
        background_fill=background_fill,
        legend_position=legend_position,
    )

    if to_file is not None:
        img.save(to_file)

    return img


@dataclass
class _FlowRenderSetup:
    """Everything computed once, up front, and reused by every frame (static or animated)."""

    architecture: Architecture
    column_layout: ColumnLayout
    edge_to_level: dict[tuple[str, str], int]
    num_levels: int
    resolved_level_gap: int
    font: ImageFont
    layer_types: list[type]
    color_map: dict
    diagram_height: float
    img_template: Image.Image


def _prepare_flow_render(
    model: nn.Module | nn.Sequential | nn.ModuleList,
    input_shape: InputShape,
    input_dtype: InputDtype | None,
    min_z: int,
    min_xy: int,
    max_z: int,
    max_xy: int,
    scale_z: float,
    scale_xy: float,
    type_ignore: list | None,
    color_map: dict | None,
    palette: str,
    low_dim_orientation: str,
    draw_volume: bool,
    padding: int,
    spacing: int,
    shade_step: int,
    font: ImageFont,
    opacity: int,
    show_dimension: bool,
    legend: bool,
    level_gap: int | None,
    show_input: bool,
    outline_width: int,
    one_dim_orientation: str | None,
    legend_position: LegendPosition,
    background_fill: str | tuple[int, ...],
) -> _FlowRenderSetup:
    """Trace the model and compute the full layout/canvas once - shared by every frame."""
    _validate_legend_position(legend_position)

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

    architecture = extract_architecture(model, input_shape, input_dtype)

    # The synthetic input column has no counterpart in this style (flow_view never drew an
    # "Input" box even before the v2 backend unification), so a single input is dropped by
    # default - unless it feeds more than one consumer, e.g. a residual block whose shortcut is
    # the raw input (`identity = x`): dropping the input node would silently discard that edge (it
    # would reference a node with no box to connect), making the skip invisible rather than
    # routed. For a model with 2+ separate input tensors, always show every input box instead -
    # hiding any of them would make it ambiguous which arrow originates from which named input.
    input_column = architecture.columns[0]
    if len(input_column) == 1:
        # Hiding is only honored when it's safe to: an input feeding more than one consumer
        # (e.g. a residual block's raw-input shortcut) must stay visible regardless of
        # show_input, since dropping it would silently discard that edge rather than just
        # tidying up the look of a plain single-consumer chain.
        input_node_id = input_column[0].node_id
        input_out_degree = int(architecture.adjacency[architecture.id_to_index[input_node_id]].sum())
        keep_input = show_input or input_out_degree > 1
        raw_columns = architecture.columns if keep_input else architecture.columns[1:]
    else:
        raw_columns = architecture.columns

    filtered_columns = [[layer for layer in column if type(layer.module) not in type_ignore] for column in raw_columns]
    filtered_columns = [column for column in filtered_columns if column]

    layer_types: list[type] = []
    color_wheel = ColorWheel(colors=resolve_palette(palette))
    make_box = _box_factory(
        low_dim_orientation,
        scale_xy,
        min_xy,
        max_xy,
        scale_z,
        min_z,
        max_z,
        draw_volume,
        shade_step,
        _color_map,
        opacity,
        color_wheel,
        layer_types,
        outline_width,
    )
    column_layout = layout_columns(
        filtered_columns,
        make_box,
        lambda box: max(spacing, getattr(box, "de", 0)),
        spacing,
        padding,
    )

    edge_to_level, num_levels = compute_skip_levels(
        (
            edge
            for edge in architecture.edges()
            if edge[0] in column_layout.id_to_box and edge[1] in column_layout.id_to_box
        ),
        architecture.id_to_column,
        lambda *_: True,
        column_layout.bbox_for,
    )
    resolved_level_gap = level_gap if level_gap is not None else 50
    top_margin_for_skips = num_levels * resolved_level_gap

    if font is None and (show_dimension or legend):
        font = ImageFont.load_default()

    # Reserve a fixed-height row below the diagram for shape labels, measured up front so
    # boxes can still be centered within just the diagram area (not the extended canvas),
    # and so labels never clip or overlap the diagram itself.
    label_row_height = 0
    img_width = column_layout.img_width
    if show_dimension:
        label_row_height = font.getbbox("Ag")[3] + 5
        fitted_max_right = _fit_dimension_labels(
            column_layout.boxes_by_column,
            font,
            column_layout.x_off,
            column_layout.max_right,
        )
        img_width = fitted_max_right + column_layout.x_off + padding

    # Generate image
    diagram_height = top_margin_for_skips + column_layout.diagram_height
    img_template = Image.new(
        "RGBA",
        (int(ceil(img_width)), int(ceil(diagram_height + label_row_height))),
        background_fill,
    )

    _apply_centering(column_layout, top_margin_for_skips, column_layout.diagram_height)

    return _FlowRenderSetup(
        architecture=architecture,
        column_layout=column_layout,
        edge_to_level=edge_to_level,
        num_levels=num_levels,
        resolved_level_gap=resolved_level_gap,
        font=font,
        layer_types=layer_types,
        color_map=_color_map,
        diagram_height=diagram_height,
        img_template=img_template,
    )


def _draw_diagram_frame(
    setup: _FlowRenderSetup,
    reveal_up_to: int,
    draw_funnel_flag: bool,
    connector_fill: str | tuple[int, ...] | None,
    connector_width: int,
    padding: int,
    show_dimension: bool,
    font_color: str | tuple[int, ...],
    legend: bool,
    draw_volume: bool,
    shade_step: int,
    opacity: int,
    spacing: int,
    background_fill: str | tuple[int, ...],
    legend_position: LegendPosition,
) -> PIL.Image:
    """Draw a single frame: everything in columns `0..reveal_up_to`, on a copy of the shared canvas.

    `_draw_funnels_and_boxes`/`_draw_skip_connectors` already gate every box/edge on
    `column_layout.id_to_box` membership, so passing a `ColumnLayout` whose `boxes_by_column`/
    `id_to_box` are truncated to the revealed columns is sufficient - no separate edge-list
    filtering is needed here (unlike `graph.py`, whose `_draw_connectors` takes a raw edge list).

    The static render calls this once with `reveal_up_to` set to the last column - an animated
    render calls it once per column, each time with one more column revealed than the last.
    """
    img = setup.img_template.copy()
    draw = aggdraw.Draw(img)

    visible_boxes_by_column = setup.column_layout.boxes_by_column[: reveal_up_to + 1]
    revealed_box_ids = {id(box) for column in visible_boxes_by_column for box in column}
    visible_layout = replace(
        setup.column_layout,
        boxes_by_column=visible_boxes_by_column,
        id_to_box={
            node_id: box for node_id, box in setup.column_layout.id_to_box.items() if id(box) in revealed_box_ids
        },
    )

    _draw_funnels_and_boxes(
        draw,
        setup.architecture,
        visible_layout,
        setup.edge_to_level,
        draw_funnel_flag,
        connector_fill,
        connector_width,
    )
    _draw_skip_connectors(
        draw,
        setup.architecture,
        visible_layout,
        setup.edge_to_level,
        setup.num_levels,
        padding,
        setup.resolved_level_gap,
        connector_fill,
        connector_width,
    )

    draw.flush()

    # Print each layer's output shape in the reserved row below the diagram
    if show_dimension:
        img = _draw_dimension_labels(img, visible_boxes_by_column, setup.diagram_height, setup.font, font_color)

    # Create layer color legend - drawn identically (full, unchanging) on every frame, since
    # `layer_types` was collected across the whole model during the one upfront layout pass, not
    # per-frame. Keeping it fixed from the first frame avoids the canvas resizing mid-animation
    # that progressively revealing legend entries would otherwise cause.
    if legend:
        img = _draw_legend(
            img,
            setup.layer_types,
            setup.color_map,
            setup.font,
            font_color,
            draw_volume,
            shade_step,
            opacity,
            spacing,
            padding,
            background_fill,
            legend_position,
        )

    return img


def _flow_view_animate(
    model: nn.Module | nn.Sequential | nn.ModuleList,
    input_shape: InputShape,
    input_dtype: InputDtype | None = None,
    to_file: str | None = None,
    min_z: int = 10,
    min_xy: int = 10,
    max_z: int = 400,
    max_xy: int = 2000,
    scale_z: float = 0.1,
    scale_xy: float = 1,
    type_ignore: list | None = None,
    color_map: dict | None = None,
    palette: str = "okabe_ito",
    low_dim_orientation: str = "z",
    background_fill: str | tuple[int, ...] = "white",
    draw_volume: bool = True,
    padding: int = 10,
    spacing: int = 10,
    draw_funnel: bool = True,
    shade_step: int = 10,
    legend: bool = False,
    font: ImageFont = None,
    font_color: str | tuple[int, ...] = "black",
    opacity: int = 255,
    show_dimension: bool = False,
    level_gap: int | None = None,
    show_input: bool = True,
    outline_width: int = 1,
    connector_fill: str | tuple[int, ...] | None = None,
    connector_width: int = 1,
    one_dim_orientation: str | None = None,
    legend_position: LegendPosition = "bottom-left",
    frame_duration: int = 600,
    final_hold_duration: int = 1500,
    loop: bool = True,
) -> list[PIL.Image] | None:
    """Generate an animated GIF revealing a flow-style visualization one column at a time.

    Every parameter matches `flow_view()` exactly (same behavior, same defaults), plus the 3
    animation-only parameters at the end - see `to_file`/the return value note below for the one
    behavioral difference.

    Args:
        model (torch.nn.Module): A torch model that will be visualized.
        input_shape (tuple): The shape of the input tensor (default: (1, 3, 224, 224)). For a
            model whose forward() takes multiple separate input tensors, pass a tuple of
            per-tensor shapes instead, one per positional argument in order, e.g.
            ((1, 3, 224, 224), (1, 10)).
        input_dtype (torch.dtype, optional): The dtype of the dummy input tensor(s) used to trace
            the model. Defaults to `None` for every input, giving a uniformly random float
            tensor. Needed for a model that starts with an integer/bool-input layer (e.g.
            `nn.Embedding`, which rejects a float index tensor): pass `torch.long`, or - for a
            model with multiple input tensors of different kinds - a tuple of per-tensor dtypes,
            e.g. `(torch.long, None)`.
        to_file (str, optional): Path to write the animated GIF to. See the Returns note below -
            unlike `flow_view()`, providing None returns the frame list instead of writing a file.
        min_z (int, optional): Minimum size in pixels that a layer will have along the z-axis.
        min_xy (int, optional): Minimum size in pixels that a layer will have along the x and y axes.
        max_z (int, optional): Maximum size in pixels that a layer will have along the z-axis.
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
        draw_volume (bool, optional): Flag to switch between 3D volumetric view and 2D box view.
        padding (int, optional): Distance in pixels before the first and after the last layer.
        spacing (int, optional): Spacing in pixels between two layers.
        draw_funnel (bool, optional): If True, a funnel will be drawn between consecutive layers.
        shade_step (int, optional): Deviation in lightness for drawing shades (only in volumetric view).
        legend (bool, optional): Add a legend of the layers to the image.
        font (PIL.ImageFont, optional): Font that will be used for the legend. If None, default font will be used.
        font_color (str or tuple, optional): Color for the font if used. Can be a string or a tuple (R, G, B, A).
        opacity (int): Transparency of the color (0 ~ 255).
        show_dimension (bool, optional): If True, print each layer's output shape below it.
        level_gap (int, optional): Vertical spacing in pixels between stacked skip-connection detour
            routes. If None, defaults to 50.
        show_input (bool, optional): For a single-input model, whether to draw the synthetic
            "Input" box. Defaults to True - e.g. so an autoencoder's or U-Net's output shape has
            a visible size-matched partner on the input side. Set False to hide it - e.g. if
            you're overlaying your own custom input illustration instead. Has no effect on a
            multi-input model, where every input is always shown (omitting any of them would
            make it ambiguous which arrow belongs to which named input).
        outline_width (int, optional): Line width in pixels for the shape borders. Defaults to 1.
        connector_fill (str or tuple, optional): Color for funnel and skip-connection lines. Can be a string
            or a tuple (R, G, B, A). If None, inherits the target box's outline color.
        connector_width (int, optional): Line width in pixels for funnel and skip-connection lines. Defaults to 1.
        one_dim_orientation (str, optional): Deprecated, use `low_dim_orientation` instead.
        legend_position (str, optional): Where to place the legend when `legend=True`. One of
            `"top-left"`, `"top-right"`, `"top-center"`, `"bottom-left"`, `"bottom-right"`, or
            `"bottom-center"`. Defaults to `"bottom-left"`, matching the previous layout.
        frame_duration (int, optional): Milliseconds each intermediate frame is displayed.
        final_hold_duration (int, optional): Milliseconds the final, fully-revealed frame is
            displayed before the GIF loops - gives a viewer time to actually see the complete
            diagram rather than immediately looping.
        loop (bool, optional): If True, the GIF loops forever. If False, it plays once.

    Note:
        GIF output does not support full alpha transparency - a translucent `background_fill` is
        flattened by the GIF format itself, unlike the static `flow_view()`'s PNG output.

    Returns:
        list[Image.Image] | None: The rendered frames (one per column, in reveal order) if
        `to_file` is None. If `to_file` is given, the GIF is written to that path and `None` is
        returned instead - unlike `flow_view()`, which always returns the image even when also
        writing to a file. `to_file` should end in `.gif`; a mismatched extension will not raise,
        it will silently save only the first frame.
    """
    setup = _prepare_flow_render(
        model,
        input_shape,
        input_dtype,
        min_z,
        min_xy,
        max_z,
        max_xy,
        scale_z,
        scale_xy,
        type_ignore,
        color_map,
        palette,
        low_dim_orientation,
        draw_volume,
        padding,
        spacing,
        shade_step,
        font,
        opacity,
        show_dimension,
        legend,
        level_gap,
        show_input,
        outline_width,
        one_dim_orientation,
        legend_position,
        background_fill,
    )

    def render_frame(reveal_up_to: int) -> Image.Image:
        return _draw_diagram_frame(
            setup,
            reveal_up_to,
            draw_funnel,
            connector_fill,
            connector_width,
            padding,
            show_dimension,
            font_color,
            legend,
            draw_volume,
            shade_step,
            opacity,
            spacing,
            background_fill,
            legend_position,
        )

    return animate_frames(
        render_frame,
        n_columns=len(setup.column_layout.boxes_by_column),
        to_file=to_file,
        frame_duration=frame_duration,
        final_hold_duration=final_hold_duration,
        loop=loop,
    )


def _validate_legend_position(legend_position: str) -> None:
    if legend_position not in _LEGEND_POSITIONS:
        supported = ", ".join(f"{position!r}" for position in _LEGEND_POSITIONS)
        error_msg = f"unsupported legend_position: {legend_position!r}. Supported positions: {supported}."
        raise ValueError(error_msg)


def _box_factory(
    low_dim_orientation: str,
    scale_xy: float,
    min_xy: int,
    max_xy: int,
    scale_z: float,
    min_z: int,
    max_z: int,
    draw_volume: bool,
    shade_step: int,
    color_map: dict,
    opacity: int,
    color_wheel: ColorWheel,
    layer_types: list[type],
    outline_width: int = 1,
) -> Callable[[TracedLayer], Box]:
    """Build a `make_box` callback: given a traced layer, return a sized, unpositioned `Box`."""

    def make_box(layer: TracedLayer) -> Box:
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
        z = min(max(int(self_multiply(shape[0:1] + shape[3:]) * scale_z), min_z), max_z)

        layer_type = type(layer.module)
        if layer_type not in layer_types:
            layer_types.append(layer_type)

        box = Box()
        box.output_shape = tuple(ori_shape)
        box.extra_output_shapes = layer.extra_output_shapes
        box.de = int(x / 3) if draw_volume else 0

        box.x1 = 0
        box.y1 = 0
        box.x2 = z
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


def _apply_centering(column_layout: ColumnLayout, top_margin_for_skips: float, band_height: float) -> None:
    """Center each column vertically within the band below the reserved skip-detour strip."""
    for column, column_height in zip(column_layout.boxes_by_column, column_layout.column_heights, strict=True):
        y_off = top_margin_for_skips + (band_height - column_height) / 2
        for box in column:
            box.y1 += y_off
            box.y2 += y_off
            box.x1 += column_layout.x_off
            box.x2 += column_layout.x_off


def _draw_funnel(
    draw: aggdraw.Draw,
    start_box: VolumetricBox,
    end_box: VolumetricBox,
    connector_fill: str | tuple[int, ...] | None = None,
    connector_width: int = 1,
) -> None:
    """Draw a tapered funnel connecting two boxes in adjacent columns."""
    resolved_color = connector_fill if connector_fill is not None else end_box.outline
    pen = aggdraw.Pen(get_rgba_tuple(resolved_color), connector_width)
    start_de = getattr(start_box, "de", 0)
    end_de = getattr(end_box, "de", 0)

    draw.line(
        [start_box.x2 + start_de, start_box.y1 - start_de, end_box.x1 + end_de, end_box.y1 - end_de],
        pen,
    )
    draw.line(
        [start_box.x2 + start_de, start_box.y2 - start_de, end_box.x1 + end_de, end_box.y2 - end_de],
        pen,
    )
    draw.line([start_box.x2, start_box.y2, end_box.x1, end_box.y2], pen)
    draw.line([start_box.x2, start_box.y1, end_box.x1, end_box.y1], pen)


def _draw_funnels_and_boxes(
    draw: aggdraw.Draw,
    architecture: Architecture,
    column_layout: ColumnLayout,
    edge_to_level: dict[tuple[str, str], int],
    draw_funnel_flag: bool,
    connector_fill: str | tuple[int, ...] | None = None,
    connector_width: int = 1,
) -> None:
    """Draw each column's incoming funnels, then that column's own boxes, column by column.

    This interleaving matters: a funnel is drawn *before* the box it points to (so the box's own
    fill covers the funnel's far end, matching a real vanishing-point taper) but *after* the
    column before it (so the funnel's near end isn't hidden by a box that's about to be
    re-covered). Drawing every connector first and every box second - simpler, but wrong -
    would let each box's fill blot out large parts of its own incoming funnel whenever
    neighboring layers have a very different `de` (3D depth), which is the common case for a
    real CNN (channel/spatial size changes a lot layer to layer).
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
                        _draw_funnel(draw, column_layout.id_to_box[start_id], box, connector_fill, connector_width)

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

    Each endpoint is anchored at the midpoint of the box's top ridge (the boundary between its
    top face and its side face) rather than the flat front corner, so the line reflects the box's
    3D depth instead of looking like it's attached to a flat 2D rectangle. This midpoint is
    specifically on the box's own outline, not inside either face - anchoring anywhere else on a
    slanted side face would make the line cross back over that face's own fill on its way up to
    the detour, since it always travels straight up first.
    """
    for start_id, end_id in architecture.edges():
        level = edge_to_level.get((start_id, end_id))
        if level is None or start_id not in column_layout.id_to_box or end_id not in column_layout.id_to_box:
            continue

        start_box = column_layout.id_to_box[start_id]
        end_box = column_layout.id_to_box[end_id]
        detour_y = padding + (num_levels - 1 - level) * resolved_level_gap
        start_de = getattr(start_box, "de", 0)
        end_de = getattr(end_box, "de", 0)
        resolved_color = connector_fill if connector_fill is not None else end_box.outline
        draw_connector(
            draw,
            start_box.x2 + start_de / 2,
            start_box.y1 - start_de / 2,
            end_box.x1 + end_de / 2,
            end_box.y1 - end_de / 2,
            color=resolved_color,
            width=connector_width,
            detour_y=detour_y,
        )


def _draw_legend(
    img: PIL.Image,
    layer_types: list[type],
    color_map: dict,
    font: ImageFont,
    font_color: str | tuple[int, ...],
    draw_volume: bool,
    shade_step: int,
    opacity: int,
    spacing: int,
    padding: int,
    background_fill: str | tuple[int, ...],
    legend_position: LegendPosition,
) -> PIL.Image:
    """Build and place a color legend, one entry per layer type."""
    text_height = font.getbbox("Ag")[3]
    cube_size = text_height

    de = 0
    if draw_volume:
        de = cube_size // 2

    patches = []

    for layer_type in layer_types:
        label = layer_type.__name__
        text_size = font.getbbox(label)
        label_patch_size = (cube_size + de + spacing + text_size[2], cube_size + de)
        # this only works if cube_size is bigger than text height

        img_box = Image.new("RGBA", label_patch_size, background_fill)
        img_text = Image.new("RGBA", label_patch_size, (0, 0, 0, 0))
        draw_box = aggdraw.Draw(img_box)
        draw_text = ImageDraw.Draw(img_text)

        box = Box()
        box.x1 = 0
        box.x2 = box.x1 + cube_size
        box.y1 = de
        box.y2 = box.y1 + cube_size
        box.de = de
        box.shade = shade_step
        box.set_fill(color_map.get(layer_type, {}).get("fill", "#000000"), opacity)
        box.outline = color_map.get(layer_type, {}).get("outline", "#000000")
        box.draw(draw_box)

        text_x = box.x2 + box.de + spacing
        text_y = (label_patch_size[1] - text_height) / 2  # 2D center; use text_height and not the current label!
        draw_text.text((text_x, text_y), label, font=font, fill=font_color)

        draw_box.flush()
        img_box.paste(img_text, mask=img_text)
        patches.append(img_box)

    legend_image = linear_layout(
        patches,
        max_width=img.width,
        max_height=img.height,
        padding=padding,
        spacing=spacing,
        background_fill=background_fill,
        horizontal=True,
    )
    return _place_legend(img, legend_image, legend_position, background_fill)


def _place_legend(
    img: PIL.Image,
    legend_image: PIL.Image,
    legend_position: LegendPosition,
    background_fill: str | tuple[int, ...],
) -> PIL.Image:
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


def _column_label_and_center(column: list[VolumetricBox]) -> tuple[str, float]:
    """A column's shape label (joined across branches) and its shared x-center."""
    label = " / ".join(format_shape_label(box.output_shape, box.extra_output_shapes) for box in column)
    center_x = (column[0].x1 + column[0].x2) / 2
    return label, center_x


def _fit_dimension_labels(
    boxes_by_column: list[list[VolumetricBox]],
    font: ImageFont,
    x_off: float,
    max_right: float,
) -> float:
    """Reposition columns so shape labels never overlap each other or clip the canvas edges.

    Closely packed, thin columns would otherwise smear adjacent shape labels together, so the
    gap between columns is widened wherever their labels would collide, and the whole diagram is
    extended (shifted right, if necessary) so the outermost labels - which can be wider than the
    column they belong to - never run past the left or right edge. A column with more than one
    box (a branch) gets one combined label (joined with " / "), since every box in a column
    shares the same x-position and would otherwise draw overlapping labels.

    Returns:
        float: The updated `max_right` bound after any widening/shifting.
    """
    non_empty_columns = [column for column in boxes_by_column if column]
    label_widths = [font.getbbox(_column_label_and_center(column)[0])[2] for column in non_empty_columns]

    shift = 0.0
    prev_center: float | None = None
    prev_label_width = 0.0
    for column, label_width in zip(non_empty_columns, label_widths, strict=True):
        for box in column:
            box.x1 += shift
            box.x2 += shift
        center = (column[0].x1 + column[0].x2) / 2
        if prev_center is not None:
            min_gap = (prev_label_width + label_width) / 2 + 5
            if center - prev_center < min_gap:
                extra = min_gap - (center - prev_center)
                for box in column:
                    box.x1 += extra
                    box.x2 += extra
                shift += extra
                center += extra
        prev_center = center
        prev_label_width = label_width
    max_right += shift

    extra_left = 0.0
    extra_right = 0.0
    for column, label_width in zip(non_empty_columns, label_widths, strict=True):
        center = (column[0].x1 + column[0].x2) / 2
        extra_left = max(extra_left, label_width / 2 - (center + x_off))
        extra_right = max(extra_right, (center + label_width / 2) - max_right)
    if extra_left > 0:
        for column in non_empty_columns:
            for box in column:
                box.x1 += extra_left
                box.x2 += extra_left
        max_right += extra_left

    return max_right + max(extra_right, 0.0)


def _draw_dimension_labels(
    img: PIL.Image,
    boxes_by_column: list[list[VolumetricBox]],
    diagram_height: float,
    font: ImageFont,
    font_color: str | tuple[int, ...],
) -> PIL.Image:
    """Draw each column's output shape(s) centered beneath it, in the reserved label row."""
    text_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw_text = ImageDraw.Draw(text_img)

    for column in boxes_by_column:
        if not column:
            continue
        label, center_x = _column_label_and_center(column)
        text_width = font.getbbox(label)[2]
        draw_text.text((center_x - text_width / 2, diagram_height + 2), label, font=font, fill=font_color)

    return Image.alpha_composite(img, text_img)


def layered_view(*args: object, **kwargs: object) -> PIL.Image:
    """Pre-1.0 alias for `flow_view`.

    Accepts the same arguments as `flow_view`, plus the removed `index_ignore` (silently
    dropped - it has no equivalent under the topology-based layout) and the renamed
    `one_dim_orientation` (mapped to `low_dim_orientation`). Positional calls using the old
    pre-1.0 argument order (which had `index_ignore` where `flow_view` now has `color_map`)
    will not be translated correctly - use keyword arguments to migrate safely.

    Deprecated:
        Since version 1.0.0. Use `flow_view` instead.
    """
    warnings.warn(
        "`layered_view` is deprecated and will be removed in a future release, use `flow_view` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    kwargs.pop("index_ignore", None)
    if "one_dim_orientation" in kwargs:
        kwargs["low_dim_orientation"] = kwargs.pop("one_dim_orientation")
    return flow_view(*args, **kwargs)  # type: ignore[arg-type]
