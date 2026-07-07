"""Single public entry point for pytorch model visualization.

`graph_view`/`flow_view`/`lenet_view` (in `visualtorch.graph`/`.flow`/`.lenet_style`) render the
same `extract_architecture`-derived structure three different ways; this module consolidates
them behind one function, `render(model, input_shape, style=..., **kwargs)`, so `style` picks
the rendering style and every other parameter is style-appropriate keyword arguments. Kwargs are
validated by constructing a per-style dataclass from them - a typo'd kwarg raises `TypeError`
immediately, rather than being silently ignored.
"""

# Copyright (C) 2024 Willy Fitra Hendria
# SPDX-License-Identifier: MIT

from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any, Literal

from PIL import Image
from torch import nn

from .flow import flow_view
from .graph import graph_view
from .lenet_style import lenet_view
from .utils.utils import InputShape

Style = Literal["graph", "flow", "lenet"]


@dataclass
class CommonOptions:
    """Options accepted by every rendering style."""

    to_file: str | None = None
    color_map: dict[Any, Any] | None = None
    palette: str = "okabe_ito"
    background_fill: str | tuple[int, ...] = "white"
    padding: int = 10
    opacity: int = 255
    font: Any = None
    font_color: str | tuple[int, ...] = "black"
    level_gap: int | None = None


@dataclass
class GraphStyleOptions:
    """Options specific to `style="graph"` - a node/edge diagram, one node per neuron or layer."""

    node_size: int = 50
    layer_spacing: int = 250
    node_spacing: int = 10
    connector_fill: str | tuple[int, ...] = "gray"
    connector_width: int = 1
    ellipsize_after: int = 10
    show_neurons: bool = True
    show_dimension: bool = False
    show_input: bool = True


@dataclass
class FlowStyleOptions:
    """Options specific to `style="flow"` - stacked volumetric/2D boxes connected by funnels."""

    min_z: int = 10
    min_xy: int = 10
    max_z: int = 400
    max_xy: int = 2000
    scale_z: float = 0.1
    scale_xy: float = 1
    type_ignore: list[type] | None = None
    low_dim_orientation: str = "z"
    draw_volume: bool = True
    spacing: int = 10
    draw_funnel: bool = True
    shade_step: int = 10
    legend: bool = False
    show_dimension: bool = False
    show_input: bool = True
    connector_fill: str | tuple[int, ...] | None = None
    connector_width: int = 1
    one_dim_orientation: str | None = None  # deprecated, use low_dim_orientation


@dataclass
class LenetStyleOptions:
    """Options specific to `style="lenet"` - the classic LeNet stacked-plane look."""

    min_z: int = 1
    min_xy: int = 10
    max_xy: int = 2000
    scale_z: float = 1
    scale_xy: float = 1
    type_ignore: list[type] | None = None
    low_dim_orientation: str = "z"
    spacing: int = 10
    draw_funnel: bool = True
    shade_step: int = 10
    max_channels: int = 100
    offset_z: int = 10
    show_dimension: bool = True
    show_input: bool = True
    connector_fill: str | tuple[int, ...] | None = None
    connector_width: int = 1
    one_dim_orientation: str | None = None  # deprecated, use low_dim_orientation


def _render_graph(
    model: nn.Module,
    input_shape: InputShape,
    options: GraphStyleOptions,
    common: CommonOptions,
) -> Image.Image:
    return graph_view(
        model,
        input_shape,
        to_file=common.to_file,
        color_map=common.color_map,
        palette=common.palette,
        node_size=options.node_size,
        background_fill=common.background_fill,
        padding=common.padding,
        layer_spacing=options.layer_spacing,
        node_spacing=options.node_spacing,
        connector_fill=options.connector_fill,
        connector_width=options.connector_width,
        ellipsize_after=options.ellipsize_after,
        show_neurons=options.show_neurons,
        opacity=common.opacity,
        show_dimension=options.show_dimension,
        font=common.font,
        font_color=common.font_color,
        level_gap=common.level_gap,
        show_input=options.show_input,
    )


def _render_flow(
    model: nn.Module,
    input_shape: InputShape,
    options: FlowStyleOptions,
    common: CommonOptions,
) -> Image.Image:
    return flow_view(
        model,
        input_shape,
        to_file=common.to_file,
        min_z=options.min_z,
        min_xy=options.min_xy,
        max_z=options.max_z,
        max_xy=options.max_xy,
        scale_z=options.scale_z,
        scale_xy=options.scale_xy,
        type_ignore=options.type_ignore,
        color_map=common.color_map,
        palette=common.palette,
        low_dim_orientation=options.low_dim_orientation,
        background_fill=common.background_fill,
        draw_volume=options.draw_volume,
        padding=common.padding,
        spacing=options.spacing,
        draw_funnel=options.draw_funnel,
        shade_step=options.shade_step,
        legend=options.legend,
        font=common.font,
        font_color=common.font_color,
        opacity=common.opacity,
        show_dimension=options.show_dimension,
        level_gap=common.level_gap,
        show_input=options.show_input,
        connector_fill=options.connector_fill,
        connector_width=options.connector_width,
        one_dim_orientation=options.one_dim_orientation,
    )


def _render_lenet(
    model: nn.Module,
    input_shape: InputShape,
    options: LenetStyleOptions,
    common: CommonOptions,
) -> Image.Image:
    return lenet_view(
        model,
        input_shape,
        to_file=common.to_file,
        min_z=options.min_z,
        min_xy=options.min_xy,
        max_xy=options.max_xy,
        scale_z=options.scale_z,
        scale_xy=options.scale_xy,
        type_ignore=options.type_ignore,
        color_map=common.color_map,
        palette=common.palette,
        low_dim_orientation=options.low_dim_orientation,
        background_fill=common.background_fill,
        padding=common.padding,
        spacing=options.spacing,
        draw_funnel=options.draw_funnel,
        shade_step=options.shade_step,
        font=common.font,
        font_color=common.font_color,
        opacity=common.opacity,
        max_channels=options.max_channels,
        offset_z=options.offset_z,
        level_gap=common.level_gap,
        show_dimension=options.show_dimension,
        show_input=options.show_input,
        connector_fill=options.connector_fill,
        connector_width=options.connector_width,
        one_dim_orientation=options.one_dim_orientation,
    )


_STYLE_REGISTRY: dict[str, tuple[type, Callable[..., Image.Image]]] = {
    "graph": (GraphStyleOptions, _render_graph),
    "flow": (FlowStyleOptions, _render_flow),
    "lenet": (LenetStyleOptions, _render_lenet),
}

_COMMON_FIELDS = {f.name for f in fields(CommonOptions)}


def render(
    model: nn.Module,
    input_shape: InputShape,
    style: Style = "graph",
    **kwargs: Any,  # noqa: ANN401
) -> Image.Image:
    """Generate an architecture visualization for a given PyTorch model.

    Args:
        model (torch.nn.Module): A PyTorch model that will be visualized.
        input_shape (tuple): The shape of the input tensor, including batch dim. For a model
            whose forward() takes multiple separate input tensors, pass a tuple of per-tensor
            shapes instead, one per positional argument in order, e.g.
            ((1, 3, 224, 224), (1, 10)).
        style (str, optional): Which rendering style to use - `"graph"` (a node/edge diagram),
            `"flow"` (stacked volumetric/2D boxes connected by funnels), or `"lenet"` (the
            classic LeNet look).
        **kwargs: Style-specific and common options (see `GraphStyleOptions`/`FlowStyleOptions`/
            `LenetStyleOptions`/`CommonOptions` for the full list per style). Forwarded into the
            relevant dataclass constructor, so an unrecognized keyword raises `TypeError` rather
            than being silently ignored.

    Returns:
        Image.Image: Generated architecture image.
    """
    if style not in _STYLE_REGISTRY:
        supported = ", ".join(sorted(_STYLE_REGISTRY))
        error_msg = f"Unsupported style {style!r}. Supported styles: {supported}."
        raise ValueError(error_msg)

    options_cls, render_fn = _STYLE_REGISTRY[style]
    common_kwargs = {k: v for k, v in kwargs.items() if k in _COMMON_FIELDS}
    style_kwargs = {k: v for k, v in kwargs.items() if k not in _COMMON_FIELDS}

    common = CommonOptions(**common_kwargs)
    options = options_cls(**style_kwargs)
    return render_fn(model, input_shape, options, common)
