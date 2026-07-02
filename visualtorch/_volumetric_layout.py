"""Shared volumetric column layout, used by every "stacked box" rendering style.

Generalizes the single-cursor sequential layout `layered_view`/`lenet_view` used before the v2
backend unification to the same column concept `graph_view` already uses: layers grouped by
depth, with more than one layer in a column when the model branches. Multiple boxes in a column
stack top-to-bottom; the caller decides each box's own size/extrusion (`make_box`) and the safe
vertical gap needed so one box's 3D extrusion doesn't visually occlude the box stacked below it
(`gap_for`) - both differ between `Box` (a single extruded cuboid) and `StackedBox` (a stack of
offset flat rectangles).
"""

# Copyright (C) 2024 Willy Fitra Hendria
# SPDX-License-Identifier: MIT

from collections.abc import Callable
from dataclasses import dataclass

from .utils.traced_layer import TracedLayer
from .utils.utils import Box


@dataclass
class ColumnLayout:
    """The result of laying out one box per traced layer, column by column."""

    boxes_by_column: list[list[Box]]
    id_to_box: dict[str, Box]
    column_heights: list[float]
    diagram_height: float
    max_right: float
    img_width: float
    x_off: float


def layout_columns(
    columns: list[list[TracedLayer]],
    make_box: Callable[[TracedLayer], Box],
    gap_for: Callable[[Box], float],
    spacing: int,
    padding: int,
) -> ColumnLayout:
    """Lay out one box per traced layer, advancing through columns left to right.

    Boxes within the same column (parallel branches at the same depth) stack top to bottom,
    each starting at `gap_for(previous_box)` below the one above it.

    Args:
        columns: Layers grouped by depth, in traversal order (e.g. `Architecture.columns`).
        make_box: Given a layer, return a box already sized (`x2 - x1` is its screen width,
            `y2 - y1` its screen height, `de` its 3D extrusion depth if any) but unpositioned
            (`x1 == y1 == 0`).
        gap_for: Given a positioned box, the safe vertical gap to leave before the next box
            stacked below it in the same column.
        spacing: Horizontal gap between columns.
        padding: Distance in pixels before the first column.

    Returns:
        ColumnLayout: The positioned boxes plus the bookkeeping every rendering frontend needs
            to finish drawing (centering, canvas size, connector endpoints).
    """
    boxes_by_column: list[list[Box]] = []
    id_to_box: dict[str, Box] = {}
    column_heights: list[float] = []
    current_x = float(padding)
    max_right = 0.0
    x_off: float | None = None

    for layer_list in columns:
        boxes = [make_box(layer) for layer in layer_list]
        column_width = 0.0
        y_cursor = 0.0

        for layer, box in zip(layer_list, boxes, strict=True):
            de = getattr(box, "de", 0)
            width = box.x2 - box.x1
            height = box.y2 - box.y1

            if x_off is None:
                x_off = de / 2

            box.x1 = current_x - int(de / 2)
            box.x2 = box.x1 + width
            box.y1 = y_cursor + de
            box.y2 = box.y1 + height

            id_to_box[layer.node_id] = box
            column_width = max(column_width, width)
            max_right = max(max_right, box.x2 + de)
            y_cursor = box.y2 + gap_for(box)

        column_heights.append(y_cursor - gap_for(boxes[-1]) if boxes else 0.0)
        boxes_by_column.append(boxes)
        current_x += column_width + spacing

    resolved_x_off = x_off if x_off is not None else 0.0
    return ColumnLayout(
        boxes_by_column=boxes_by_column,
        id_to_box=id_to_box,
        column_heights=column_heights,
        diagram_height=max(column_heights, default=0.0),
        max_right=max_right,
        img_width=max_right + resolved_x_off + padding,
        x_off=resolved_x_off,
    )
