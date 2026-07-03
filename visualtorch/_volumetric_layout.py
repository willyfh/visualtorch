"""Shared volumetric column layout, used by every "stacked box" rendering style.

Generalizes the single-cursor sequential layout `flow_view`/`lenet_view` used before the v2
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
from .utils.utils import Box, StackedBox

VolumetricBox = Box | StackedBox


@dataclass
class ColumnLayout:
    """The result of laying out one box per traced layer, column by column."""

    boxes_by_column: list[list[VolumetricBox]]
    id_to_box: dict[str, VolumetricBox]
    column_heights: list[float]
    diagram_height: float
    max_right: float
    img_width: float
    x_off: float

    def bbox_for(self, node_id: str) -> tuple[float, float, float, float] | None:
        """A node's `(x1, y1, x2, y2)` bounding box, or None if it has no box (e.g. type_ignore'd)."""
        box = self.id_to_box.get(node_id)
        if box is None:
            return None
        return box.x1, box.y1, box.x2, box.y2


def _default_top_offset(box: VolumetricBox) -> float:
    return getattr(box, "de", 0)


def _default_x_shift(box: VolumetricBox) -> float:
    return -int(getattr(box, "de", 0) / 2)


def _default_x_extra(_box: VolumetricBox) -> float:
    return 0.0


def _default_right_extent(box: VolumetricBox) -> float:
    return getattr(box, "de", 0)


def layout_columns(
    columns: list[list[TracedLayer]],
    make_box: Callable[[TracedLayer], VolumetricBox],
    gap_for: Callable[[VolumetricBox], float],
    spacing: int,
    padding: int,
    top_offset_for: Callable[[VolumetricBox], float] | None = None,
    x_shift_for: Callable[[VolumetricBox], float] | None = None,
    x_extra_for: Callable[[VolumetricBox], float] | None = None,
    right_extent_for: Callable[[VolumetricBox], float] | None = None,
) -> ColumnLayout:
    """Lay out one box per traced layer, advancing through columns left to right.

    Boxes within the same column (parallel branches at the same depth) stack top to bottom,
    each starting at `gap_for(previous_box)` below the one above it.

    The four `*_for` callbacks all default to `Box`'s extruded-cuboid geometry (a small
    diagonal offset `de`, shifting only left/up) and only need overriding for a shape with a
    different footprint, e.g. `StackedBox`'s symmetric offset-copy spread.

    Args:
        columns: Layers grouped by depth, in traversal order (e.g. `Architecture.columns`).
        make_box: Given a layer, return a box already sized (`x2 - x1` is its screen width,
            `y2 - y1` its screen height, `de` its 3D extrusion depth if any) but unpositioned
            (`x1 == y1 == 0`).
        gap_for: Given a positioned box, the safe vertical gap to leave before the next box
            stacked below it in the same column.
        spacing: Horizontal gap between columns.
        padding: Distance in pixels before the first column.
        top_offset_for: Vertical offset from the column's y-cursor to this box's `y1`.
        x_shift_for: Signed horizontal shift from the column's x-cursor to this box's `x1`.
        x_extra_for: Extra horizontal room (beyond the widest box's width plus `spacing`) to
            reserve before the next column.
        right_extent_for: How far this box's rendering extends past its own `x2`, for the
            returned `max_right` bound.

    Returns:
        ColumnLayout: The positioned boxes plus the bookkeeping every rendering frontend needs
            to finish drawing (centering, canvas size, connector endpoints).
    """
    top_offset_for = top_offset_for or _default_top_offset
    x_shift_for = x_shift_for or _default_x_shift
    x_extra_for = x_extra_for or _default_x_extra
    right_extent_for = right_extent_for or _default_right_extent

    boxes_by_column: list[list[VolumetricBox]] = []
    id_to_box: dict[str, VolumetricBox] = {}
    column_heights: list[float] = []
    current_x = float(padding)
    max_right = 0.0
    x_off: float | None = None

    for layer_list in columns:
        boxes = [make_box(layer) for layer in layer_list]
        column_width = 0.0
        column_extra_x = 0.0
        y_cursor = 0.0

        for layer, box in zip(layer_list, boxes, strict=True):
            width = box.x2 - box.x1
            height = box.y2 - box.y1

            if x_off is None:
                x_off = max(0.0, -x_shift_for(box))

            box.x1 = current_x + x_shift_for(box)
            box.x2 = box.x1 + width
            box.y1 = y_cursor + top_offset_for(box)
            box.y2 = box.y1 + height

            id_to_box[layer.node_id] = box
            column_width = max(column_width, width)
            column_extra_x = max(column_extra_x, x_extra_for(box))
            max_right = max(max_right, box.x2 + right_extent_for(box))
            y_cursor = box.y2 + gap_for(box)

        column_heights.append(y_cursor - gap_for(boxes[-1]) if boxes else 0.0)
        boxes_by_column.append(boxes)
        current_x += column_width + spacing + column_extra_x

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
