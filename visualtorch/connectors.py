"""Connector-routing helpers shared by every rendering frontend.

Handles the general problem of an edge whose endpoints are more than one column apart (a
skip/residual connection): drawn as a plain straight line, it would be collinear with - and
hidden under - the ordinary adjacent-column edges beneath it. These helpers assign such edges a
detour "level" via greedy interval-graph-coloring over their column spans, and draw the routed
line, without any awareness of what a frontend's node objects actually look like.
"""

# Copyright (C) 2024 Willy Fitra Hendria
# SPDX-License-Identifier: MIT

from collections.abc import Callable, Iterable

import aggdraw


def compute_skip_levels(
    edges: Iterable[tuple[str, str]],
    id_to_column: dict[str, int],
    edge_has_content: Callable[[str, str], bool],
) -> tuple[dict[tuple[str, str], int], int]:
    """Assign a detour level to every edge whose column span is > 1 (a genuine skip).

    Levels are assigned via greedy interval-graph-coloring over each edge's (start_col, end_col)
    span, so that skip connections whose spans overlap never share a level (and so never draw
    collinear with each other), while non-overlapping spans - e.g. back-to-back residual blocks -
    can safely share the same level. Edges that would draw nothing (per `edge_has_content`) don't
    consume a level.

    Args:
        edges (Iterable): The model's `(start_id, end_id)` edges, e.g. from `Architecture.edges()`.
        id_to_column (dict): Mapping from node IDs to their column index.
        edge_has_content (Callable): Given `(start_id, end_id)`, returns whether this edge would
            actually draw something visible - a frontend-specific check (e.g. `graph_view`'s
            per-neuron `Ellipses` gating), kept out of this module entirely.

    Returns:
        tuple: A tuple containing:
            - edge_to_level (dict): Mapping from `(start_id, end_id)` to its detour level (0 =
                closest to the diagram). Only contains edges with span > 1 that draw something.
            - num_levels (int): Total distinct levels used (0 if there are no qualifying edges).
    """
    intervals: list[tuple[int, int, tuple[str, str]]] = []
    for start_id, end_id in edges:
        start_col = id_to_column[start_id]
        end_col = id_to_column[end_id]
        if end_col - start_col <= 1:
            continue
        if edge_has_content(start_id, end_id):
            intervals.append((start_col, end_col, (start_id, end_id)))

    intervals.sort(key=lambda interval: interval[0])

    levels: list[list[tuple[int, int]]] = []
    edge_to_level: dict[tuple[str, str], int] = {}
    for start_col, end_col, edge_key in intervals:
        for level_idx, occupied in enumerate(levels):
            if not any(start_col < o_end and o_start < end_col for o_start, o_end in occupied):
                occupied.append((start_col, end_col))
                edge_to_level[edge_key] = level_idx
                break
        else:
            levels.append([(start_col, end_col)])
            edge_to_level[edge_key] = len(levels) - 1

    return edge_to_level, len(levels)


def draw_connector(
    draw: aggdraw.Draw,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str | tuple[int, ...],
    width: int,
    detour_y: float | None = None,
) -> None:
    """Draw the line connector between two points.

    A plain straight line, unless `detour_y` is given - a skip connection whose column span
    would otherwise draw collinear with (and hidden under) the ordinary adjacent-column edges
    beneath it, so it's routed with a right-angle detour instead.
    """
    pen = aggdraw.Pen(color, width)
    if detour_y is None:
        draw.line([x1, y1, x2, y2], pen)
        return
    draw.line([x1, y1, x1, detour_y, x2, detour_y, x2, y2], pen)
