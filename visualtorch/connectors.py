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
    get_bbox: Callable[[str], tuple[float, float, float, float] | None],
) -> tuple[dict[tuple[str, str], int], int]:
    """Assign a detour level to every edge whose column span is > 1 AND would actually collide.

    A span > 1 edge is a genuine skip connection needing a routed detour only if a straight line
    between its two connection points would visually intersect some *other* node's box in one of
    the intervening columns - e.g. a residual connection bypassing same-row layers. A span > 1
    edge whose intervening columns are all occupied by boxes in a *different* row (e.g. a sibling
    branch that's simply deeper than this one, with no shared row) draws a clear diagonal line
    with nothing behind it, and is treated exactly like an ordinary adjacent-column edge instead
    (never enters the returned `edge_to_level`, so callers see `level is None` and draw a plain
    line with no detour).

    Levels are assigned via greedy interval-graph-coloring over each qualifying edge's
    (start_col, end_col) span, so that skip connections whose spans overlap never share a level
    (and so never draw collinear with each other), while non-overlapping spans - e.g. back-to-back
    residual blocks - can safely share the same level. Edges that would draw nothing (per
    `edge_has_content`) don't consume a level.

    Args:
        edges (Iterable): The model's `(start_id, end_id)` edges, e.g. from `Architecture.edges()`.
        id_to_column (dict): Mapping from node IDs to their column index.
        edge_has_content (Callable): Given `(start_id, end_id)`, returns whether this edge would
            actually draw something visible - a frontend-specific check (e.g. `graph_view`'s
            per-neuron `Ellipses` gating), kept out of this module entirely.
        get_bbox (Callable): Given a node ID, returns its `(x1, y1, x2, y2)` pixel bounding box, or
            `None` if that ID has no drawable box (e.g. filtered out by `type_ignore`) - kept
            generic so this module stays unaware of any specific box/node class, matching
            `edge_has_content`'s design.

    Returns:
        tuple: A tuple containing:
            - edge_to_level (dict): Mapping from `(start_id, end_id)` to its detour level (0 =
                closest to the diagram). Only contains edges with span > 1 that draw something
                and would actually collide with an intervening box.
            - num_levels (int): Total distinct levels used (0 if there are no qualifying edges).
    """
    column_to_ids: dict[int, list[str]] = {}
    for node_id, col in id_to_column.items():
        column_to_ids.setdefault(col, []).append(node_id)

    intervals: list[tuple[int, int, tuple[str, str]]] = []
    for start_id, end_id in edges:
        start_col = id_to_column[start_id]
        end_col = id_to_column[end_id]
        if end_col - start_col <= 1:
            continue
        if not edge_has_content(start_id, end_id):
            continue
        if _skip_edge_collides(start_id, end_id, start_col, end_col, column_to_ids, get_bbox):
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


def _skip_edge_collides(
    start_id: str,
    end_id: str,
    start_col: int,
    end_col: int,
    column_to_ids: dict[int, list[str]],
    get_bbox: Callable[[str], tuple[float, float, float, float] | None],
) -> bool:
    """Whether a straight line from start_id's to end_id's connection point would visually
    intersect some *other* node's box in one of the strictly-intervening columns.

    If either endpoint has no bbox, geometry can't be evaluated - conservatively assume a
    collision (keep the always-detour behavior) rather than silently dropping the route.
    """  # noqa: D205
    start_bbox = get_bbox(start_id)
    end_bbox = get_bbox(end_id)
    if start_bbox is None or end_bbox is None:
        return True

    sx1, sy1, sx2, sy2 = start_bbox
    ex1, ey1, ex2, ey2 = end_bbox
    line_x1, line_y1 = sx2, (sy1 + sy2) / 2
    line_x2, line_y2 = ex1, (ey1 + ey2) / 2

    for col in range(start_col + 1, end_col):
        for other_id in column_to_ids.get(col, []):
            other_bbox = get_bbox(other_id)
            if other_bbox is None:
                # Can't evaluate this obstacle's geometry - conservatively assume it collides
                # rather than silently treating an unresolvable box as if it weren't there.
                return True
            if _segment_intersects_rect(line_x1, line_y1, line_x2, line_y2, *other_bbox):
                return True
    return False


def _segment_intersects_rect(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    rx1: float,
    ry1: float,
    rx2: float,
    ry2: float,
) -> bool:
    """Whether the segment (x1,y1)-(x2,y2) intersects the axis-aligned rect [rx1,ry1,rx2,ry2].

    Conservative: touching an edge or corner counts as intersecting (a false-positive detour is
    far less harmful than a straight line silently drawn through a box). Implemented as a
    Liang-Barsky clip: the segment is parameterized as P(t) = (x1,y1) + t*(dx,dy) for t in
    [0, 1], and each of the rect's 4 half-plane constraints tightens the surviving [t0, t1]
    range. If [t0, t1] is non-empty at the end, some point of the segment lies inside (or on the
    boundary of) the rect. This handles horizontal/vertical/degenerate (dx==0 and/or dy==0,
    including a zero-length point segment) segments uniformly, with no special-casing.
    """
    rx_lo, rx_hi = (rx1, rx2) if rx1 <= rx2 else (rx2, rx1)
    ry_lo, ry_hi = (ry1, ry2) if ry1 <= ry2 else (ry2, ry1)

    dx = x2 - x1
    dy = y2 - y1
    t0, t1 = 0.0, 1.0

    # p*t <= q for each of the 4 half-plane boundaries (left, right, top, bottom).
    for p, q in (
        (-dx, x1 - rx_lo),
        (dx, rx_hi - x1),
        (-dy, y1 - ry_lo),
        (dy, ry_hi - y1),
    ):
        if p == 0:
            # Segment is parallel to this pair of edges: if it's already outside on this axis,
            # no t can bring it back in - reject outright, regardless of the other axis.
            if q < 0:
                return False
            continue
        t = q / p
        if p < 0:
            if t > t1:
                return False
            t0 = max(t0, t)
        else:
            if t < t0:
                return False
            t1 = min(t1, t)

    return t0 <= t1


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
