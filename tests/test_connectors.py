"""Tests for the frontend-agnostic connector-routing helpers."""

# Copyright (C) 2024 VisualTorch Contributors
# SPDX-License-Identifier: MIT

from visualtorch.connectors import _segment_intersects_rect, compute_skip_levels


def _no_bbox(_node_id: str) -> tuple[float, float, float, float] | None:
    return None


# ---- compute_skip_levels: span/content gating ----


def test_compute_skip_levels_ignores_span_le_1_edges() -> None:
    """An edge with column span <= 1 should never be assigned a detour level."""
    id_to_column = {"a": 0, "b": 1}

    edge_to_level, num_levels = compute_skip_levels([("a", "b")], id_to_column, lambda *_: True, _no_bbox)

    assert edge_to_level == {}
    assert num_levels == 0


def test_compute_skip_levels_assigns_distinct_levels_to_overlapping_skips() -> None:
    """Two skip edges whose column spans genuinely overlap must get different levels."""
    id_to_column = {"a": 0, "b": 3, "c": 2, "d": 5}
    edges = [("a", "b"), ("c", "d")]

    edge_to_level, num_levels = compute_skip_levels(edges, id_to_column, lambda *_: True, _no_bbox)

    assert num_levels == 2
    assert edge_to_level[("a", "b")] != edge_to_level[("c", "d")]


def test_compute_skip_levels_allows_touching_intervals_to_share_a_level() -> None:
    """Two skip edges that only touch at a shared column boundary can share a level."""
    id_to_column = {"a": 0, "b": 3, "c": 3, "d": 5}
    edges = [("a", "b"), ("c", "d")]

    edge_to_level, num_levels = compute_skip_levels(edges, id_to_column, lambda *_: True, _no_bbox)

    assert num_levels == 1
    assert edge_to_level[("a", "b")] == edge_to_level[("c", "d")] == 0


def test_compute_skip_levels_ignores_edges_with_no_content() -> None:
    """A skip edge that `edge_has_content` reports as empty shouldn't consume a level."""
    id_to_column = {"a": 0, "b": 3}

    edge_to_level, num_levels = compute_skip_levels([("a", "b")], id_to_column, lambda *_: False, _no_bbox)

    assert edge_to_level == {}
    assert num_levels == 0


# ---- compute_skip_levels: collision-awareness ----


def test_compute_skip_levels_keeps_level_when_intervening_box_collides() -> None:
    """A span>1 edge whose straight line genuinely crosses an intervening same-row box."""
    id_to_column = {"a": 0, "mid": 1, "b": 2}
    bboxes = {
        "a": (0.0, 0.0, 10.0, 10.0),
        "mid": (20.0, 0.0, 30.0, 10.0),  # same row (y=0..10) as a and b - directly in the path
        "b": (40.0, 0.0, 50.0, 10.0),
    }

    edge_to_level, num_levels = compute_skip_levels(
        [("a", "b")],
        id_to_column,
        lambda *_: True,
        lambda node_id: bboxes[node_id],
    )

    assert num_levels == 1
    assert edge_to_level[("a", "b")] == 0


def test_compute_skip_levels_drops_level_when_intervening_box_is_a_different_row() -> None:
    """A span>1 edge whose straight line passes clear of an intervening box in a different row."""
    id_to_column = {"a": 0, "mid": 1, "b": 2}
    bboxes = {
        "a": (0.0, 0.0, 10.0, 10.0),
        "mid": (20.0, 100.0, 30.0, 110.0),  # a different row entirely - well below the line
        "b": (40.0, 0.0, 50.0, 10.0),
    }

    edge_to_level, num_levels = compute_skip_levels(
        [("a", "b")],
        id_to_column,
        lambda *_: True,
        lambda node_id: bboxes[node_id],
    )

    assert edge_to_level == {}
    assert num_levels == 0


def test_compute_skip_levels_missing_bbox_conservatively_assumes_collision() -> None:
    """If an intervening id's bbox can't be resolved, assume a collision rather than guessing clear."""
    id_to_column = {"a": 0, "mid": 1, "b": 2}
    bboxes = {"a": (0.0, 0.0, 10.0, 10.0), "b": (40.0, 0.0, 50.0, 10.0)}

    edge_to_level, num_levels = compute_skip_levels(
        [("a", "b")],
        id_to_column,
        lambda *_: True,
        lambda node_id: bboxes.get(node_id),
    )

    assert num_levels == 1
    assert edge_to_level[("a", "b")] == 0


def test_compute_skip_levels_missing_endpoint_bbox_conservatively_assumes_collision() -> None:
    """If the edge's own start/end bbox can't be resolved, also assume a collision."""
    id_to_column = {"a": 0, "b": 2}

    edge_to_level, num_levels = compute_skip_levels([("a", "b")], id_to_column, lambda *_: True, _no_bbox)

    assert num_levels == 1
    assert edge_to_level[("a", "b")] == 0


# ---- _segment_intersects_rect ----


def test_segment_intersects_rect_segment_fully_inside() -> None:
    """A segment entirely within the rect counts as intersecting."""
    assert _segment_intersects_rect(2, 2, 8, 8, 0, 0, 10, 10) is True


def test_segment_intersects_rect_segment_crosses_through_middle() -> None:
    """A segment passing straight through the rect's interior intersects."""
    assert _segment_intersects_rect(-5, 5, 15, 5, 0, 0, 10, 10) is True


def test_segment_intersects_rect_segment_passes_above() -> None:
    """A segment entirely above the rect doesn't intersect."""
    assert _segment_intersects_rect(-5, -20, 15, -20, 0, 0, 10, 10) is False


def test_segment_intersects_rect_segment_passes_below() -> None:
    """A segment entirely below the rect doesn't intersect."""
    assert _segment_intersects_rect(-5, 50, 15, 50, 0, 0, 10, 10) is False


def test_segment_intersects_rect_touches_one_corner() -> None:
    """Touching exactly one corner counts as intersecting (conservative)."""
    assert _segment_intersects_rect(-5, -5, 0, 0, 0, 0, 10, 10) is True


def test_segment_intersects_rect_touches_one_edge() -> None:
    """Touching exactly one edge counts as intersecting (conservative)."""
    assert _segment_intersects_rect(-5, 5, 0, 5, 0, 0, 10, 10) is True


def test_segment_intersects_rect_horizontal_segment_clear() -> None:
    """A horizontal segment clear of the rect doesn't intersect."""
    assert _segment_intersects_rect(-5, -5, 15, -5, 0, 0, 10, 10) is False


def test_segment_intersects_rect_horizontal_segment_through_rect() -> None:
    """A horizontal segment passing through the rect intersects."""
    assert _segment_intersects_rect(-5, 5, 15, 5, 0, 0, 10, 10) is True


def test_segment_intersects_rect_vertical_segment_through_rect() -> None:
    """A vertical segment passing through the rect intersects."""
    assert _segment_intersects_rect(5, -5, 5, 15, 0, 0, 10, 10) is True


def test_segment_intersects_rect_vertical_segment_clear() -> None:
    """A vertical segment clear of the rect doesn't intersect."""
    assert _segment_intersects_rect(50, -5, 50, 15, 0, 0, 10, 10) is False


def test_segment_intersects_rect_degenerate_point_inside() -> None:
    """A zero-length segment (a point) inside the rect counts as intersecting."""
    assert _segment_intersects_rect(5, 5, 5, 5, 0, 0, 10, 10) is True


def test_segment_intersects_rect_degenerate_point_outside() -> None:
    """A zero-length segment (a point) outside the rect doesn't intersect."""
    assert _segment_intersects_rect(50, 50, 50, 50, 0, 0, 10, 10) is False
