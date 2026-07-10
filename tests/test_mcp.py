"""Focused tests for the migrated VisualTorch MCP integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from visualtorch_mcp.api_reference import normalize_style_name
from visualtorch_mcp.runner import (
    normalize_input_shape,
    render_model,
    resolve_output_path,
)
from visualtorch_mcp.worker import _coerce_options, _restore_tuples


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("graph", "graph"),
        ("FLOW", "flow"),
        ("layered", "flow"),
        ("layered-view", "flow"),
        (" lenet_style ", "lenet"),
        ("LENET_VIEW", "lenet"),
    ],
)
def test_style_aliases(alias: str, canonical: str) -> None:
    """Normalize supported style aliases to their canonical names."""
    assert normalize_style_name(alias) == canonical


def test_unknown_style_is_rejected() -> None:
    """Reject unsupported style names with an actionable error."""
    with pytest.raises(ValueError, match="Unsupported style"):
        normalize_style_name("not-a-style")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ([1, 3, 8, 8], (1, 3, 8, 8)),
        ("[[1, 3, 8, 8], [1, 5]]", ((1, 3, 8, 8), (1, 5))),
        (((1, 3), (1, 2)), ((1, 3), (1, 2))),
    ],
)
def test_input_shape_normalization(value: object, expected: object) -> None:
    """Normalize flat, JSON, and multi-input shapes into tuples."""
    assert normalize_input_shape(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        [],
        [1, [3, 8]],
        [[1, 3], []],
        [[1, 3], [1, 0]],
        [[1, 3], [1, True]],
        "not json",
    ],
)
def test_input_shape_validation(value: object) -> None:
    """Reject malformed shapes before starting a render subprocess."""
    with pytest.raises((ValueError, json.JSONDecodeError)):
        normalize_input_shape(value)


def test_output_path_resolution(tmp_path: Path) -> None:
    """Resolve explicit and generated paths beneath the requested output directory."""
    explicit = resolve_output_path("nested/diagram", str(tmp_path), "graph")
    assert explicit == (tmp_path / "nested" / "diagram.png").resolve()
    assert explicit.parent.is_dir()

    generated = resolve_output_path(None, str(tmp_path), "flow")
    assert generated.parent == tmp_path.resolve()
    assert generated.name.startswith("visualtorch_flow_")
    assert generated.suffix == ".png"


def test_worker_option_type_coercion_and_tuple_restore() -> None:
    """Coerce option type names and restore JSON lists to renderer tuples."""
    custom = type("Custom", (), {})
    namespace = {}
    coerced = _coerce_options(
        namespace,
        {"type_ignore": ["Linear", custom], "color_map": {custom: "red"}},
    )
    assert coerced["type_ignore"][0].__name__ == "Linear"
    assert coerced["type_ignore"][1] is custom
    assert coerced["color_map"] == {custom: "red"}
    assert _restore_tuples([[1, [2, 3]], 4]) == ((1, (2, 3)), 4)


def test_render_model_worker_smoke(tmp_path: Path) -> None:
    """Render a small model through the worker and report its generated PNG."""
    result = render_model(
        source="import torch\nmodel = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))",
        input_shape=(1, 1, 2, 2),
        style="graph",
        output_path="smoke.png",
        output_dir=str(tmp_path),
        options={"show_neurons": False},
        timeout_seconds=30,
    )

    output_path = Path(result["output_path"])
    assert output_path == (tmp_path / "smoke.png").resolve()
    assert output_path.is_file()
    assert result["style"] == "graph"
    assert result["bytes"] == output_path.stat().st_size
