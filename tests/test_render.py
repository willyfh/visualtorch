"""Tests for the visualtorch.render() single entry point."""

# Copyright (C) 2024 Willy Fitra Hendria
# SPDX-License-Identifier: MIT

import pytest
from torch import nn
from visualtorch import render


@pytest.fixture()
def sequential_model() -> nn.Sequential:
    """A simple conv model, exercised across all three styles."""
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, 1, 1),
        nn.ReLU(),
    )


@pytest.mark.parametrize("style", ["graph", "flow", "lenet"])
def test_render_runs_for_every_style(sequential_model: nn.Sequential, style: str) -> None:
    """Every registered style should render without error using its own defaults."""
    img = render(sequential_model, input_shape=(1, 3, 16, 16), style=style)
    assert img is not None


def test_render_defaults_to_graph_style(sequential_model: nn.Sequential) -> None:
    """Omitting style should render the graph style (the default)."""
    img = render(sequential_model, input_shape=(1, 3, 16, 16))
    assert img is not None


def test_render_forwards_style_specific_kwargs(sequential_model: nn.Sequential) -> None:
    """A style-specific kwarg (e.g. graph's node_size) should actually take effect."""
    small = render(sequential_model, input_shape=(1, 3, 16, 16), style="graph", node_size=20)
    large = render(sequential_model, input_shape=(1, 3, 16, 16), style="graph", node_size=200)

    assert small.size != large.size


def test_render_forwards_common_kwargs_across_styles(sequential_model: nn.Sequential) -> None:
    """A common kwarg (padding) should affect every style's output size."""
    for style in ("graph", "flow", "lenet"):
        tight = render(sequential_model, input_shape=(1, 3, 16, 16), style=style, padding=1)
        loose = render(sequential_model, input_shape=(1, 3, 16, 16), style=style, padding=100)
        assert tight.size != loose.size, f"padding had no effect for style={style!r}"


def test_render_rejects_unsupported_style(sequential_model: nn.Sequential) -> None:
    """An unrecognized style should raise a clear error, not silently fall back."""
    with pytest.raises(ValueError, match="Unsupported style"):
        render(sequential_model, input_shape=(1, 3, 16, 16), style="bogus")


def test_render_rejects_typo_d_style_kwarg(sequential_model: nn.Sequential) -> None:
    """A typo'd style-specific kwarg should raise TypeError, not be silently ignored."""
    with pytest.raises(TypeError):
        render(sequential_model, input_shape=(1, 3, 16, 16), style="graph", nod_size=20)


def test_render_rejects_typo_d_common_kwarg(sequential_model: nn.Sequential) -> None:
    """A typo'd common kwarg should raise TypeError, not be silently ignored."""
    with pytest.raises(TypeError):
        render(sequential_model, input_shape=(1, 3, 16, 16), style="graph", paddingg=5)


def test_render_rejects_kwarg_from_a_different_style(sequential_model: nn.Sequential) -> None:
    """A kwarg valid for one style but not another should raise TypeError for the wrong style."""
    with pytest.raises(TypeError):
        render(sequential_model, input_shape=(1, 3, 16, 16), style="graph", legend=True)
