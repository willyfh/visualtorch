"""Tests for the visualtorch.render() single entry point."""

# Copyright (C) 2024 Willy Fitra Hendria
# SPDX-License-Identifier: MIT

from collections import defaultdict

import pytest
import torch
from torch import nn
from visualtorch import render
from visualtorch.backend import extract_architecture
from visualtorch.utils.layer_utils import Input
from visualtorch.utils.utils import PALETTES


@pytest.fixture()
def sequential_model() -> nn.Sequential:
    """A simple conv model, exercised across all three styles."""
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, 1, 1),
        nn.ReLU(),
    )


@pytest.fixture()
def siamese_model() -> nn.Module:
    """A small two-branch model: an image branch and a tabular-vector branch, merged by concat."""

    class SiameseNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.image_branch = nn.Sequential(
                nn.Conv2d(3, 8, 3, 1, 1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            self.vector_branch = nn.Sequential(
                nn.Linear(10, 8),
                nn.ReLU(),
            )
            self.head = nn.Linear(16, 4)

        def forward(self, image: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
            merged = torch.cat([self.image_branch(image), self.vector_branch(vector)], dim=1)
            return self.head(merged)

    return SiameseNet()


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


def test_render_rejects_malformed_mixed_input_shape_with_clear_error(sequential_model: nn.Sequential) -> None:
    """A shape tuple mixing raw ints and nested shape-tuples at the top level is ambiguous and
    should raise a clear ValueError rather than being silently misinterpreted.
    """  # noqa: D205
    malformed_shape = (1, (3, 4), 224, 224)

    with pytest.raises(ValueError, match="input_shape must be"):
        render(sequential_model, input_shape=malformed_shape, style="graph")


@pytest.mark.parametrize("style", ["graph", "flow", "lenet"])
def test_render_runs_for_multi_input_model(siamese_model: nn.Module, style: str) -> None:
    """A model with two separate forward() input tensors should render in every style."""
    input_shape = ((1, 3, 16, 16), (1, 10))
    img = render(siamese_model, input_shape=input_shape, style=style)
    assert img is not None


def test_render_multi_input_reflects_distinct_input_shapes(siamese_model: nn.Module) -> None:
    """Each input's own shape should reach the diagram (via extract_architecture), not just one."""
    architecture = extract_architecture(siamese_model, ((1, 3, 16, 16), (1, 10)))
    input_shapes_seen = {layer.output_shape for layer in architecture.columns[0]}
    assert input_shapes_seen == {(1, 3, 16, 16), (1, 10)}


def test_render_rejects_multi_input_shape_arity_mismatch(siamese_model: nn.Module) -> None:
    """Passing the wrong number of input shapes for a multi-input forward() should fail clearly.

    This is a plain Python TypeError from the mismatched forward() call, not a visualtorch
    ValueError - documenting that boundary explicitly here.
    """
    with pytest.raises(TypeError):
        render(siamese_model, input_shape=((1, 3, 16, 16),), style="graph")


def test_render_handles_unused_input_tensor() -> None:
    """An input tensor declared in input_shape but never consumed in forward() should still
    render (as a disconnected box), not crash.
    """  # noqa: D205

    class PartiallyUnusedNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(10, 4)

        def forward(self, used: torch.Tensor, unused: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
            return self.linear(used)

    img = render(PartiallyUnusedNet(), input_shape=((1, 10), (1, 5)), style="graph")
    assert img is not None


@pytest.mark.parametrize("palette", sorted(PALETTES))
def test_render_runs_for_every_named_palette(sequential_model: nn.Sequential, palette: str) -> None:
    """Every named palette should render without error - catches any malformed hex color."""
    img = render(sequential_model, input_shape=(1, 3, 16, 16), style="graph", palette=palette)
    assert img is not None


def test_render_rejects_unsupported_palette(sequential_model: nn.Sequential) -> None:
    """An unrecognized palette name should raise a clear error, not silently fall back."""
    with pytest.raises(ValueError, match="Unsupported palette"):
        render(sequential_model, input_shape=(1, 3, 16, 16), style="graph", palette="bogus")


def test_render_palette_changes_fallback_colors(sequential_model: nn.Sequential) -> None:
    """A different palette should actually change the colors of unmapped layer types."""
    default = render(sequential_model, input_shape=(1, 3, 16, 16), style="graph")
    dracula = render(sequential_model, input_shape=(1, 3, 16, 16), style="graph", palette="dracula")

    assert default.tobytes() != dracula.tobytes()


def test_render_color_map_overrides_palette(sequential_model: nn.Sequential) -> None:
    """An explicit color_map entry should still win over the palette fallback."""
    color_map: dict = defaultdict(dict)
    color_map[Input]["fill"] = "#abcdef"
    color_map[nn.Conv2d]["fill"] = "#123456"
    color_map[nn.ReLU]["fill"] = "#654321"

    okabe_ito = render(sequential_model, input_shape=(1, 3, 16, 16), style="graph", color_map=color_map)
    dracula = render(
        sequential_model,
        input_shape=(1, 3, 16, 16),
        style="graph",
        color_map=color_map,
        palette="dracula",
    )

    assert okabe_ito.tobytes() == dracula.tobytes()
