"""Tests for the visualtorch.render() single entry point."""

# Copyright (C) 2024 VisualTorch Contributors
# SPDX-License-Identifier: MIT

from collections import defaultdict

import pytest
import torch
import visualtorch
from torch import nn
from visualtorch import render
from visualtorch.backend import extract_architecture
from visualtorch.utils import layer_utils
from visualtorch.utils.layer_utils import Input
from visualtorch.utils.utils import PALETTES


@pytest.fixture
def sequential_model() -> nn.Sequential:
    """A simple conv model, exercised across all three styles."""
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, 1, 1),
        nn.ReLU(),
    )


@pytest.fixture
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
        render(sequential_model, input_shape=(1, 3, 16, 16), style="graph", draw_volume=True)


@pytest.mark.parametrize("style", ["graph", "lenet"])
def test_render_forwards_legend_for_graph_and_lenet(sequential_model: nn.Sequential, style: str) -> None:
    """Graph and LeNet styles should accept and forward legend through render()."""
    without_legend = render(sequential_model, input_shape=(1, 3, 16, 16), style=style, legend=False)
    with_legend = render(sequential_model, input_shape=(1, 3, 16, 16), style=style, legend=True)

    assert with_legend.height > without_legend.height


@pytest.mark.parametrize("style", ["graph", "lenet"])
def test_render_forwards_legend_position_for_graph_and_lenet(
    sequential_model: nn.Sequential,
    style: str,
) -> None:
    """Graph and LeNet styles should accept and forward legend_position through render()."""
    top = render(
        sequential_model,
        input_shape=(1, 3, 16, 16),
        style=style,
        legend=True,
        legend_position="top-left",
    )
    bottom = render(
        sequential_model,
        input_shape=(1, 3, 16, 16),
        style=style,
        legend=True,
        legend_position="bottom-left",
    )

    assert top.size == bottom.size
    assert top.tobytes() != bottom.tobytes()


def test_render_forwards_flow_legend_position(sequential_model: nn.Sequential) -> None:
    """Flow style should accept and forward legend_position through the render entry point."""
    top = render(
        sequential_model,
        input_shape=(1, 3, 16, 16),
        style="flow",
        legend=True,
        legend_position="top-left",
    )
    bottom = render(
        sequential_model,
        input_shape=(1, 3, 16, 16),
        style="flow",
        legend=True,
        legend_position="bottom-left",
    )

    assert top.size == bottom.size
    assert top.tobytes() != bottom.tobytes()


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


@pytest.mark.parametrize("style", ["graph", "flow", "lenet"])
def test_render_supports_integer_input_dtype_for_embedding_model(style: str) -> None:
    """A model starting with nn.Embedding needs an integer dummy tensor, not the default float
    one - input_dtype=torch.long should make every style render it successfully.
    """  # noqa: D205

    class TokenModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(1000, 16)
            self.fc = nn.Linear(16, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(self.embed(x))

    img = render(TokenModel(), input_shape=(1, 8), style=style, input_dtype=torch.long)
    assert img is not None


def test_render_multi_input_supports_per_tensor_input_dtype() -> None:
    """A multi-input model mixing a token (long) branch and a continuous (float) branch should
    render once each input tensor gets its own dtype via a per-position tuple.
    """  # noqa: D205

    class TokenAndFeatureModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(1000, 8)
            self.fc_tokens = nn.Linear(8, 8)
            self.fc_features = nn.Linear(10, 8)
            self.head = nn.Linear(16, 4)

        def forward(self, tokens: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
            token_repr = self.fc_tokens(self.embed(tokens)).mean(dim=1)
            feature_repr = self.fc_features(features)
            return self.head(torch.cat([token_repr, feature_repr], dim=-1))

    img = render(
        TokenAndFeatureModel(),
        input_shape=((1, 8), (1, 10)),
        style="graph",
        input_dtype=(torch.long, None),
    )
    assert img is not None


def test_render_rejects_invalid_input_dtype(sequential_model: nn.Sequential) -> None:
    """A non-dtype value for input_dtype should raise a clear ValueError, not an obscure one
    from deep inside torch.rand/torch.zeros.
    """  # noqa: D205
    with pytest.raises(ValueError, match="input_dtype must be"):
        render(sequential_model, input_shape=(1, 3, 16, 16), style="graph", input_dtype="not-a-dtype")


def test_render_rejects_input_dtype_tuple_arity_mismatch(siamese_model: nn.Module) -> None:
    """A per-position input_dtype tuple with the wrong length for a multi-input model should
    raise a clear ValueError.
    """  # noqa: D205
    with pytest.raises(ValueError, match="input_dtype must be"):
        render(
            siamese_model,
            input_shape=((1, 3, 16, 16), (1, 10)),
            style="graph",
            input_dtype=(torch.float32,),
        )


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


def test_input_dummy_layer_alias_still_works() -> None:
    """The deprecated top-level InputDummyLayer alias should still resolve to Input and warn."""
    with pytest.warns(DeprecationWarning, match="InputDummyLayer"):
        legacy_cls = visualtorch.InputDummyLayer
    assert legacy_cls is Input


def test_input_dummy_layer_alias_from_layer_utils_still_works() -> None:
    """The deprecated InputDummyLayer alias should also resolve via its original submodule path."""
    with pytest.warns(DeprecationWarning, match="InputDummyLayer"):
        legacy_cls = layer_utils.InputDummyLayer
    assert legacy_cls is Input


def test_render_one_dim_orientation_still_works(sequential_model: nn.Sequential) -> None:
    """The deprecated one_dim_orientation kwarg should still work through render() for both styles."""
    with pytest.warns(DeprecationWarning, match="one_dim_orientation"):
        flow_deprecated = render(sequential_model, input_shape=(1, 3, 16, 16), style="flow", one_dim_orientation="x")
    flow_current = render(sequential_model, input_shape=(1, 3, 16, 16), style="flow", low_dim_orientation="x")
    assert flow_deprecated.tobytes() == flow_current.tobytes()

    with pytest.warns(DeprecationWarning, match="one_dim_orientation"):
        lenet_deprecated = render(sequential_model, input_shape=(1, 3, 16, 16), style="lenet", one_dim_orientation="x")
    lenet_current = render(sequential_model, input_shape=(1, 3, 16, 16), style="lenet", low_dim_orientation="x")
    assert lenet_deprecated.tobytes() == lenet_current.tobytes()
