"""Tests for graph view."""

# Copyright (C) 2024 Willy Fitra Hendria
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from visualtorch import graph_view
from visualtorch.graph import _compute_skip_levels
from visualtorch.utils.layer_utils import model_to_adj_matrix
from visualtorch.utils.utils import Box, Ellipses


@pytest.fixture()
def dense_model() -> nn.Module:
    """A simple dense model creation."""

    class SimpleDense(nn.Module):
        """Simple Dense Model."""

        def __init__(self) -> None:
            super().__init__()
            self.h0 = nn.Linear(4, 8)
            self.h1 = nn.Linear(8, 8)
            self.h2 = nn.Linear(8, 4)
            self.out = nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Define the forward pass."""
            x = self.h0(x)
            x = self.h1(x)
            x = self.h2(x)
            return self.out(x)

    return SimpleDense()


@pytest.fixture()
def conv_model() -> nn.Module:
    """A simple conv model, exercising the Conv2d/ConvolutionBackward0 path."""
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, 1, 1),
        nn.Conv2d(8, 16, 3, 1, 1),
    )


@pytest.fixture()
def wide_dense_model() -> nn.Module:
    """A dense model with a hidden layer wider than the default ellipsize_after threshold."""

    class WideDense(nn.Module):
        """A dense model with more than 10 hidden units, to trigger ellipsis drawing."""

        def __init__(self) -> None:
            super().__init__()
            self.h0 = nn.Linear(4, 20)
            self.out = nn.Linear(20, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass."""
            return self.out(self.h0(x))

    return WideDense()


@pytest.fixture()
def residual_model() -> nn.Module:
    """A model with a skip connection around a hidden sub-block, to prove branching is captured."""

    class ResidualBlock(nn.Module):
        """fc1->fc2 with a skip connection from stem's output, merged into out."""

        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Linear(4, 4)
            self.fc1 = nn.Linear(4, 4)
            self.fc2 = nn.Linear(4, 4)
            self.out = nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass with a skip connection around fc1/fc2."""
            stem_out = self.stem(x)
            branch = self.fc2(self.fc1(stem_out))
            merged = branch + stem_out
            return self.out(merged)

    return ResidualBlock()


@pytest.fixture()
def batchnorm_model() -> nn.Module:
    """A model using BatchNorm2d, a layer type never supported by the old hardcoded TARGET_OPS."""
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, 1, 1),
        nn.BatchNorm2d(8),
        nn.ReLU(),
    )


def test_dense_model_graph_view_runs(dense_model: nn.Module) -> None:
    """Test graph view using dense model."""
    _ = graph_view(dense_model, input_shape=(1, 4))


def test_conv_model_graph_view_runs(conv_model: nn.Module) -> None:
    """graph_view should support Conv2d layers, not just Linear."""
    img = graph_view(conv_model, input_shape=(1, 3, 16, 16))
    assert img is not None


def test_graph_view_ellipsizes_wide_layers(wide_dense_model: nn.Module) -> None:
    """A hidden layer wider than ellipsize_after should draw an ellipsis, not crash."""
    img = graph_view(wide_dense_model, input_shape=(1, 4), ellipsize_after=10)
    assert img is not None


def test_graph_view_show_neurons_false(wide_dense_model: nn.Module) -> None:
    """show_neurons=False should render one node per layer instead of per neuron."""
    img = graph_view(wide_dense_model, input_shape=(1, 4), show_neurons=False)
    assert img is not None


def test_graph_view_writes_to_file(dense_model: nn.Module, tmp_path: Path) -> None:
    """to_file should save a readable image to disk."""
    out_file = tmp_path / "graph.png"
    graph_view(dense_model, input_shape=(1, 4), to_file=str(out_file))

    assert out_file.exists()
    with Image.open(out_file) as saved_img:
        assert saved_img.size[0] > 0
        assert saved_img.size[1] > 0


def test_graph_view_residual_model_runs(residual_model: nn.Module) -> None:
    """graph_view should not crash on a model with a skip connection."""
    img = graph_view(residual_model, input_shape=(1, 4))
    assert img is not None


def test_graph_view_batchnorm_model_runs(batchnorm_model: nn.Module) -> None:
    """graph_view should not crash on a model using an untracked layer type (BatchNorm2d)."""
    img = graph_view(batchnorm_model, input_shape=(2, 3, 8, 8))
    assert img is not None


def test_model_to_adj_matrix_edge_count_dense_model(dense_model: nn.Module) -> None:
    """model_to_adj_matrix should produce exactly the expected edges for a known simple chain."""
    _, adj_matrix, model_layers, _ = model_to_adj_matrix(dense_model, input_shape=(1, 4))

    # 4 chained Linear layers => exactly 3 edges, one layer per column.
    assert int(adj_matrix.sum()) == 3
    assert len(model_layers) == 4
    for column in model_layers:
        assert len(column) == 1


def test_model_to_adj_matrix_captures_branching(residual_model: nn.Module) -> None:
    """The merge point of a skip connection should have 2 incoming edges, not 1."""
    id_to_index, adj_matrix, _, _ = model_to_adj_matrix(residual_model, input_shape=(1, 4))

    out_node_id = f"{id(residual_model.out)}#0"
    in_degree = adj_matrix[:, id_to_index[out_node_id]].sum()

    assert in_degree == 2, "expected the merge node to have 2 incoming edges from the skip connection"


def test_model_to_adj_matrix_supports_untracked_layer_type(batchnorm_model: nn.Module) -> None:
    """A BatchNorm2d instance should actually appear among the traced layers."""
    _, _, model_layers, _ = model_to_adj_matrix(batchnorm_model, input_shape=(2, 3, 8, 8))

    traced_types = {type(wrapper.module) for column in model_layers for wrapper in column}
    assert nn.BatchNorm2d in traced_types


def test_graph_view_supports_weight_sharing() -> None:
    """A leaf module invoked more than once in one forward pass should not crash."""

    class SharedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.fc(x)
            return self.fc(x)

    img = graph_view(SharedModel(), input_shape=(1, 4))
    assert img is not None


def test_graph_view_bare_leaf_module_as_model() -> None:
    """Passing a single leaf module (no children) directly as the model should not crash."""
    img = graph_view(nn.Linear(4, 2), input_shape=(1, 4))
    assert img is not None


def test_model_to_adj_matrix_wires_deeper_direct_input_consumer() -> None:
    """A node that reads the raw input directly, in addition to a hidden predecessor, still gets its input edge."""

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Linear(4, 4)
            self.b = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            a = self.a(x)
            return self.b(x + a)

    model = M()
    _, _, _, direct_input_node_ids = model_to_adj_matrix(model, input_shape=(1, 4))

    b_node_id = next(node_id for node_id in direct_input_node_ids if node_id.startswith(f"{id(model.b)}#"))
    assert b_node_id is not None


def test_model_to_adj_matrix_survives_subclass_escaping_op() -> None:
    """Lineage should survive a leaf module whose forward escapes the tensor subclass internally."""

    class NumpyRoundTrip(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            arr = x.detach().numpy()
            return torch.from_numpy(arr.copy())

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Linear(4, 4)
            self.roundtrip = NumpyRoundTrip()
            self.b = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.a(x)
            x = self.roundtrip(x)
            return self.b(x)

    model = M()
    _, adj_matrix, model_layers, _ = model_to_adj_matrix(model, input_shape=(1, 4))

    assert int(adj_matrix.sum()) == 2
    assert [type(wrapper.module).__name__ for column in model_layers for wrapper in column] == [
        "Linear",
        "NumpyRoundTrip",
        "Linear",
    ]


def test_graph_view_with_show_dimension(dense_model: nn.Module) -> None:
    """show_dimension=True should print each layer's shape without clipping or crashing."""
    img = graph_view(dense_model, input_shape=(1, 4), show_dimension=True)
    assert img is not None


def test_graph_view_show_dimension_with_ellipsized_layer(wide_dense_model: nn.Module) -> None:
    """A wide, ellipsized layer's label should still be placed correctly."""
    img = graph_view(wide_dense_model, input_shape=(1, 4), show_dimension=True, ellipsize_after=10)
    assert img is not None


def test_graph_view_show_dimension_with_branching(residual_model: nn.Module) -> None:
    """A column with a skip-connection merge should still get a correctly placed label."""
    img = graph_view(residual_model, input_shape=(1, 4), show_dimension=True)
    assert img is not None


def test_compute_skip_levels_ignores_span_le_1_edges() -> None:
    """An edge with column span <= 1 should never be assigned a detour level."""
    id_to_num_mapping = {"a": 0, "b": 1}
    id_to_column_index = {"a": 0, "b": 1}
    id_to_node_list_map = {"a": [Box()], "b": [Box()]}
    adj_matrix = np.zeros((2, 2))
    adj_matrix[0, 1] = 1

    edge_to_level, num_levels = _compute_skip_levels(
        adj_matrix,
        id_to_num_mapping,
        id_to_column_index,
        id_to_node_list_map,
    )

    assert edge_to_level == {}
    assert num_levels == 0


def test_compute_skip_levels_assigns_distinct_levels_to_overlapping_skips() -> None:
    """Two skip edges whose column spans genuinely overlap must get different levels."""
    ids = ["a", "b", "c", "d"]
    id_to_num_mapping = {node_id: idx for idx, node_id in enumerate(ids)}
    id_to_column_index = {"a": 0, "b": 3, "c": 2, "d": 5}
    id_to_node_list_map = {node_id: [Box()] for node_id in ids}
    adj_matrix = np.zeros((4, 4))
    adj_matrix[id_to_num_mapping["a"], id_to_num_mapping["b"]] = 1
    adj_matrix[id_to_num_mapping["c"], id_to_num_mapping["d"]] = 1

    edge_to_level, num_levels = _compute_skip_levels(
        adj_matrix,
        id_to_num_mapping,
        id_to_column_index,
        id_to_node_list_map,
    )

    assert num_levels == 2
    assert edge_to_level[("a", "b")] != edge_to_level[("c", "d")]


def test_compute_skip_levels_allows_touching_intervals_to_share_a_level() -> None:
    """Two skip edges that only touch at a shared column boundary can share a level."""
    ids = ["a", "b", "c", "d"]
    id_to_num_mapping = {node_id: idx for idx, node_id in enumerate(ids)}
    id_to_column_index = {"a": 0, "b": 3, "c": 3, "d": 5}
    id_to_node_list_map = {node_id: [Box()] for node_id in ids}
    adj_matrix = np.zeros((4, 4))
    adj_matrix[id_to_num_mapping["a"], id_to_num_mapping["b"]] = 1
    adj_matrix[id_to_num_mapping["c"], id_to_num_mapping["d"]] = 1

    edge_to_level, num_levels = _compute_skip_levels(
        adj_matrix,
        id_to_num_mapping,
        id_to_column_index,
        id_to_node_list_map,
    )

    assert num_levels == 1
    assert edge_to_level[("a", "b")] == edge_to_level[("c", "d")] == 0


def test_compute_skip_levels_ignores_all_ellipses_edge() -> None:
    """A skip edge whose only node pairs are Ellipses-gated shouldn't consume a level."""
    id_to_num_mapping = {"a": 0, "b": 1}
    id_to_column_index = {"a": 0, "b": 3}
    id_to_node_list_map = {"a": [Ellipses()], "b": [Ellipses()]}
    adj_matrix = np.zeros((2, 2))
    adj_matrix[0, 1] = 1

    edge_to_level, num_levels = _compute_skip_levels(
        adj_matrix,
        id_to_num_mapping,
        id_to_column_index,
        id_to_node_list_map,
    )

    assert edge_to_level == {}
    assert num_levels == 0


def test_graph_view_residual_model_routes_above_diagram(residual_model: nn.Module) -> None:
    """The skip connection should reserve extra vertical space rather than drawing invisibly."""

    class PlainChain(nn.Module):
        """The same 4 layers as residual_model, but without the skip connection."""

        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Linear(4, 4)
            self.fc1 = nn.Linear(4, 4)
            self.fc2 = nn.Linear(4, 4)
            self.out = nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass with no skip connection."""
            return self.out(self.fc2(self.fc1(self.stem(x))))

    img_with_skip = graph_view(residual_model, input_shape=(1, 4), show_neurons=False)
    img_without_skip = graph_view(PlainChain(), input_shape=(1, 4), show_neurons=False)

    assert img_with_skip.size[1] > img_without_skip.size[1]


def test_graph_view_deep_repeated_residual_blocks_stays_reasonably_sized() -> None:
    """Back-to-back, non-overlapping residual blocks should share one detour level, not stack per block."""

    class ResBlock(nn.Module):
        """A minimal residual block: two convs with a skip connection, ending in a ReLU.

        The trailing ReLU matters: it's a leaf-module call, so it resets the tensor's producer
        set to just itself. Without it, `identity = x` would keep carrying every earlier block's
        producer forward untouched (mathematically correct - the merged tensor really is a sum
        of both - but it makes each block's skip nest inside the previous one instead of being
        independent, which is a different scenario than this test is after).
        """

        def __init__(self, channels: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass with a skip connection around conv1/conv2."""
            identity = x
            out = self.conv2(self.conv1(x))
            return self.relu(out + identity)

    class DeepModel(nn.Module):
        """A stack of N sequential, non-overlapping residual blocks."""

        def __init__(self, channels: int, n_blocks: int) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([ResBlock(channels) for _ in range(n_blocks)])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through every block in sequence."""
            for block in self.blocks:
                x = block(x)
            return x

    img_2_blocks = graph_view(DeepModel(4, 2), input_shape=(1, 4, 8, 8), show_neurons=False)
    img_6_blocks = graph_view(DeepModel(4, 6), input_shape=(1, 4, 8, 8), show_neurons=False)

    # Height should stay the same regardless of block count (every block's skip shares one
    # detour level); only width should grow as more layers are added.
    assert img_2_blocks.size[1] == img_6_blocks.size[1]


def test_graph_view_residual_model_show_neurons_true_runs(residual_model: nn.Module) -> None:
    """A skip edge under show_neurons=True should collapse to one connector, not crash on a dense mesh."""
    img = graph_view(residual_model, input_shape=(1, 4), show_neurons=True)
    assert img is not None
