"""Regression tests for shape-handling crashes reported in open GitHub issues.

See https://github.com/willyfh/visualtorch/issues/63, /68, /69, /84, /85.
"""

# Copyright (C) 2024 Willy Fitra Hendria
# SPDX-License-Identifier: MIT

import pytest
import torch
from torch import nn
from visualtorch.backend import extract_architecture
from visualtorch.flow import flow_view
from visualtorch.graph import graph_view
from visualtorch.lenet_style import lenet_view
from visualtorch.utils.utils import format_shape_label, self_multiply


@pytest.fixture()
def multi_output_container_model() -> nn.Module:
    """A model whose inner block is a container (not Sequential/ModuleList) that returns multiple tensors.

    This mirrors timm's ``FeatureListNet`` used in issue #69: a container module that isn't
    ``nn.Sequential``/``nn.ModuleList`` and returns multiple differently-shaped tensors.
    """

    class FeaturePyramidBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stage1 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
            self.stage2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
            self.stage3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)

        def forward(self, x: torch.Tensor) -> list:
            f1 = self.stage1(x)
            f2 = self.stage2(f1)
            f3 = self.stage3(f2)
            return [f1, f2, f3]

    class MultiScaleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = FeaturePyramidBlock()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.features(x)[-1]

    return MultiScaleNet()


def test_flow_view_multi_output_container(multi_output_container_model: nn.Module) -> None:
    """flow_view should not crash on container modules that output multiple tensors."""
    img = flow_view(multi_output_container_model, input_shape=(1, 3, 32, 32))
    assert img is not None


def test_lenet_view_multi_output_container(multi_output_container_model: nn.Module) -> None:
    """lenet_view should not crash on container modules that output multiple tensors."""
    img = lenet_view(multi_output_container_model, input_shape=(1, 3, 32, 32))
    assert img is not None


def test_self_multiply_handles_nested_shape() -> None:
    """self_multiply should always reduce to a scalar, even if an element is itself a shape."""
    nested_shape = (1, torch.Size([4, 8]))
    result = self_multiply(nested_shape)
    assert isinstance(result, int)


def test_extract_architecture_records_multihead_attention_as_a_node() -> None:
    """nn.MultiheadAttention has a child (out_proj) but computes attention via a fused
    functional call on raw parameter tensors, never actually calling `self.out_proj(...)`.
    Without a fallback, neither it nor its child would ever become a traced node, silently
    dropping the attention computation from every render style. See issue #84.
    """  # noqa: D205

    class TinyTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(16, 32)
            encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(self.embed(x))

    architecture = extract_architecture(TinyTransformer(), (1, 10, 16))
    module_types = {type(layer.module) for column in architecture.columns for layer in column}

    assert nn.MultiheadAttention in module_types


@pytest.mark.parametrize("recurrent_cls", [nn.LSTM, nn.GRU, nn.RNN])
def test_flow_view_recurrent_sequence_length_does_not_inflate_diagram_height(recurrent_cls: type) -> None:
    """An RNN's output shape (seq_len, hidden_size) has no channel axis. Before the fix,
    seq_len was misread as a channel/extrusion count, drawing that many stacked volume slices -
    illegible for any realistic sequence length. Diagram height must stay independent of
    seq_len; only width should reflect it. See issue #85. Covers LSTM/GRU/RNN alike since all
    three share the exact same tracer/box-sizing path.
    """  # noqa: D205

    class RecurrentModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rnn = recurrent_cls(input_size=10, hidden_size=20, batch_first=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.rnn(x)
            return out

    model = RecurrentModel()
    short_img = flow_view(model, input_shape=(1, 5, 10))
    long_img = flow_view(model, input_shape=(1, 200, 10))

    assert long_img.height == short_img.height


@pytest.fixture()
def lstm_using_hidden_state_model() -> nn.Module:
    """A model that consumes `nn.LSTM`'s hidden state (h_n), not its sequence output.

    `nn.LSTM.forward()` returns `(output, (h_n, c_n))` - three tensors, not one. Before the fix,
    only `output`'s shape was ever recorded, even when (as here) the model actually uses `h_n`
    instead - silently dropping the shape of the tensor that matters, with nothing shown to
    indicate more than one tensor even exists.
    """

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size=10, hidden_size=20, batch_first=True)
            self.fc = nn.Linear(20, 5)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _output, (h_n, _c_n) = self.lstm(x)
            return self.fc(h_n.squeeze(0))

    return Model()


def test_extract_architecture_records_every_output_tensor_shape(lstm_using_hidden_state_model: nn.Module) -> None:
    """A multi-output leaf module's TracedLayer should record all three of its output shapes."""
    architecture = extract_architecture(lstm_using_hidden_state_model, (1, 7, 10))
    lstm_layer = next(layer for column in architecture.columns for layer in column if isinstance(layer.module, nn.LSTM))

    assert lstm_layer.output_shape == (1, 7, 20)
    assert lstm_layer.extra_output_shapes == ((1, 1, 20), (1, 1, 20))


def test_format_shape_label_appends_extra_shapes() -> None:
    """format_shape_label should append every extra shape, and omit the `+` entirely when there are none."""
    assert format_shape_label((1, 7, 20), ()) == "(1, 7, 20)"
    assert format_shape_label((1, 7, 20), ((1, 1, 20), (1, 1, 20))) == "(1, 7, 20) + (1, 1, 20) + (1, 1, 20)"


@pytest.mark.parametrize("view", [graph_view, flow_view, lenet_view])
def test_show_dimension_includes_every_output_shape(lstm_using_hidden_state_model: nn.Module, view: object) -> None:
    """show_dimension=True shouldn't crash, and should still work, for a multi-output leaf module."""
    img = view(lstm_using_hidden_state_model, input_shape=(1, 7, 10), show_dimension=True)  # type: ignore[operator]
    assert img is not None


@pytest.fixture()
def embedding_model() -> nn.Module:
    """A model starting with nn.Embedding - requires an integer/long index tensor.

    The tracer's dummy input was always a uniformly random float tensor, which nn.Embedding (and
    any other integer/bool-input layer) rejects outright, crashing before any tracing could
    happen at all.
    """

    class TokenModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(1000, 32)
            self.fc = nn.Linear(32, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(self.embed(x))

    return TokenModel()


def test_extract_architecture_rejects_embedding_model_without_integer_input_dtype(embedding_model: nn.Module) -> None:
    """Without input_dtype, the dummy tensor is float - nn.Embedding rejects it outright.

    Pinning this failure down explicitly (rather than just testing the fix in isolation) makes
    sure the fix is actually doing something: if this ever stopped raising, the `input_dtype`
    tests below would no longer be verifying a real fix.
    """
    with pytest.raises(RuntimeError, match="scalar type"):
        extract_architecture(embedding_model, (1, 16))


def test_extract_architecture_supports_embedding_model_via_integer_input_dtype(embedding_model: nn.Module) -> None:
    """input_dtype=torch.long should let the tracer build a valid dummy index tensor, and the
    Embedding layer should be traced like any other layer, with its real output shape recorded.
    """  # noqa: D205
    architecture = extract_architecture(embedding_model, (1, 16), input_dtype=torch.long)
    embedding_layer = next(
        layer for column in architecture.columns for layer in column if isinstance(layer.module, nn.Embedding)
    )

    assert embedding_layer.output_shape == (1, 16, 32)


@pytest.mark.parametrize("view", [graph_view, flow_view, lenet_view])
def test_view_renders_embedding_model_with_integer_input_dtype(embedding_model: nn.Module, view: object) -> None:
    """Every style should render a token-embedding model end to end once given input_dtype."""
    img = view(embedding_model, input_shape=(1, 16), input_dtype=torch.long)  # type: ignore[operator]
    assert img is not None
