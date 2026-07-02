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
from visualtorch.lenet_style import lenet_view
from visualtorch.utils.utils import self_multiply


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
