"""Tests for graph view."""

# Copyright (C) 2024 Willy Fitra Hendria
# SPDX-License-Identifier: MIT

import pytest
import torch.nn as nn

from visualtorch import graph_view


@pytest.fixture
def dense_model():
    class SimpleDense(nn.Module):
        def __init__(self):
            super(SimpleDense, self).__init__()
            self.h0 = nn.Linear(4, 8)
            self.h1 = nn.Linear(8, 8)
            self.h2 = nn.Linear(8, 4)
            self.out = nn.Linear(4, 2)

        def forward(self, x):
            x = self.h0(x)
            x = self.h1(x)
            x = self.h2(x)
            x = self.out(x)
            return x

    model = SimpleDense()
    return model


def test_dense_model_graph_view_runs(dense_model):
    try:
        _ = graph_view(dense_model, input_shape=(1, 4))
    except Exception as e:
        pytest.fail(f"graph_view raised an exception with a simple dense model: {e}")
