"""Custom Node Size
=======================================

Visualization of custom node size
"""  # noqa: D205

import matplotlib.pyplot as plt
import torch
import visualtorch
from torch import nn


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


model = SimpleDense()

input_shape = (1, 4)

img = visualtorch.graph_view(model, input_shape, node_size=100)

plt.axis("off")
plt.tight_layout()
plt.imshow(img)
plt.show()
