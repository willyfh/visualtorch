"""Basic Custom
=======================================

Visualization of basic custom model
"""  # noqa: D205

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as func
import visualtorch
from torch import nn


# Example of a simple CNN model
class SimpleCNN(nn.Module):
    """Simple CNN Model."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the forward pass."""
        x = self.conv1(x)
        x = func.relu(x)
        x = func.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = func.relu(x)
        x = func.max_pool2d(x, 2, 2)
        x = self.conv3(x)
        x = func.relu(x)
        x = func.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = func.relu(x)
        return self.fc2(x)


# Create an instance of the SimpleCNN
model = SimpleCNN()

input_shape = (1, 3, 224, 224)

img = visualtorch.layered_view(model, input_shape=input_shape, legend=True)

plt.axis("off")
plt.tight_layout()
plt.imshow(img)
plt.show()
