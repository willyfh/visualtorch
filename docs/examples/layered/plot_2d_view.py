"""2D View

=======================================

Visualization of 2D view
"""

import matplotlib.pyplot as plt
import visualtorch
from torch import nn

# Example of a simple CNN model using nn.Sequential
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(64 * 28 * 28, 256),  # Adjusted the input size for the Linear layer
    nn.ReLU(),
    nn.Linear(256, 10),  # Assuming 10 output classes
)

input_shape = (1, 3, 224, 224)
img = visualtorch.layered_view(model, input_shape=input_shape, draw_volume=False)

plt.axis("off")
plt.tight_layout()
plt.imshow(img)
plt.show()
