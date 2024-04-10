"""LeNet Style Basic Sequential
=======================================

Visualization of basic custom model
"""  # noqa: D205

import matplotlib.pyplot as plt
import visualtorch
from torch import nn

# Example of a simple CNN model using nn.Sequential
model = nn.Sequential(
    nn.Conv2d(3, 8, kernel_size=3, padding=1),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(8, 16, kernel_size=3, padding=1),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.MaxPool2d(2, 2),
)

input_shape = (1, 3, 128, 128)

img = visualtorch.lenet_view(model, input_shape=input_shape)

plt.axis("off")
plt.tight_layout()
plt.imshow(img)
plt.show()
