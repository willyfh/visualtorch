# ⭐ VisualTorch ⭐

[![python](https://img.shields.io/badge/python-3.10%2B-blue)]() [![pytorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)]() [![Downloads](https://static.pepy.tech/personalized-badge/visualtorch?period=total&units=international_system&left_color=grey&right_color=green&left_text=PyPI%20Downloads)](https://pepy.tech/project/visualtorch) [![Run Tests](https://github.com/willyfh/visualtorch/actions/workflows/pytest.yml/badge.svg)](https://github.com/willyfh/visualtorch/actions/workflows/pytest.yml)

**VisualTorch** aims to help visualize Torch-based neural network architectures. Currently, this package supports generating layered-style and graph-style architectures for PyTorch Sequential and Custom models. This package is adapted from [visualkeras](https://github.com/paulgavrikov/visualkeras), [pytorchviz](https://github.com/szagoruyko/pytorchviz), and [pytorch-summary](https://github.com/sksq96/pytorch-summary).

**v0.2**: Added support for custom models and implemented graph view functionality.

**v0.1.1**: Added support for the layered architecture of Torch Sequential.

## Installation

### Install from PyPI (Latest release)

```bash
pip install visualtorch
```

### Install from source

```bash
pip install git+https://github.com/willyfh/visualtorch
```

## Usage

### Sequential

```python
import visualtorch
import torch.nn as nn

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
    nn.Linear(256, 10)  # Assuming 10 output classes
)

input_shape = (1, 3, 224, 224)

visualtorch.layered_view(model, input_shape=input_shape, legend=True).show() # display using your system viewer
```

![simple-cnn-sequential](https://github.com/willyfh/visualtorch/assets/5786636/9b646fac-c336-4253-ac01-8f3e6b2fcc0b)

### Custom Model

In a custom model, only the components defined within the model's **init** method are visualized. The operations that are defined exclusively within the forward function are not visualized.

```python
import torch.nn as nn
import torch.nn.functional as F
import visualtorch

# Example of a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

# Create an instance of the SimpleCNN
model = SimpleCNN()

input_shape = (1, 3, 224, 224)

visualtorch.layered_view(model, input_shape=input_shape, legend=True).show() # display using your system viewer
```

![simple-cnn-custom](https://github.com/willyfh/visualtorch/assets/5786636/f22298b4-f341-4a0d-b85b-11f01e207ad8)

### Graph View

```python
import torch
import torch.nn as nn
import visualtorch

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

input_shape = (1, 4)

visualtorch.graph_view(model, input_shape).show()
```

![graph-view](https://github.com/willyfh/visualtorch/assets/5786636/a65b4208-72da-497b-b6c9-aafc82b67b58)

### Save the Image

```python
visualtorch.layered_view(model, input_shape=input_shape, legend=True, to_file='output.png')
```

### 2D View

```python
visualtorch.layered_view(model, input_shape=input_shape, draw_volume=False)
```

![2d-view](https://github.com/willyfh/visualtorch/assets/5786636/5b16c252-f589-4b3f-8ea4-1bc188e6c124)

### Custom Color

Use 'fill' to change the color of the layer, and use 'outline' to change the color of the lines.

```python
from collections import defaultdict

color_map = defaultdict(dict)
color_map[nn.Conv2d]['fill'] = '#FF6F61' # Coral red
color_map[nn.ReLU]['fill'] = 'skyblue'
color_map[nn.MaxPool2d]['fill'] = '#88B04B' # Sage green
color_map[nn.Flatten]['fill'] = 'gold'
color_map[nn.Linear]['fill'] = '#6B5B95'    # Royal purple

input_shape = (1, 3, 224, 224)
visualtorch.layered_view(model, input_shape=input_shape, color_map=color_map
```

![custom-color](https://github.com/willyfh/visualtorch/assets/5786636/57f28191-d86e-4419-a015-f5fc7fa17b5a)

## Contributing

Please feel free to send a pull request to contribute to this project.

## License

This poject is available as open source under the terms of the [MIT License](https://github.com/willyfh/visualtorch/blob/update-readme/LICENSE).

Originally, this project was based on the [visualkeras](https://github.com/paulgavrikov/visualkeras) (under the MIT license), with additional modifications inspired by [pytorchviz](https://github.com/szagoruyko/pytorchviz), and [pytorch-summary](https://github.com/sksq96/pytorch-summary), both of which are also licensed under the MIT license.

## Citation

Please cite this project in your publications if it helps your research as follows:

```bibtex
@misc{Hendria2024VisualTorch,
  author = {Hendria, Willy Fitra},
  title = {visualtorch},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  note = {\url{https://github.com/willyfh/visualtorch}},
}
```
