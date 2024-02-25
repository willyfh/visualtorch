<div align="center">
 <h1>🔥 VisualTorch 🔥</h1>

[![python](https://img.shields.io/badge/python-3.10%2B-blue)]() [![pytorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)]() [![Downloads](https://static.pepy.tech/personalized-badge/visualtorch?period=total&units=international_system&left_color=grey&right_color=green&left_text=PyPI%20Downloads)](https://pepy.tech/project/visualtorch) [![Run Tests](https://github.com/willyfh/visualtorch/actions/workflows/pytest.yml/badge.svg)](https://github.com/willyfh/visualtorch/actions/workflows/pytest.yml)

</div>

**VisualTorch** aims to help visualize Torch-based neural network architectures. It currently supports generating layered-style and graph-style architectures for PyTorch Sequential and Custom models. This tool is adapted from [visualkeras](https://github.com/paulgavrikov/visualkeras), [pytorchviz](https://github.com/szagoruyko/pytorchviz), and [pytorch-summary](https://github.com/sksq96/pytorch-summary).

**Note:** VisualTorch may not yet support complex models, but contributions are welcome!

![layered-and-graph](https://github.com/willyfh/visualtorch/assets/5786636/694e6e6c-58ea-46d6-9280-348337c08ec7)

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

![simple-cnn](https://github.com/willyfh/visualtorch/assets/5786636/e8da2a52-66c6-4fda-85f8-7243702fd1f2)

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

![simple-cnn-custom](https://github.com/willyfh/visualtorch/assets/5786636/9f18db76-838d-4cd1-87ac-3ac5d3509423)

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

![graph](https://github.com/willyfh/visualtorch/assets/5786636/9868f8be-7bfb-4892-ad3b-72de56955c75)

### Save the Image

```python
visualtorch.layered_view(model, input_shape=input_shape, legend=True, to_file='output.png')
```

### 2D View

```python
visualtorch.layered_view(model, input_shape=input_shape, draw_volume=False)
```

![2d-view](https://github.com/willyfh/visualtorch/assets/5786636/71848bfa-5447-4e66-bf4c-84f9e51a581e)

### Custom Color

Use 'fill' to change the color of the layer, and use 'outline' to change the color of the lines.

```python
from collections import defaultdict

color_map = defaultdict(dict)
color_map[nn.Conv2d]['fill'] = 'LightSlateGray' # Light Slate Gray
color_map[nn.ReLU]['fill'] = '#87CEFA' # Light Sky Blue
color_map[nn.MaxPool2d]['fill'] = 'LightSeaGreen' # Light Sea Green
color_map[nn.Flatten]['fill'] = '#98FB98' # Pale Green
color_map[nn.Linear]['fill'] = 'LightSteelBlue' # Light Steel Blue

input_shape = (1, 3, 224, 224)
visualtorch.layered_view(model, input_shape=input_shape, color_map=color_map
```

![custom-color](https://github.com/willyfh/visualtorch/assets/5786636/2e536ffd-8441-4e66-90ff-d152da67363e)

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
