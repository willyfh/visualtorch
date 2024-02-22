# ⭐ VisualTorch ⭐

[![python](https://img.shields.io/badge/python-3.10%2B-blue)]() [![pytorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)]() [![Downloads](https://static.pepy.tech/personalized-badge/visualtorch?period=total&units=international_system&left_color=grey&right_color=green&left_text=PyPI%20Downloads)](https://pepy.tech/project/visualtorch)

**VisualTorch** aims to help visualize Torch-based neural network architectures. Currently, this package supports generating layered-style architectures for Torch Sequential and Functional models. This package is adapted from [visualkeras](https://github.com/paulgavrikov/visualkeras) by [@paulgavrikov](https://github.com/paulgavrikov).

**v0.2**: Support for functional models has been added.

**v0.1.1**: Support for the layered architecture of Torch Sequential.

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

### Functional

```python
import torch.nn as nn

# Example of a simple CNN model with two conv layers
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

# Create an instance of the SimpleCNN
model = SimpleCNN()

input_shape = (1, 3, 224, 224)

visualtorch.layered_view(model, input_shape=input_shape, legend=True).show() # display using your system viewer
```

![simple-cnn-functional](https://github.com/willyfh/visualtorch/assets/5786636/b5f56233-cde1-428f-9b49-8f5088746c7f)

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

Originally, this project was based on the [visualkeras](https://github.com/paulgavrikov/visualkeras) (under the MIT license).

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
