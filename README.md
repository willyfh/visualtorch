# ⭐ VisualTorch ⭐

**VisualTorch** aims to help visualize Torch-based neural network architectures. Currently, this package supports generating layered-style architectures for Torch Sequential models. This package is adapted from [visualkeras](https://github.com/paulgavrikov/visualkeras) by [@paulgavrikov](https://github.com/paulgavrikov).

**v0.1**: Support for layered architecture of torch Sequential.

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

### Legend

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

![simple-cnn](https://github.com/willyfh/visualtorch/assets/5786636/9b646fac-c336-4253-ac01-8f3e6b2fcc0b)

### Save the image

```python
visualtorch.layered_view(model, input_shape=input_shape, legend=True, to_file='output.png') # write to disk
```

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
