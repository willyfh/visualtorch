<div align="center">
 <h1>🔥 VisualTorch 🔥</h1>

[![python](https://img.shields.io/badge/python-3.10%2B-blue)]() [![pytorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)]() [![Downloads](https://static.pepy.tech/personalized-badge/visualtorch?period=total&units=international_system&left_color=grey&right_color=green&left_text=PyPI%20Downloads)](https://pepy.tech/project/visualtorch) [![Run Tests](https://github.com/willyfh/visualtorch/actions/workflows/pytest.yml/badge.svg)](https://github.com/willyfh/visualtorch/actions/workflows/pytest.yml) [![Documentation Status](https://readthedocs.org/projects/visualtorch/badge/?version=latest)](https://visualtorch.readthedocs.io/en/latest/?badge=latest)

</div>

**VisualTorch** aims to help visualize Torch-based neural network architectures. It currently supports generating flow-style, graph-style, and LeNet-style architectures for PyTorch Sequential and Custom models. Its original visual styles were inspired by [visualkeras](https://github.com/paulgavrikov/visualkeras), [pytorchviz](https://github.com/szagoruyko/pytorchviz), [pytorch-summary](https://github.com/sksq96/pytorch-summary), and [torchview](https://github.com/mert-kurttutan/torchview); since then, it has grown its own unified tracing backend and architecture-handling logic well beyond its origins.

**Note:** `1.0+` is a major release with breaking API changes, but with significantly better features and algorithms - upgrading is recommended. For the old API, use `0.2.5` or older.

**Limitation:** VisualTorch traces a real forward pass to build the diagram, which has an inherent
limitation shared by any tracing-based approach (not a bug, and not fixable without full symbolic
execution): models with **data-dependent control flow** (e.g. a branch only taken if a tensor
value crosses some threshold) only show whichever branch the traced dummy input happened to take.
Separately, a layer that returns **multiple meaningful output tensors** (e.g. a custom multi-task
head, or `nn.LSTM`'s `(output, (h_n, c_n))`) still has its node's size based on only its first
tensor; with `show_dimension=True`, every output tensor's shape is shown in the label, not just
the first. Downstream connections are correct either way. Contributions are welcome!

<div align="center">

![VisualTorch Examples](https://raw.githubusercontent.com/willyfh/visualtorch/e6ad79751e0f7412b1074beb45f9baeccd1419e4/docs/source/_static/images/banners/readme-examples.png)

</div>

### Animated Reveal

Every style can also render as an animated GIF, revealing the model one layer/column at a
time - see `flow_view_animate`/`graph_view_animate`/`lenet_view_animate` in the
[usage examples](https://visualtorch.readthedocs.io/en/latest/usage_examples/index.html).

<div align="center">

![Animated VisualTorch Example](https://raw.githubusercontent.com/willyfh/visualtorch/57ce9d41e7a2dfdb76c4b6cf0df82b0c5c0846e5/docs/source/_static/images/banners/readme-animated-demo.gif)

</div>

## Documentation

Online documentation is available at [visualtorch.readthedocs.io](https://visualtorch.readthedocs.io/en/latest/).

The docs include [usage examples](https://visualtorch.readthedocs.io/en/latest/usage_examples/index.html), [API references](https://visualtorch.readthedocs.io/en/latest/markdown/api_references/index.html), and other useful information.

## Installation

See the [Installation page](https://visualtorch.readthedocs.io/en/latest/markdown/get_started/installation.html).

## MCP integration

VisualTorch includes an optional MCP server for generating architecture diagrams from model source
provided by an MCP client. Install it with `pip install "visualtorch[mcp]"` and see the
[MCP integration guide](https://visualtorch.readthedocs.io/en/latest/markdown/get_started/mcp.html)
for configuration and usage details.

## Used in Research

VisualTorch has been used in published research, including works published in Nature, IEEE, and MDPI.

See the [Research Showcase page](https://visualtorch.readthedocs.io/en/latest/markdown/showcase/index.html) for the full list.

Used VisualTorch in your research, built something with it, or found a paper that cites it? [Tell us about it](https://github.com/willyfh/visualtorch/discussions) or [open a pull request](https://github.com/willyfh/visualtorch/pulls) to add it directly - we'd love to hear.

## Examples

See the [Usage Examples page](https://visualtorch.readthedocs.io/en/latest/usage_examples/index.html).

## Contributing

Please feel free to send a pull request to contribute to this project by following this [guideline](https://github.com/willyfh/visualtorch/blob/main/CONTRIBUTING.md).

## Releases

See [GOVERNANCE.md](https://github.com/willyfh/visualtorch/blob/main/GOVERNANCE.md#release-process) for release methodology and cadence, and the [PyPI release history](https://pypi.org/project/visualtorch/#history) for past releases.

## License

This poject is available as open source under the terms of the [MIT License](https://github.com/willyfh/visualtorch/blob/main/LICENSE.md).

Originally, this project was based on the [visualkeras](https://github.com/paulgavrikov/visualkeras) (under the MIT license), with additional modifications inspired by [pytorchviz](https://github.com/szagoruyko/pytorchviz), [pytorch-summary](https://github.com/sksq96/pytorch-summary), and [torchview](https://github.com/mert-kurttutan/torchview), all of which are also licensed under the MIT license.

## Citation

Please cite this project in your publications if it helps your research.

**Note:** the paper below describes VisualTorch as of its publication date (2024). The project has
since been substantially refactored, including breaking API changes (see the
[documentation](https://visualtorch.readthedocs.io/en/latest/) for the current API) - the DOI
always resolves to what was actually reviewed and published.

```bibtex
@article{Hendria2024,
  doi = {10.21105/joss.06678},
  url = {https://doi.org/10.21105/joss.06678},
  year = {2024},
  publisher = {The Open Journal},
  volume = {9},
  number = {102},
  pages = {6678},
  author = {Willy Fitra Hendria and Paul Gavrikov},
  title = {VisualTorch: Streamlining Visualization for PyTorch Neural Network Architectures},
  journal = {Journal of Open Source Software}
}
```

## Star History

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/willyfh/visualtorch/assets/docs/source/_static/images/star-history-dark.png" />
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/willyfh/visualtorch/assets/docs/source/_static/images/star-history-light.png" />
  <img alt="Star History Chart" src="https://raw.githubusercontent.com/willyfh/visualtorch/assets/docs/source/_static/images/star-history-light.png" />
</picture>
