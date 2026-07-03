# Installation

Installation can be done through PyPI or the GitHub repository. PyPI installation should be chosen if you want to use the release version. If you prefer to use the latest version from the GitHub repository, then installation from the source is recommended.

:::::{dropdown} Installation via pip
:open:

VisualTorch can be installed using the following commands:

::::{tab-set}

:::{tab-item} PyPI
:sync: label-1

```{literalinclude} ../../snippets/install/pypi.txt
:language: bash
```

:::

:::{tab-item} Source
:sync: label-2

```{literalinclude} ../../snippets/install/source.txt
:language: bash
```

:::
::::

:::::

Note: `1.0+` is a major release with breaking API changes, but with significantly better
features and algorithms - upgrading is recommended. For the old API, use `0.2.5` or older
(`pip install visualtorch==0.2.5`), with docs at
[readthedocs.io/en/v0.2.5](https://visualtorch.readthedocs.io/en/v0.2.5/).

The following dependencies will also be installed if you run the above command:

```{literalinclude} ../../snippets/install/requirements.txt
:language: bash
```
