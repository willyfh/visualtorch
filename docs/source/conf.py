"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html

-- Project information -----------------------------------------------------
https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
"""

from __future__ import annotations

import sys
from pathlib import Path
from sphinx_gallery.sorting import ExplicitOrder

# Define the path to your module using Path
module_path = Path(__file__).parent.parent / "src"

# Insert the path to sys.path
sys.path.insert(0, str(module_path.resolve()))

project = "VisualTorch"
copyright = "2024, Willy Fitra Hendria"  # noqa: A001
release = "2024"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

myst_enable_extensions = [
    "colon_fence",
    # other MyST extensions...
]
nbsphinx_allow_errors = True
templates_path = ["_templates"]
exclude_patterns: list[str] = []

# Automatic exclusion of prompts from the copies
# https://sphinx-copybutton.readthedocs.io/en/latest/use.html#automatic-exclusion-of-prompts-from-the-copies
copybutton_exclude = ".linenos, .gp, .go"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_logo = "_static/images/logos/fire-icon.png"
html_favicon = "_static/images/logos/fire-icon.png"
html_static_path = ["_static"]
html_theme_options = {
    "logo": {
        "text": "<b>VisualTorch</b>",
    },
    "icon_links": [
        {
            "name": "Downloads",
            "url": "https://pepy.tech/project/visualtorch",
            "icon": "https://static.pepy.tech/personalized-badge/visualtorch?period=total&units=international_system&left_color=grey&right_color=orange&left_text=PyPI%20Downloads",
            "type": "url",
        },
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/willyfh/visualtorch",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-brands fa-square-github",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        },
    ],
}


sphinx_gallery_conf = {
    "examples_dirs": "../examples",  # path to your example scripts
    "gallery_dirs": "usage_examples",  # path to where to save gallery generated output
    "min_reported_time": 10,
    "subsection_order": ExplicitOrder(["../examples/layered", "../examples/graph"]),
}

exclude_patterns = [
    # exclude .py and .ipynb files in usage_examples generated by sphinx-gallery
    # this is to prevent sphinx from complaining about duplicate source files
    "usage_examples/*/*.ipynb",
    "usage_examples/*/*.py",
]
