"""Setup file for visualtorch."""

# Copyright (C) 2024 Willy Fitra Hendria
# SPDX-License-Identifier: MIT

from pathlib import Path

import setuptools

file_path = Path("README.md")
with file_path.open("r") as fh:
    long_description = fh.read()


def _read_requirements(file: str) -> list:
    file_path = Path(file)
    with file_path.open("r") as fh:
        reqs = fh.read()
    return reqs.strip().split("\n")


setuptools.setup(
    name="visualtorch",
    version="0.2.4",
    author="Willy Fitra Hendria",
    author_email="willyfitrahendria@gmail.com",
    description="Architecture visualization of Torch models",
    keywords=["visualize architecture", "torch visualization", "visualtorch"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/willyfh/visualtorch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=_read_requirements("requirements.txt"),
    extras_require={"dev": _read_requirements("docs/requirements.txt") + _read_requirements("dev-requirements.txt")},
    python_requires=">=3.10",
    license="MIT",
    license_files=("LICENSE",),
)
