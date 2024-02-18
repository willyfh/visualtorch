import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="visualtorch",
    version="0.1",
    author="Willy Fitra Hendria",
    author_email="willyfitrahendria@gmail.com",
    description="Architecture visualization of Torch models",
    keywords=["visualize architecture", "torch visualization", "visualtorch"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/willyfh/visualtorch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "pillow>=10.0.0",
        "numpy>=1.18.1",
        "aggdraw>=1.3.11",
        "torch>=2.0.0",
    ],
    python_requires=">=3.6",
)
