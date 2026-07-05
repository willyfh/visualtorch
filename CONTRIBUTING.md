# Contributing to VisualTorch

We welcome your feedbacks and contributions!

## Bug Report and Feature Request

We use GitHub issues to track bugs and feature requests. If you want to report a bug or request a feature, please create a new issue [here](https://github.com/willyfh/visualtorch/issues). Please make sure there are no existing issues to avoid duplication.

## Discussion

The [GitHub Discussions](https://github.com/willyfh/visualtorch/discussions/) is enabled in visualtorch to help the community asking questions and/or propose ideas/solutions.

## Picking Up an Issue

Before starting work, please comment on the issue to let others know you're picking it up, so
effort isn't duplicated. If you're a first-time contributor, we'd appreciate it if you claimed just
one issue at a time (especially ones labeled `good first issue`) rather than several at once - it
gives more people a chance to get involved, and you're welcome to grab another once your first PR
is merged.

## Development & PRs

###  Getting Started

#### 1. Fork and Clone the Repository

Fork the VisualTorch repository by following the GitHub documentation on [forking a repo](https://docs.github.com/en/enterprise-cloud@latest/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo). After forking, clone your forked repository to your local machine, and then create a new branch from the `main` branch.

#### 2. Set Up Your Development Environment

Set up your development environment to begin contributing. This includes installing the necessary dependencies and configuring pre-commit hooks for code quality checks. Please note that this guide assumes you are using Conda for package management, but the steps are similar for other package managers.

##### Development Environment Setup Instructions

1. Create and activate a new Conda environment:

   ```bash
   conda create -n visualtorch_env python=3.10
   conda activate visualtorch_env
   ```

2. Install the development requirements:

   ```bash
   pip install -e .[dev]
   ```

3. Install and configure pre-commit hooks:

   ```bash
   pre-commit install
   ```

Pre-commit hooks help ensure code quality and consistency. After each commit,
`pre-commit` will automatically run the configured checks for the changed file.
To manually run the checks for all files, use:

```bash
pre-commit run --all-files
```

### Making Changes

1. **Write Code:** Follow the project's coding standards and write your code with clear intent. Ensure your code is well-documented and includes examples where appropriate. For code quality we use `ruff`.

2. **Add Tests:** If your code includes new functionality, add corresponding tests using `pytest`.

3. **Update Documentation:** If you've changed APIs or added new features, update the documentation accordingly.

4. **Pass Tests and Quality Checks:** Ensure the test suite passes and that your code meets quality standards by running:

   ```bash
   pre-commit run --all-files
   pytest tests/
   ```

5. **Check Licensing:** Ensure you either own the code or have the rights to use it, in accordance with the appropriate licensing.

### Adding an Example

Gallery examples live under `docs/examples/{graph,flow,lenet_style}/`, one script per example
(`plot_*.py`). Sphinx-gallery auto-discovers every `plot_*.py` file in those directories - there's
no manifest or index to update, and no separate registration step.

Each script should follow the pattern of the existing examples: a module docstring explaining
what the example demonstrates, a small model definition (or a directly imported one, e.g. from
`torchvision`), and a call to `visualtorch.render(...)` whose output is displayed via
`matplotlib`. Since readthedocs actually executes every example during the docs build, any new
third-party dependency your example needs (beyond what's already installed) must be added to
`docs/requirements.txt`.

Before opening a PR, run your new example script locally to confirm it executes without error.

### Submitting Pull Requests

After completing the necessary steps and ensuring your changes meet the requirements:

1. Push your changes to your forked repository.
2. Visit the original VisualTorch repository you forked and select "New pull request".
3. Choose your fork and the relevant branch with your changes to create a pull request.
4. Provide all necessary details about your modifications.

## License

By contributing to this repository, you agree to license your contributions under the MIT License. Please contact the maintainers if you have any concerns regarding licensing.
