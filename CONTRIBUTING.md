# Contributing to Anomalib

We welcome your feedbacks and contributions!

## Bug Report and Feature Request

We use GitHub issues to track bugs and feature requests. If you want to report a bug or request a feature, please create a new issue [here](https://github.com/willyfh/visualtorch/issues). Please make sure there are no existing issues to avoid duplication.

## Discussion

The [GitHub Discussions](https://github.com/willyfh/visualtorch/discussions/) is enabled in visualtorch to help the community asking questions and/or propose ideas/solutions.

## Development & PRs

We welcome your pull requests:

###  Getting Started

#### 1. Fork and Clone the Repository

First, fork the VisualTorch repository by following the GitHub documentation on [forking a repo](https://docs.github.com/en/enterprise-cloud@latest/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo). Then, clone your forked repository to your local machine and create a new branch from `main`.

#### 2. Set Up Your Development Environment

Set up your development environment to start contributing. This involves installing the required dependencies and setting up pre-commit hooks for code quality checks. Note that this guide assumes you are using [Conda](https://docs.conda.io/en/latest/) for package management. However, the steps are similar for other package managers.

<details>
<summary>Development Environment Setup Instructions</summary>

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
If you would like to manually run the checks for all files, use:

```bash
pre-commit run --all-files
```

To bypass pre-commit hooks temporarily (e.g., for a work-in-progress commit),
use:

```bash
git commit -m 'WIP commit' --no-verify
```

However, make sure to address any pre-commit issues before finalizing your pull request.

</details>

### Making Changes

1. **Write Code:** Follow the project's coding standards and write your code with clear intent. Ensure your code is well-documented and includes examples where appropriate. For code quality we use ruff, whose configuration is in [`pyproject.toml`](pyproject.toml) file.

2. **Add Tests:** If your code includes new functionality, add corresponding tests using `pytest`.
   
4. **Update Documentation:** If you've changed APIs or added new features, update the documentation accordingly.

5. **Pass Tests and Quality Checks:** Ensure the test suite passes and that your code meets quality standards by running:

   ```bash
   pre-commit run --all-files
   pytest tests/
   ```
   
6. **Check Licensing:** Ensure you own the code or have rights to use it, adhering to appropriate licensing.

### Submitting Pull Requests

Once you've followed the above steps and are satisfied with your changes:

1. Push your changes to your forked repository.
2. Go to the original VisualTorch repository you forked and click "New pull request".
3. Choose your fork and the branch with your changes to open a pull request.
4. Fill in the necessary details about your changes.

We look forward to your contributions!

## License

You accept that your contributions will be licensed under the `MIT License` if you contribute to this repository. If this is a concern, please notify the maintainers.
