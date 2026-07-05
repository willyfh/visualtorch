# Governance

This document describes how decisions are made in VisualTorch and how maintainer roles work. For
how to actually submit a change, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Roles

**Maintainer**

- Reviews and merges pull requests.
- Cuts releases and publishes to PyPI.
- Has final say on the project's direction, breaking changes, and API design.

**Contributor**

- Anyone who opens an issue, discussion, or pull request.
- Pull requests require review and approval from a maintainer before merging.

Current maintainer(s):

- [@willyfh](https://github.com/willyfh)

## Decision-Making

For day-to-day changes (bug fixes, examples, documentation, small features), a maintainer reviews
and merges directly. For larger or breaking changes (a new major version, a rewrite of a core
module), the maintainer(s) discuss the approach in the relevant issue or pull request before
implementing, so the reasoning is visible and reviewable, not just the final diff.

If there is more than one maintainer and they disagree on a decision, they work toward consensus
through discussion on the relevant issue or PR. There is no formal voting process at this project's
current size - this section will be revisited if the maintainer team grows enough to need one.

## Becoming a Maintainer

There's no fixed contribution count or tenure requirement, but a contributor is a strong
candidate for becoming a maintainer once they've:

- Submitted several non-trivial, well-tested pull requests over a sustained period (not just a
  single one-off contribution).
- Demonstrated familiarity with the codebase's conventions (e.g. the shared tracing backend in
  `visualtorch/backend.py`, the per-style rendering modules, the test suite).
- Shown a willingness to help review other contributors' pull requests, not just submit their own.

An existing maintainer nominates the candidate (e.g. in an issue or discussion), and current
maintainers approve by consensus. New maintainers are added to this document and given write
access to the repository.

## Release Process

Releases are cut by a maintainer: a version bump PR is merged to `main`, then a GitHub Release is
published with a matching `vX.Y.Z` tag, which triggers an automated build and upload to PyPI.
Semantic versioning is followed - breaking changes require a major version bump, and are
accompanied by a deprecation path where practical (see recent examples in the changelog) rather
than a silent removal.
