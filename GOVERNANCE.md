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

A contributor who brings in and takes ownership of a substantial existing component (e.g. merging
in a tool they've already built and maintained elsewhere) can also become a maintainer directly,
without the sustained-PR history above. The trust signal there is the track record of the
existing work, not incremental PRs to this repo.

An existing maintainer nominates the candidate (e.g. in an issue, discussion, or pull request).
Any existing maintainer may approve the nomination - this doesn't require active sign-off from
every maintainer, just visibility so others can raise concerns if they have any. New maintainers
are added to this document and are typically given write access to the repository right away.

Any maintainer who objects can still reverse the promotion within 14 days of it being announced,
as long as they give a clear reason. Past that window, the promotion stands.

## Stepping Down / Inactivity

A maintainer can step down at any time by opening a PR removing themselves from this document.

If a maintainer is unresponsive (no reviews, merges, or activity) for 6+ months with no notice,
the remaining maintainer(s) may open an issue proposing their removal, giving the inactive
maintainer at least 2 weeks to respond before removing them from this document and revoking write
access. This isn't punitive, it's about keeping the maintainer list accurate for anyone
(contributors, Ecosystem reviewers) relying on it.

## Release Process

Releases are cut by a maintainer: a version bump PR is merged to `main`, then a GitHub Release is
published with a matching `vX.Y.Z` tag, which triggers an automated build and upload to PyPI.
Semantic versioning is followed - breaking changes require a major version bump, and are
accompanied by a deprecation path where practical (see recent examples in the changelog) rather
than a silent removal.
