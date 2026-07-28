"""Guards against the docs install snippet drifting from setup.py's version.

`docs/source/snippets/install/pypi.txt` is regenerated automatically from setup.py's version on
every Sphinx build (see docs/source/conf.py), so this shouldn't drift in practice. This test is a
cheap, build-independent safety net for the case where docs get built some other way that skips
conf.py's normal execution - it was manually maintained and missed on 3 releases in a row before
the auto-regeneration was added.
"""

# Copyright (C) 2024 VisualTorch Contributors
# SPDX-License-Identifier: MIT

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _setup_py_version() -> str:
    setup_py = (REPO_ROOT / "setup.py").read_text(encoding="utf-8")
    match = re.search(r'version\s*=\s*"([^"]+)"', setup_py)
    assert match is not None, 'Could not find version="..." in setup.py.'
    return match.group(1)


def test_pypi_install_snippet_matches_setup_py_version() -> None:
    """The pinned `pip install visualtorch==X.Y.Z` snippet must match setup.py's version."""
    snippet_path = REPO_ROOT / "docs" / "source" / "snippets" / "install" / "pypi.txt"
    snippet = snippet_path.read_text(encoding="utf-8").strip()
    expected = f"pip install visualtorch=={_setup_py_version()}"

    assert snippet == expected, (
        f"{snippet_path} is stale - run a docs build to regenerate it, "
        f"or update it to match setup.py's version ({_setup_py_version()})."
    )
