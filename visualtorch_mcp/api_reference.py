"""Upstream VisualTorch documentation pointers exposed through MCP."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DocsPage:
    """Describe one upstream documentation page."""

    title: str
    github_path: str
    source_url: str
    readthedocs_url: str


DOCS_BASE = "https://github.com/willyfh/visualtorch/blob/main/docs"
RAW_BASE = "https://raw.githubusercontent.com/willyfh/visualtorch/main/docs"
RTD_BASE = "https://visualtorch.readthedocs.io/en/latest"

STYLE_ALIASES: dict[str, tuple[str, ...]] = {
    "graph": ("graph",),
    "flow": ("flow", "layered", "layered_view"),
    "lenet": ("lenet", "lenet_style", "lenet_view"),
}

DOCS_PAGES: dict[str, DocsPage] = {
    "index": DocsPage(
        title="VisualTorch Documentation",
        github_path="source/index.md",
        source_url=f"{RAW_BASE}/source/index.md",
        readthedocs_url=f"{RTD_BASE}/",
    ),
    "installation": DocsPage(
        title="Installation",
        github_path="source/markdown/get_started/installation.md",
        source_url=f"{RAW_BASE}/source/markdown/get_started/installation.md",
        readthedocs_url=f"{RTD_BASE}/markdown/get_started/installation.html",
    ),
    "render": DocsPage(
        title="Render",
        github_path="source/markdown/api_references/render.md",
        source_url=f"{RAW_BASE}/source/markdown/api_references/render.md",
        readthedocs_url=f"{RTD_BASE}/markdown/api_references/render.html",
    ),
    "flow": DocsPage(
        title="Flow View",
        github_path="source/markdown/api_references/flow.md",
        source_url=f"{RAW_BASE}/source/markdown/api_references/flow.md",
        readthedocs_url=f"{RTD_BASE}/markdown/api_references/flow.html",
    ),
    "graph": DocsPage(
        title="Graph View",
        github_path="source/markdown/api_references/graph.md",
        source_url=f"{RAW_BASE}/source/markdown/api_references/graph.md",
        readthedocs_url=f"{RTD_BASE}/markdown/api_references/graph.html",
    ),
    "lenet": DocsPage(
        title="LeNet Style View",
        github_path="source/markdown/api_references/lenet_style.md",
        source_url=f"{RAW_BASE}/source/markdown/api_references/lenet_style.md",
        readthedocs_url=f"{RTD_BASE}/markdown/api_references/lenet_style.html",
    ),
    "examples": DocsPage(
        title="Usage Examples",
        github_path="examples",
        source_url=f"{DOCS_BASE}/examples",
        readthedocs_url=f"{RTD_BASE}/usage_examples/index.html",
    ),
}


def docs_manifest(style: str | None = None) -> str:
    """Return upstream VisualTorch docs links without duplicating API documentation."""
    keys = ["index", "installation", "render", "flow", "graph", "lenet", "examples"]
    if style:
        keys = [normalize_style_name(style)]

    lines = [
        "Upstream VisualTorch documentation:",
        f"- GitHub docs root: {DOCS_BASE}",
        f"- Read the Docs root: {RTD_BASE}/",
        "",
        "MCP style aliases accepted by this server:",
    ]
    for canonical, aliases in STYLE_ALIASES.items():
        lines.append(f"- {canonical}: {', '.join(aliases)}")

    lines.extend(["", "Relevant upstream docs pages:"])
    for key in keys:
        page = DOCS_PAGES[key]
        lines.extend(
            [
                f"- {page.title}",
                f"  GitHub: {DOCS_BASE}/{page.github_path}",
                f"  Source: {page.source_url}",
                f"  Read the Docs: {page.readthedocs_url}",
            ],
        )

    return "\n".join(lines)


# Backward-compatible name for the MCP tool/resource created before docs were wired in.
api_reference = docs_manifest


def normalize_style_name(style: str) -> str:
    """Return the canonical VisualTorch style name for a user-facing alias."""
    normalized = style.strip().lower().replace("-", "_")
    for canonical, aliases in STYLE_ALIASES.items():
        if normalized == canonical or normalized in aliases:
            return canonical
    supported = sorted({alias for aliases in STYLE_ALIASES.values() for alias in aliases})
    message = f"Unsupported style {style!r}. Supported styles and aliases: {', '.join(supported)}."
    raise ValueError(message)
