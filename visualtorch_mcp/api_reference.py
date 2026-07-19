"""Upstream VisualTorch documentation pointers exposed through MCP."""

from __future__ import annotations

from dataclasses import dataclass

from . import MCP_API_VERSION, __version__


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

STYLE_DESCRIPTIONS = {
    "graph": "Node-and-edge architecture diagram with configurable neuron and layer detail.",
    "flow": "Stacked volumetric or two-dimensional layer boxes connected by funnels.",
    "lenet": "Classic LeNet-style stacked feature-map planes.",
}

# Keep discovery lightweight and deterministic. Importing the public VisualTorch option
# dataclasses also imports PyTorch; doing that from a live stdio request can block the MCP
# event loop on Windows. The focused MCP contract test guards these names/defaults against
# drift from the public dataclasses and palettes.
COMMON_OPTION_SPECS: dict[str, tuple[str, object]] = {
    "input_dtype": ("dtype string | list[dtype string | null] | null", None),
    "to_file": ("string | null", None),
    "color_map": ("object | null", None),
    "palette": ("string", "okabe_ito"),
    "background_fill": ("color string | integer array", "white"),
    "padding": ("integer", 10),
    "opacity": ("integer", 255),
    "font": ("'default' | font descriptor | null", None),
    "font_color": ("color string | integer array", "black"),
    "level_gap": ("integer | null", None),
}

STYLE_OPTION_SPECS: dict[str, dict[str, tuple[str, object]]] = {
    "graph": {
        "node_size": ("integer", 50),
        "layer_spacing": ("integer", 250),
        "node_spacing": ("integer", 10),
        "type_ignore": ("list[type name] | null", None),
        "outline_width": ("integer", 1),
        "connector_fill": ("color string | integer array", "gray"),
        "connector_width": ("integer", 1),
        "ellipsize_after": ("integer", 10),
        "show_neurons": ("boolean", True),
        "show_dimension": ("boolean", False),
        "show_input": ("boolean", True),
        "show_arrows": ("boolean", False),
        "legend": ("boolean", False),
        "legend_position": ("string", "bottom-left"),
    },
    "flow": {
        "min_z": ("integer", 10),
        "min_xy": ("integer", 10),
        "max_z": ("integer", 400),
        "max_xy": ("integer", 2000),
        "scale_z": ("number", 0.1),
        "scale_xy": ("number", 1),
        "type_ignore": ("list[type name] | null", None),
        "outline_width": ("integer", 1),
        "low_dim_orientation": ("string", "z"),
        "draw_volume": ("boolean", True),
        "spacing": ("integer", 10),
        "draw_funnel": ("boolean", True),
        "shade_step": ("integer", 10),
        "legend": ("boolean", False),
        "legend_position": ("string", "bottom-left"),
        "show_dimension": ("boolean", False),
        "show_input": ("boolean", True),
        "connector_fill": ("color string | integer array | null", None),
        "connector_width": ("integer", 1),
        "one_dim_orientation": ("string | null", None),
    },
    "lenet": {
        "min_z": ("integer", 1),
        "min_xy": ("integer", 10),
        "max_xy": ("integer", 2000),
        "scale_z": ("number", 1),
        "scale_xy": ("number", 1),
        "type_ignore": ("list[type name] | null", None),
        "outline_width": ("integer", 1),
        "low_dim_orientation": ("string", "z"),
        "spacing": ("integer", 10),
        "draw_funnel": ("boolean", True),
        "shade_step": ("integer", 10),
        "max_channels": ("integer", 100),
        "offset_z": ("integer", 10),
        "show_dimension": ("boolean", True),
        "show_input": ("boolean", True),
        "connector_fill": ("color string | integer array | null", None),
        "connector_width": ("integer", 1),
        "one_dim_orientation": ("string | null", None),
        "legend": ("boolean", False),
        "legend_position": ("string", "bottom-left"),
    },
}

PALETTE_SPECS: dict[str, list[str]] = {
    "okabe_ito": ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"],
    "tol_bright": ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"],
    "tol_muted": [
        "#CC6677",
        "#332288",
        "#DDCC77",
        "#117733",
        "#88CCEE",
        "#882255",
        "#44AA99",
        "#999933",
        "#AA4499",
    ],
    "tab10": [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ],
    "grayscale": ["#404040", "#595959", "#737373", "#8c8c8c", "#a6a6a6", "#bfbfbf", "#d9d9d9"],
    "nord": ["#bf616a", "#d08770", "#ebcb8b", "#a3be8c", "#b48ead", "#8fbcbb", "#88c0d0", "#81a1c1", "#5e81ac"],
    "dracula": ["#FF5555", "#FFB86C", "#F1FA8C", "#50FA7B", "#8BE9FD", "#BD93F9", "#FF79C6"],
    "gruvbox": ["#fb4934", "#b8bb26", "#fabd2f", "#83a598", "#d3869b", "#8ec07c", "#fe8019"],
    "solarized": ["#b58900", "#cb4b16", "#dc322f", "#d33682", "#6c71c4", "#268bd2", "#2aa198", "#859900"],
    "material": ["#f44336", "#e91e63", "#9c27b0", "#3f51b5", "#2196f3", "#009688", "#4caf50", "#ffc107", "#ff5722"],
    "catppuccin": [
        "#f38ba8",
        "#fab387",
        "#f9e2af",
        "#a6e3a1",
        "#94e2d5",
        "#89dceb",
        "#89b4fa",
        "#b4befe",
        "#cba6f7",
        "#f5c2e7",
    ],
}

OPTION_DESCRIPTIONS = {
    "input_dtype": "Torch dtype for one input, or one dtype/null entry per model input.",
    "to_file": "Destination managed by the MCP output_path/output_dir arguments.",
    "color_map": "Override colors by layer type; MCP object keys are Python or torch.nn type names.",
    "palette": "Named built-in palette used when color_map has no override.",
    "background_fill": "Background color, preferably a CSS/Pillow color string for MCP clients.",
    "padding": "Outer image padding in pixels.",
    "opacity": "Layer fill opacity from 0 (transparent) to 255 (opaque).",
    "font": "Built-in default font or a custom TrueType/OpenType font descriptor.",
    "font_color": "Annotation color, preferably a CSS/Pillow color string for MCP clients.",
    "level_gap": "Optional override for vertical spacing between architecture levels.",
    "node_size": "Graph node diameter in pixels.",
    "layer_spacing": "Horizontal space between graph layers in pixels.",
    "node_spacing": "Space between nodes within a graph layer in pixels.",
    "type_ignore": "Layer types to omit; MCP values are Python or torch.nn type names.",
    "outline_width": "Layer or node outline width in pixels.",
    "connector_fill": "Connector color; null selects the renderer default where supported.",
    "connector_width": "Connector line width in pixels.",
    "ellipsize_after": "Collapse large graph layers after this many visible neurons.",
    "show_neurons": "Draw individual neurons instead of layer-level graph nodes when possible.",
    "show_dimension": "Show tensor dimensions on the diagram.",
    "show_input": "Include the model input in the rendered architecture.",
    "show_arrows": "Draw directional arrowheads on graph connectors.",
    "min_z": "Minimum rendered depth/channel-axis size.",
    "min_xy": "Minimum rendered spatial-axis size.",
    "max_z": "Maximum rendered depth/channel-axis size.",
    "max_xy": "Maximum rendered spatial-axis size.",
    "scale_z": "Scale factor for the depth/channel axis.",
    "scale_xy": "Scale factor for spatial axes.",
    "low_dim_orientation": "Axis used for one-dimensional layers: x, y, or z.",
    "draw_volume": "Render flow layers as volumes where their dimensions permit it.",
    "spacing": "Space between adjacent layer drawings in pixels.",
    "draw_funnel": "Draw funnel-shaped connectors between layers.",
    "shade_step": "Brightness step applied to shaded layer faces.",
    "legend": "Add a layer-color legend to the diagram.",
    "legend_position": "Legend placement above or below the rendered diagram, aligned left, center, or right.",
    "one_dim_orientation": "Deprecated alias for low_dim_orientation.",
    "max_channels": "Maximum number of feature-map channels drawn for a LeNet-style layer.",
    "offset_z": "Depth-axis offset between LeNet-style feature-map planes.",
}

ANIMATION_OPTIONS: dict[str, dict[str, object]] = {
    "frame_duration": {
        "type": "integer",
        "default": 600,
        "minimum": 1,
        "unit": "milliseconds",
        "description": "Display duration for each intermediate reveal frame.",
    },
    "final_hold_duration": {
        "type": "integer",
        "default": 1500,
        "minimum": 1,
        "unit": "milliseconds",
        "description": "Display duration for the final complete architecture frame.",
    },
    "loop": {
        "type": "boolean",
        "default": True,
        "description": "Loop forever when true; play once when false.",
    },
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
    if style is not None:
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


def capabilities_manifest(style: str | None = None) -> dict[str, object]:
    """Build a machine-readable description of the MCP and VisualTorch render options."""
    canonical_style = normalize_style_name(style) if style is not None else None
    style_names = [canonical_style] if canonical_style else list(STYLE_OPTION_SPECS)
    common_options = _options_manifest(COMMON_OPTION_SPECS, scope="common")
    styles: dict[str, object] = {}
    for style_name in style_names:
        style_options = _options_manifest(STYLE_OPTION_SPECS[style_name], scope=style_name)
        styles[style_name] = {
            "aliases": list(STYLE_ALIASES[style_name]),
            "description": STYLE_DESCRIPTIONS[style_name],
            "options": {**common_options, **style_options},
        }

    return {
        "schema_version": MCP_API_VERSION,
        "server": {
            "name": "visualtorch-mcp",
            "transport": "stdio",
            "mcp_api_version": MCP_API_VERSION,
            "visualtorch_version": __version__,
        },
        "security": {
            "executes_python_source": True,
            "trust_requirement": "trusted-source-only",
            "sandboxed": False,
            "process_model": "Rendering runs in a child process, but it inherits the local user's permissions.",
            "warning": (
                "Never submit untrusted source; it can access local files, environment variables, and the network."
            ),
        },
        "input_shape": {
            "accepted": ["non-empty list of positive integers", "non-empty list of per-input shapes"],
            "examples": [[1, 3, 224, 224], [[1, 3, 224, 224], [1, 10]]],
            "legacy_json_string": True,
            "includes_batch_dimension": True,
        },
        "styles": styles,
        "palettes": {name: list(colors) for name, colors in sorted(PALETTE_SPECS.items())},
        "animation_options": {name: dict(spec) for name, spec in ANIMATION_OPTIONS.items()},
        "outputs": {
            "visualize_model": {
                "kind": "image",
                "media_type": "image/png",
                "default_extension": ".png",
                "metadata": ["output_path", "kind", "style", "media_type", "width", "height", "mode", "bytes"],
            },
            "animate_model": {
                "kind": "animation",
                "media_type": "image/gif",
                "default_extension": ".gif",
                "metadata": [
                    "output_path",
                    "kind",
                    "style",
                    "media_type",
                    "width",
                    "height",
                    "mode",
                    "bytes",
                    "frame_count",
                    "durations_ms",
                    "loop",
                ],
            },
        },
        "tools": {
            "visualize_model": "Render a static PNG and return file metadata.",
            "animate_model": "Render an animated GIF and return file metadata.",
            "visualtorch_capabilities": "Discover styles, options, palettes, outputs, and safety constraints.",
            "visualtorch_reference": "Get canonical upstream documentation links.",
        },
        "resources": [
            "visualtorch://capabilities",
            "visualtorch://docs",
            "visualtorch://api-reference",
            "visualtorch://version",
        ],
    }


# Backward-compatible name for the MCP tool/resource created before docs were wired in.
api_reference = docs_manifest


def normalize_style_name(style: str) -> str:
    """Return the canonical VisualTorch style name for a user-facing alias."""
    if not isinstance(style, str) or not style.strip():
        message = "style must be a non-empty string."
        raise ValueError(message)
    normalized = style.strip().lower().replace("-", "_")
    for canonical, aliases in STYLE_ALIASES.items():
        if normalized == canonical or normalized in aliases:
            return canonical
    supported = sorted({alias for aliases in STYLE_ALIASES.values() for alias in aliases})
    message = f"Unsupported style {style!r}. Supported styles and aliases: {', '.join(supported)}."
    raise ValueError(message)


def _options_manifest(
    option_specs: dict[str, tuple[str, object]],
    *,
    scope: str,
) -> dict[str, dict[str, object]]:
    """Add descriptions and MCP encodings to the lightweight public option contract."""
    result: dict[str, dict[str, object]] = {}
    for name, (option_type, default) in option_specs.items():
        item: dict[str, object] = {
            "type": option_type,
            "required": False,
            "scope": scope,
            "description": OPTION_DESCRIPTIONS.get(name, "VisualTorch rendering option."),
            "default": default,
        }
        _add_mcp_option_behavior(name, item)
        result[name] = item
    return result


def _add_mcp_option_behavior(
    name: str,
    item: dict[str, object],
) -> None:
    """Document MCP-specific JSON coercion and managed values."""
    if name == "palette":
        item["enum"] = sorted(PALETTE_SPECS)
    elif name == "type_ignore":
        item["mcp_format"] = "array of type-name strings, e.g. ['ReLU', 'torch.nn.Dropout']"
    elif name == "color_map":
        item["mcp_format"] = (
            "object mapping type-name strings to {fill, outline}; each color is a Pillow-compatible "
            "string or an RGB/RGBA integer array"
        )
    elif name == "input_dtype":
        item["mcp_format"] = "dtype name string or one string/null per input, e.g. 'float32'"
    elif name in {"background_fill", "font_color", "connector_fill"}:
        item["mcp_format"] = "Pillow-compatible color string or RGB/RGBA integer array"
    elif name == "font":
        item["mcp_format"] = "'default' or {'path': '/path/to/font.ttf', 'size': 16}; size range is 1..512"
    elif name == "to_file":
        item["mcp_available"] = False
        item["reason"] = "Use output_path or output_dir; the worker manages to_file."
    elif name == "low_dim_orientation":
        item["enum"] = ["x", "y", "z"]
    elif name == "one_dim_orientation":
        item["deprecated"] = True
        item["enum"] = ["x", "y", "z", None]
    elif name == "legend_position":
        item["enum"] = [
            "top-left",
            "top-right",
            "top-center",
            "bottom-left",
            "bottom-right",
            "bottom-center",
        ]
