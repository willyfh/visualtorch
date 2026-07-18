"""Client-neutral MCP integration for VisualTorch."""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["MCP_API_VERSION", "__version__"]

MCP_API_VERSION = "1.0"

try:
    __version__ = version("visualtorch")
except PackageNotFoundError:
    __version__ = "0+unknown"
