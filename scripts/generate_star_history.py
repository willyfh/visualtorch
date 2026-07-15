"""Regenerate the self-hosted star history chart.

Fetches this repo's stargazer timestamps from the GitHub API and plots cumulative stars over
time, in a hand-drawn "xkcd" style matching star-history.com's own look. Self-hosted because
star-history.com's image-rendering API has been observed to silently return an empty chart for
this repo while the timestamped data itself is available from GitHub. Run this periodically
(e.g. via a scheduled GitHub Actions workflow) to keep the chart current.

Usage: `python scripts/generate_star_history.py` from anywhere - writes directly to
`docs/source/_static/images/star-history-{light,dark}.png`. Set `GITHUB_TOKEN` in the
environment to avoid GitHub's low unauthenticated rate limit.
"""

# Copyright (C) 2024 VisualTorch Contributors
# SPDX-License-Identifier: MIT

import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib import patheffects
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from scipy.interpolate import PchipInterpolator

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = REPO_ROOT / "docs" / "source" / "_static" / "images"
LOGO_PATH = IMAGE_DIR / "logos" / "traced-flame-icon.png"
REPO = "willyfh/visualtorch"
ACCENT_COLOR = "#E69F00"  # first color of visualtorch's own "okabe_ito" palette.

_THEMES = {
    "light": {"figure": "#ffffff", "text": "#24292f"},
    "dark": {"figure": "#0d1117", "text": "#c9d1d9"},
}


def _fetch_stargazer_timestamps() -> list[datetime]:
    """Fetch every stargazer's starred_at timestamp for `REPO`, paginating through all pages."""
    headers = {
        "Accept": "application/vnd.github.star+json",
        "User-Agent": "visualtorch-star-history-script",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    timestamps: list[datetime] = []
    page = 1
    while True:
        url = f"https://api.github.com/repos/{REPO}/stargazers?per_page=100&page={page}"
        request = urllib.request.Request(url, headers=headers)  # noqa: S310
        with urllib.request.urlopen(request) as response:  # noqa: S310
            entries = json.loads(response.read())
        if not entries:
            break
        timestamps.extend(datetime.fromisoformat(entry["starred_at"].replace("Z", "+00:00")) for entry in entries)
        page += 1
    return sorted(timestamps)


def _monotone_curve(
    dates: list[datetime],
    counts: list[int],
    num_points: int = 2000,
) -> tuple[list[datetime], np.ndarray]:
    """Smoothly interpolate a quarterly resampling of the data via monotonic cubic interpolation.

    Matches star-history.com's own approach (d3-shape's `curveMonotoneX`) for the smooth,
    flowing curve look - a handful of big sweeping waves, not a kink at every single star event.
    Interpolating through all real points directly (hundreds of them) still looks visibly
    stair-stepped, since PCHIP preserves every real kink exactly; resampling to one anchor per
    quarter first (taking the true cumulative count as of each anchor time, a step function
    lookup) reduces that noise the same way a coarser data density would, while the true final
    value is still forced to stay exact. A fixed calendar unit scales naturally with the repo's
    actual history length, unlike an arbitrary fixed point count.

    Anchors span from the join date (the first star's timestamp) through right now, not just the
    last star's timestamp - so the line still extends flat to the present even if the most recent
    star happened a while ago, instead of visually stopping short of today.
    """
    x_full = np.array([d.timestamp() for d in dates])
    y_full = np.array(counts, dtype=float)
    now = datetime.now(tz=timezone.utc).timestamp()

    seconds_per_quarter = 91.31 * 24 * 3600
    num_anchors = max(int((now - x_full[0]) / seconds_per_quarter), 2)
    anchor_x = np.linspace(x_full[0], now, num_anchors)
    indices = np.clip(np.searchsorted(x_full, anchor_x, side="right") - 1, 0, len(y_full) - 1)
    anchor_y = y_full[indices]
    anchor_y[-1] = y_full[-1]

    # PchipInterpolator requires strictly increasing x - collapse ties (multiple anchors falling
    # in the same real-data gap can land on the same step value but distinct times, which is
    # fine; true ties only happen at the very start) by keeping the last of each duplicate.
    deduped: dict[float, float] = {}
    for xi, yi in zip(anchor_x, anchor_y, strict=True):
        deduped[xi] = yi
    x = np.array(sorted(deduped))
    y = np.array([deduped[xi] for xi in x], dtype=float)

    interpolator = PchipInterpolator(x, y)
    dense_x = np.linspace(x[0], x[-1], num_points)
    dense_dates = [datetime.fromtimestamp(t, tz=timezone.utc) for t in dense_x]
    return dense_dates, interpolator(dense_x)


def _render_theme(
    dates: list[datetime],
    counts: list[int],
    logo: Image.Image,
    theme_name: str,
) -> None:
    theme = _THEMES[theme_name]
    curve_dates, curve_counts = _monotone_curve(dates, counts)

    with plt.xkcd(scale=1, length=150, randomness=2):
        # xkcd() hardcodes a white outline stroke on every line/spine/text (invisible on a white
        # background, but a stark halo on a dark one) - re-point it at this theme's own
        # background so the stroke blends in instead, on both themes.
        plt.rcParams["path.effects"] = [patheffects.withStroke(linewidth=4, foreground=theme["figure"])]
        # xkcd() also picks whichever comic-style font happens to be installed (e.g. "Comic Sans
        # MS" on macOS), silently falling back to a plain sans-serif elsewhere (e.g. CI's Ubuntu
        # runner has none of them) - pin to a font bundled with matplotlib itself so local and CI
        # renders match.
        plt.rcParams["font.family"] = ["DejaVu Sans"]

        fig, ax = plt.subplots(figsize=(6.5, 5), dpi=150)
        fig.patch.set_facecolor(theme["figure"])
        fig.patch.set_edgecolor(theme["figure"])
        ax.set_facecolor(theme["figure"])

        ax.plot(curve_dates, curve_counts, color=ACCENT_COLOR, linewidth=2.5, label=REPO)
        ax.fill_between(curve_dates, curve_counts, color=ACCENT_COLOR, alpha=0.15)

        ax.set_ylabel("GitHub Stars", color=theme["text"])
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
        ax.margins(x=0.02, y=0.05)

        ax.spines["top"].set_visible(False)  # noqa: FBT003
        ax.spines["right"].set_visible(False)  # noqa: FBT003
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(theme["text"])
        ax.tick_params(colors=theme["text"])

        legend = ax.legend(loc="upper left", frameon=True, fontsize=11)
        legend.get_frame().set_facecolor(theme["figure"])
        legend.get_frame().set_edgecolor(theme["text"])
        for text in legend.get_texts():
            text.set_color(theme["text"])

        fig.subplots_adjust(top=0.82, bottom=0.18, left=0.13, right=0.97)

        fig.suptitle(
            "Star History",
            x=0.59,
            y=0.94,
            va="center",
            fontsize=18,
            fontweight="bold",
            color=theme["text"],
        )
        imagebox = OffsetImage(logo, zoom=0.5)
        ab = AnnotationBbox(
            imagebox,
            (0.41, 0.945),
            xycoords="figure fraction",
            frameon=False,
            box_alignment=(0.5, 0.5),
        )
        ax.add_artist(ab)

        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            IMAGE_DIR / f"star-history-{theme_name}.png",
            facecolor=theme["figure"],
            edgecolor=theme["figure"],
        )
        plt.close(fig)


def _plot(timestamps: list[datetime]) -> None:
    dates = [timestamps[0], *timestamps]
    counts = list(range(len(timestamps) + 1))
    logo = Image.open(LOGO_PATH).convert("RGBA")
    for theme_name in _THEMES:
        _render_theme(dates, counts, logo, theme_name)


def main() -> None:
    """Fetch stargazer history from GitHub and write both theme variants of the chart."""
    timestamps = _fetch_stargazer_timestamps()
    print(f"{REPO}: {len(timestamps)} stars as of {timestamps[-1].isoformat()}")
    _plot(timestamps)


if __name__ == "__main__":
    main()
