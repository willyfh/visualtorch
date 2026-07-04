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

# Copyright (C) 2024 Willy Fitra Hendria
# SPDX-License-Identifier: MIT

import json
import os
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib import patheffects
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = REPO_ROOT / "docs" / "source" / "_static" / "images"
LOGO_PATH = IMAGE_DIR / "logos" / "fire-icon.png"
REPO = "willyfh/visualtorch"
ACCENT_COLOR = "#E69F00"  # first color of visualtorch's own "okabe_ito" palette.
SMOOTHING_WINDOW_DAYS = 10  # small enough that a genuine growth spurt still reads as a sharp rise.

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


def _smoothed_daily(dates: list[datetime], counts: list[int]) -> tuple[list[datetime], np.ndarray]:
    """Resample to one point per day, then apply a small rolling average.

    Collapses the raw per-star timestamps (which jump in sharp vertical steps) into a smooth
    curve. The day grid always includes the exact final timestamp (not just whole days since the
    first star) - otherwise a floor-day grid can land up to 24h before the true last star and
    silently drop a recent burst from the average entirely.
    """
    total_days = (dates[-1] - dates[0]).days
    day_grid = [dates[0] + timedelta(days=i) for i in range(total_days + 1)]
    if day_grid[-1] < dates[-1]:
        day_grid.append(dates[-1])

    daily_counts = np.interp(
        [d.timestamp() for d in day_grid],
        [d.timestamp() for d in dates],
        counts,
    )

    window = min(SMOOTHING_WINDOW_DAYS, len(daily_counts))
    kernel = np.ones(window) / window
    padded = np.pad(daily_counts, (window // 2, window - window // 2 - 1), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return day_grid, smoothed


def _render_theme(
    dates: list[datetime],
    counts: list[int],
    logo: Image.Image,
    theme_name: str,
) -> None:
    theme = _THEMES[theme_name]
    plot_dates, plot_counts = _smoothed_daily(dates, counts)

    with plt.xkcd(scale=1, length=150, randomness=2):
        # xkcd() hardcodes a white outline stroke on every line/spine/text (invisible on a white
        # background, but a stark halo on a dark one) - re-point it at this theme's own
        # background so the stroke blends in instead, on both themes.
        plt.rcParams["path.effects"] = [patheffects.withStroke(linewidth=4, foreground=theme["figure"])]

        fig, ax = plt.subplots(figsize=(6.5, 5), dpi=150)
        fig.patch.set_facecolor(theme["figure"])
        fig.patch.set_edgecolor(theme["figure"])
        ax.set_facecolor(theme["figure"])

        ax.plot(plot_dates, plot_counts, color=ACCENT_COLOR, linewidth=2.5, label=REPO)
        ax.fill_between(plot_dates, plot_counts, color=ACCENT_COLOR, alpha=0.15)

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
            x=0.58,
            y=0.945,
            va="center",
            fontsize=18,
            fontweight="bold",
            color=theme["text"],
        )
        imagebox = OffsetImage(logo, zoom=0.5)
        ab = AnnotationBbox(
            imagebox,
            (0.42, 0.945),
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
    _plot(timestamps)


if __name__ == "__main__":
    main()
