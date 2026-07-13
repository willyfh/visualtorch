"""Shared GIF-assembly orchestration for every style's `*_view_animate()` entry point.

The reveal-by-column animation loop is style-agnostic: call a per-style "render one frame at this
reveal cutoff" callback once per column, collect frames, then either return them or save a GIF.
All layout-specific knowledge (what a "reveal cutoff" filters, how a column count is obtained)
stays inside each style module's own frame callback - this module knows nothing about
Box/Circle/ColumnLayout/etc.
"""

# Copyright (C) 2024 VisualTorch Contributors
# SPDX-License-Identifier: MIT

from collections.abc import Callable

from PIL import Image


def animate_frames(
    render_frame: Callable[[int], Image.Image],
    n_columns: int,
    to_file: str | None,
    frame_duration: int,
    final_hold_duration: int,
    loop: bool,
) -> list[Image.Image] | None:
    """Render one frame per reveal cutoff `0..n_columns-1`, then return or save them as a GIF.

    Args:
        render_frame: Given `reveal_up_to` (a column index, inclusive), render and return that
            frame's full-canvas image. Called once per column, in increasing order.
        n_columns: Total number of (already-filtered) columns in the traced architecture - the
            resulting frame count.
        to_file: Path to write the animated GIF to. If None, no file is written and the list of
            frames is returned instead. Format is inferred from the filename, same as every other
            `*_view()`'s `to_file` - a non-`.gif` extension won't raise, it'll silently save only
            the first frame, since `save_all` is GIF-specific.
        frame_duration: Per-frame display duration in milliseconds, for every frame except the
            last.
        final_hold_duration: Display duration in milliseconds for the last (fully revealed) frame,
            so the completed diagram holds on screen before the GIF loops.
        loop: Whether the GIF should loop forever (`loop=0` to PIL) or play once (`loop=1`).

    Returns:
        list[Image.Image] | None: The rendered frames if `to_file` is None, else None.
    """
    frames = [render_frame(reveal_up_to) for reveal_up_to in range(n_columns)]
    durations = [frame_duration] * (n_columns - 1) + [final_hold_duration]

    if to_file is not None:
        frames[0].save(
            to_file,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0 if loop else 1,
        )
        return None
    return frames
