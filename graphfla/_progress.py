"""Progress bars for the library's verbose loops, rendered with rich.

All of graphfla's progress reporting funnels through :func:`track`, so the bar
implementation lives in exactly one place. ``track`` wraps an iterable in a
rich progress bar when ``verbose`` is true, and is otherwise a zero-overhead
pass-through. If rich is not installed it degrades silently to plain iteration,
so progress display is never load-bearing.
"""

from __future__ import annotations

from typing import Iterable, Optional


def track(
    iterable: Iterable,
    *,
    description: str,
    total: Optional[int] = None,
    verbose: bool = False,
) -> Iterable:
    """Return *iterable*, wrapped in a rich progress bar when *verbose*.

    Parameters
    ----------
    iterable : iterable
        The sequence to iterate over.
    description : str
        Label shown to the left of the bar.
    total : int, optional
        Number of steps; inferred via ``len(iterable)`` when omitted and the
        iterable is sized, otherwise the bar is shown as indeterminate.
    verbose : bool, default False
        When false (the default) *iterable* is returned unchanged, with no
        display and no overhead.

    Returns
    -------
    iterable
        Either *iterable* itself, or a generator yielding the same items while
        advancing a progress bar.
    """
    if not verbose:
        return iterable
    try:
        return _rich_track(iterable, description, total)
    except ImportError:
        # rich is an optional convenience; never let its absence break a run.
        return iterable


def _rich_track(iterable, description, total):
    # Imported lazily so ``import graphfla`` never pays for rich unless a
    # verbose loop actually runs.
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeRemainingColumn,
    )

    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=Console(stderr=True),  # keep bars off stdout (the data channel)
        transient=False,
    )

    def _iter():
        with progress:
            task = progress.add_task(description, total=total)
            for item in iterable:
                yield item
                progress.advance(task)

    return _iter()
