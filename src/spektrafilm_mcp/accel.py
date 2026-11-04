"""Speed-ups for spektrafilm's hot spots that don't change its output.

Profiling a 9 MP export tile showed ~40% of the time in output gamut
compression (colour-science CAM16-UCS round trips in float64), which runs on
a single core. It is a pure per-pixel transform with no numba inside, so it is
split into row bands and run on a thread pool: numpy releases the GIL in its
array kernels, and the result is identical to the single-threaded call.

Numba code must NOT be wrapped this way: its default threading layer aborts
the process when parallel kernels are entered from several threads.
"""

from __future__ import annotations

import functools
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

WORKERS = max(1, min(os.cpu_count() or 4, 12))
MIN_PIXELS = 300_000  # below this, threading overhead outweighs the gain
_pool = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="spektrafilm-band")


def parallel_rows(fn):
    """Run a per-pixel (..., 3) -> (..., 3) function over row bands in parallel."""
    warmed = False

    @functools.wraps(fn)
    def wrapper(rgb, *args, **kwargs):
        nonlocal warmed
        arr = np.asarray(rgb)
        if arr.ndim != 3 or arr.shape[0] * arr.shape[1] < MIN_PIXELS:
            return fn(arr, *args, **kwargs)
        if not warmed:
            # Let lazily-built lookup tables and caches fill on one thread first.
            fn(arr[:2, :2], *args, **kwargs)
            warmed = True
        bands = np.array_split(arr, min(WORKERS, arr.shape[0]), axis=0)
        return np.concatenate(list(_pool.map(lambda band: fn(band, *args, **kwargs), bands)), axis=0)

    wrapper.__wrapped__ = fn
    return wrapper


def install() -> None:
    """Patch spektrafilm in place. Safe to call more than once."""
    from spektrafilm.runtime.stages import scanning

    if not hasattr(scanning.compress_rgb, "__wrapped__"):
        scanning.compress_rgb = parallel_rows(scanning.compress_rgb)
