"""Optional tqdm progress reporting for the metadata-cache build.

Opt-in via ``build_metadata_cache(..., progress=True)``. When disabled (the default)
tqdm is never imported and there is zero overhead. Everything here degrades gracefully:

* if ``tqdm`` is somehow not installed the progress just turns off (never raises);
* under a **distributed** dask scheduler the per-file bar cannot be driven — the
  ``dask.diagnostics`` callbacks used by :func:`dask_progress` only fire for the local
  get-based schedulers (``sync``/``threads``/``processes``) — so the caller detects that
  via :func:`is_distributed` and logs a plain line instead.

The build runs out-of-band in the catalog (often a container, not a TTY), so the bar is
kept **visible** even off-TTY (``disable=False``) but throttled with a wide
``mininterval`` so it does not spam ``docker logs``. To hide it off-TTY instead, flip the
single ``disable=False`` below to ``disable=None`` (tqdm then auto-disables when
``stderr`` is not a TTY).
"""
from __future__ import annotations

__all__ = ["progress_iter", "dask_progress", "is_distributed"]

import contextlib
from typing import Iterable, Optional

# Off-TTY refresh throttle (seconds): keep the bar visible in container logs without
# hammering ``docker logs`` with carriage-return redraws.
_MININTERVAL = 2.0


def _load_tqdm():
    """Return the ``tqdm`` class, or ``None`` if it is not importable."""
    try:
        from tqdm import tqdm
    except Exception:  # pragma: no cover - tqdm is a declared dependency
        return None
    return tqdm


def progress_iter(
    iterable: Iterable,
    *,
    enabled: bool,
    desc: str,
    total: Optional[int] = None,
    leave: bool = True,
) -> Iterable:
    """Wrap ``iterable`` in a tqdm bar when ``enabled`` (else return it unchanged).

    ``leave=True`` keeps the finished bar on screen (the outer, per-cube bar);
    ``leave=False`` clears it once exhausted (the inner, per-file bars, so successive
    cubes reuse the same line). Off-TTY the bar stays visible (see the module docstring).
    """
    if not enabled:
        return iterable
    tqdm = _load_tqdm()
    if tqdm is None:
        return iterable
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        leave=leave,
        mininterval=_MININTERVAL,
        disable=False,  # flip to None to auto-hide when stderr is not a TTY
    )


@contextlib.contextmanager
def dask_progress(total: int, *, enabled: bool, desc: str, leave: bool = False):
    """A tqdm-driven ``dask.diagnostics.Callback`` around a ``dask.compute`` batch.

    No-op when disabled or when tqdm is unavailable. The callback increments the bar once
    per completed task, so ``total`` should be the number of delayed tasks (one per file).
    Only fires for the local get-based schedulers; under a distributed scheduler the
    callbacks never run (guard the call with :func:`is_distributed`).
    """
    tqdm = _load_tqdm() if enabled else None
    if tqdm is None:
        yield
        return

    from dask.diagnostics import Callback

    class _TqdmCallback(Callback):
        def _start_state(self, dsk, state):
            self._bar = tqdm(
                total=total,
                desc=desc,
                leave=leave,
                mininterval=_MININTERVAL,
                disable=False,  # flip to None to auto-hide when stderr is not a TTY
            )

        def _posttask(self, key, result, dsk, state, worker_id):
            self._bar.update(1)

        def _finish(self, dsk, state, errored):
            self._bar.close()

    with _TqdmCallback():
        yield


# Local schedulers for which ``dask.diagnostics`` callbacks fire (so the per-file bar
# works). Listed explicitly so an active ambient distributed client does not make an
# explicit ``scheduler="threads"`` look distributed.
_LOCAL_SCHEDULERS = frozenset({
    "threads", "threading", "processes", "multiprocessing",
    "sync", "synchronous", "single-threaded",
})


def is_distributed(scheduler) -> bool:
    """True when the effective dask scheduler is the distributed one.

    ``dask.diagnostics.Callback`` hooks only fire for the local get-based schedulers; under
    distributed they never run, so the per-file bar can't be driven and the caller logs a
    plain line instead. Resolves ``"auto"``/``None`` to the ambient default (an active
    ``Client``, a ``dask.config`` scheduler, ...) via ``dask.base.get_scheduler``.
    """
    if isinstance(scheduler, str) and scheduler in _LOCAL_SCHEDULERS:
        return False
    try:
        from distributed import Client
        if isinstance(scheduler, Client):
            return True
    except Exception:
        pass
    if scheduler in ("distributed", "dask.distributed"):
        return True
    try:
        from dask.base import get_scheduler
        resolved = get_scheduler(
            scheduler=None if scheduler in (None, "auto") else scheduler
        )
    except Exception:
        return False
    return "distributed" in (getattr(resolved, "__module__", "") or "")
