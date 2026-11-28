"""
Unified tqdm utility — consistent progress-bar configuration across the codebase.

Provides a single entry-point for creating :class:`tqdm` instances with
project-wide defaults.  Supports both terminal and Jupyter notebook
environments via ``tqdm.auto``.

Usage::

    from flex.ml_utils.tqdm_utils import pbar, tqdm_write

    # Quick drop-in replacement for tqdm.tqdm(...)
    for item in pbar(items, desc="Training"):
        ...

    # Temporarily disable all progress bars
    pbar.disable()
    ...
    pbar.enable()

    # Configure global defaults
    pbar.set_defaults(ncols=120, leave=False)

    # Or create a custom-configured instance
    from flex.ml_utils.tqdm_utils import TqdmWrapper
    my_pbar = TqdmWrapper(disable=False, ncols=100, leave=False, unit="batch")
    for item in my_pbar(items, desc="Loading"):
        ...
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, TypeVar

from tqdm.auto import tqdm as _auto_tqdm

_T = TypeVar("_T")

# ---------------------------------------------------------------------------
# TqdmWrapper — callable class with explicit config as constructor args
# ---------------------------------------------------------------------------


class TqdmWrapper:
    """Callable wrapper around ``tqdm.auto.tqdm`` with project-wide defaults.

    The constructor accepts explicit configuration arguments (not opaque
    ``**kwargs``) so that IDE auto-completion and type-checking work
    reliably.  Extra keyword arguments can be passed via ``**extra`` and
    are forwarded to the underlying tqdm constructor when the instance is
    called.

    Parameters
    ----------
    disable:
        Globally suppress all progress bars created by this wrapper.
    ncols:
        Width of the progress bar in characters.  ``None`` means
        auto-detect (the tqdm default).
    leave:
        Whether to leave the finished bar on screen.  Defaults to
        ``False`` to prevent stacked bars from accumulating when
        multiple progress bars are created in quick succession
        (e.g. multiple ``NoniidDataGenerator`` instances loading data).
    mininterval:
        Minimum update interval in seconds (default 0.1).
    unit:
        Unit string for iteration speed display (default ``"it"``).
    bar_format:
        Custom bar format string forwarded to tqdm.
    position:
        Bar position offset (for nested bars).  ``None`` means auto.
    **extra:
        Additional keyword arguments forwarded as defaults to every
        :meth:`__call__` invocation.
    """

    def __init__(
        self,
        disable: bool = False,
        ncols: Optional[int] = None,
        leave: Optional[bool] = None,
        mininterval: float = 0.1,
        unit: str = "it",
        bar_format: Optional[str] = None,
        position: Optional[int] = None,
        **extra: Any,
    ) -> None:
        self._disabled: bool = disable

        # Build the defaults dict from explicit args (only store non-None values
        # for optional params so they don't override tqdm's own defaults unless
        # the user intentionally set them).
        self._defaults: dict[str, Any] = dict(extra)
        self._defaults["mininterval"] = mininterval
        self._defaults["unit"] = unit
        if ncols is not None:
            self._defaults["ncols"] = ncols
        if leave is not None:
            self._defaults["leave"] = leave
        if bar_format is not None:
            self._defaults["bar_format"] = bar_format
        if position is not None:
            self._defaults["position"] = position

    # -- callable interface --------------------------------------------------

    def __call__(
        self,
        iterable: Optional[Iterable[_T]] = None,
        desc: Optional[str] = None,
        total: Optional[int] = None,
        disable: Optional[bool] = None,
        **kwargs: Any,
    ) -> _auto_tqdm[_T]:  # type: ignore[valid-type]
        """Create a :class:`tqdm` progress bar with this wrapper's defaults.

        All keyword arguments are forwarded to the underlying
        ``tqdm.auto.tqdm`` constructor.  Per-call kwargs take precedence
        over the wrapper's configured defaults.

        Parameters
        ----------
        iterable:
            An iterable to wrap.  Pass ``None`` to create a
            manually-updated bar.
        desc:
            Prefix for the progress bar.
        total:
            The number of expected iterations (required when *iterable*
            is ``None``).
        disable:
            Override the wrapper-level disable flag for this specific bar.
        **kwargs:
            Additional arguments forwarded to ``tqdm.auto.tqdm``.

        Returns
        -------
        ``tqdm.auto.tqdm``
        """
        if disable is None:
            disable = self._disabled

        # Merge: wrapper defaults first, then per-call kwargs override
        merged: dict[str, Any] = {}
        merged.update(self._defaults)
        merged.update(kwargs)

        # Project-wide single-slot policy.  A caller must not leave completed
        # bars behind or allocate another terminal row: training commonly nests
        # a batch bar inside a round bar, and AWS runs relay output through a
        # pipe.  Enforcing these values here prevents one accidental
        # per-call display overrides from bringing stacked bars back.
        merged["leave"] = False
        merged["position"] = 0
        merged["ascii"] = True

        return _auto_tqdm(
            iterable=iterable,
            desc=desc,
            total=total,
            disable=disable,
            **merged,
        )

    # -- configuration API ---------------------------------------------------

    def disable(self) -> None:
        """Globally suppress all progress bars produced by this wrapper."""
        self._disabled = True

    def enable(self) -> None:
        """Globally enable progress bars (the default)."""
        self._disabled = False

    @property
    def is_disabled(self) -> bool:
        """Return ``True`` when progress bars are globally suppressed."""
        return self._disabled

    def set_defaults(self, **kwargs: Any) -> None:
        """Set keyword arguments that are merged into every call.

        Example::

            pbar.set_defaults(ncols=120, leave=False, unit="batch")
        """
        self._defaults.update(kwargs)

    def clear_defaults(self) -> None:
        """Remove all previously-set default keyword arguments."""
        self._defaults.clear()

    def get_defaults(self) -> dict[str, Any]:
        """Return a copy of the current default keyword arguments."""
        return dict(self._defaults)

    # -- static helpers ------------------------------------------------------

    @staticmethod
    def write(message: str, **kwargs: Any) -> None:
        """Write a message via tqdm without interfering with active progress bars.

        Thin wrapper around ``tqdm.auto.tqdm.write``.  Use this instead
        of ``print()`` inside a tqdm loop to avoid garbled output.

        Parameters
        ----------
        message:
            The message string to print.
        **kwargs:
            Forwarded to ``tqdm.auto.tqdm.write`` (e.g. ``file=sys.stderr``).
        """
        _auto_tqdm.write(message, **kwargs)


# ---------------------------------------------------------------------------
# Module-level convenience instances & functions
# ---------------------------------------------------------------------------

# Default progress-bar factory — use this as the drop-in replacement for
# ``tqdm.tqdm(...)``.  It is an instance of :class:`TqdmWrapper`, which
# means it is callable and also exposes ``.disable()``, ``.enable()``,
# ``.set_defaults()``, and ``.write()``.
#
# Usage::
#     from flex.ml_utils.tqdm_utils import pbar
#     for x in pbar(items, desc="Processing"):
#         ...

pbar = TqdmWrapper(
    ncols=120,
    leave=False,
    mininterval=0.1,
    ascii=True,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
)


def tqdm_write(message: str, **kwargs: Any) -> None:
    """Write a message via tqdm without interfering with active progress bars.

    Convenience function that delegates to :meth:`TqdmWrapper.write`.
    Use this instead of ``print()`` inside a tqdm loop to avoid garbled
    output.

    Parameters
    ----------
    message:
        The message string to print.
    **kwargs:
        Forwarded to ``tqdm.auto.tqdm.write`` (e.g. ``file=sys.stderr``).
    """
    _auto_tqdm.write(message, **kwargs)
