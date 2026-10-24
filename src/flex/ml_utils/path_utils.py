from __future__ import annotations
import os


class PathUtils:
    """Utility helpers for resolving well-known paths relative to the flex library."""

    @staticmethod
    def flex_dir() -> str:
        """Absolute path of the flex package directory (…/src/flex/)."""
        return os.path.dirname(os.path.abspath(__file__ + "/.."))

    @staticmethod
    def lib_parent_dir() -> str:
        """
        Absolute path of the library's parent directory.

        Directory layout::

            <project_root>/          ← returned by this method
                src/
                    flex/            ← flex package
                .dataset/            ← default dataset root

        Computed as two levels up from the flex package ``__init__.py``.
        """
        flex_pkg = os.path.dirname(os.path.abspath(__file__ + "/.."))  # …/src/flex/
        src_dir  = os.path.dirname(flex_pkg)                            # …/src/
        return os.path.dirname(src_dir)                                 # …/<project_root>/

    @staticmethod
    def resolve_path(path: str, base: str | None = None) -> str:
        """
        Return an absolute path from *path*.

        - If *path* is already absolute, return it unchanged.
        - Otherwise, resolve it relative to *base* (defaults to
          :meth:`lib_parent_dir`).
        """
        if os.path.isabs(path):
            return path
        base = base or PathUtils.lib_parent_dir()
        return os.path.normpath(os.path.join(base, path))

    @staticmethod
    def dataset_root(relative: str = ".dataset") -> str:
        """
        Convenience helper: resolve a dataset root path relative to the
        project root directory.

        By default this resolves to ``<project_root>/.dataset``.
        """
        return PathUtils.resolve_path(relative)
