"""One directory per run: where the agent works and what survives afterwards.

Isolation here is not a nicety. Two work items from two tenants run in the same
container, and the only thing keeping one from reading the other's draft
contract is that they were given different directories and a permission level
that cannot leave them.

The directory outlives the run on purpose. A paused run resumes into it, a user
downloads from it, and a replay diffs against it — all of which need it to
still be there after the process that made it has exited.
"""

from __future__ import annotations

import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from .settings import settings

#: Anything outside this becomes an underscore. Run identifiers come from the
#: database, and a path is not the place to find out one contained "../".
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def _safe(component: str, *, fallback: str = "run") -> str:
    cleaned = _UNSAFE.sub("_", (component or "").strip())[:120].strip("._-")
    return cleaned or fallback


@dataclass(frozen=True)
class Workspace:
    """A run's working root."""

    path: Path
    run_id: str

    @property
    def exists(self) -> bool:
        return self.path.is_dir()

    def contains(self, candidate: str | Path) -> bool:
        """True when ``candidate`` is inside this workspace.

        Resolves both sides first: a path that reaches outside via ``..`` or a
        symlink is outside, whatever it looks like textually. This is the check
        the download route depends on, so it answers about the *real* location.
        """
        try:
            resolved = Path(candidate).resolve()
            root = self.path.resolve()
        except OSError:
            return False
        return resolved == root or root in resolved.parents

    def resolve_within(self, relative: str) -> Path:
        """Turn a client-supplied relative path into a real one, or refuse."""
        candidate = (self.path / relative).resolve()
        if not self.contains(candidate):
            raise PermissionError(f"path escapes the workspace: {relative}")
        return candidate

    def relative(self, path: str | Path) -> str:
        try:
            return str(Path(path).resolve().relative_to(self.path.resolve()))
        except (ValueError, OSError):
            return str(path)

    def files(self) -> list[Path]:
        if not self.exists:
            return []
        return sorted(p for p in self.path.rglob("*") if p.is_file())


def locate(run_id: str, *, tenant_id: str = "") -> Workspace:
    """Where a run's workspace *would* be, without creating anything.

    Read-only callers must use this. :func:`for_run` creates the directory,
    which would make a swept run indistinguishable from an empty one — and
    would let anyone with a URL litter the volume with directories.

    Tenant-scoped so a listing of the root cannot be read as a listing of one
    tenant's work, and so retention can be reasoned about per tenant later.
    """
    parts = [settings.workspace_root]
    if tenant_id:
        parts.append(Path(_safe(tenant_id, fallback="tenant")))
    parts.append(Path(_safe(run_id)))

    path = Path(os.path.join(*[str(p) for p in parts]))
    return Workspace(path=path, run_id=run_id)


def for_run(run_id: str, *, tenant_id: str = "") -> Workspace:
    """The workspace for ``run_id``, created if this is its first run."""
    workspace = locate(run_id, tenant_id=tenant_id)
    workspace.path.mkdir(parents=True, exist_ok=True)
    return workspace


def sweep(*, now: float | None = None) -> list[Path]:
    """Delete workspaces past the retention window. Returns what went.

    Age is taken from the most recently touched file rather than from the
    directory's own mtime: a run that was resumed yesterday is not stale
    because its folder was created last week.
    """
    root = settings.workspace_root
    if not root.is_dir():
        return []

    cutoff = (now or time.time()) - settings.retention_seconds
    removed: list[Path] = []

    for candidate in _run_dirs(root):
        try:
            if _last_touched(candidate) >= cutoff:
                continue
            shutil.rmtree(candidate)
            removed.append(candidate)
        except OSError:
            # A workspace we cannot remove is a disk-space problem, not a
            # reason to abandon the rest of the sweep.
            continue
    return removed


def _run_dirs(root: Path) -> list[Path]:
    """Run directories, whether or not they sit under a tenant folder."""
    found: list[Path] = []
    for first in root.iterdir() if root.is_dir() else []:
        if not first.is_dir():
            continue
        children = [c for c in first.iterdir() if c.is_dir()]
        # A tenant folder holds run folders; a run folder holds the agent's own
        # files. Treating the former as a run would delete every run inside it.
        if children and all(_looks_like_run(c) for c in children):
            found.extend(children)
        else:
            found.append(first)
    return found


def _looks_like_run(path: Path) -> bool:
    return path.is_dir() and not path.name.startswith(".")


def _last_touched(path: Path) -> float:
    newest = path.stat().st_mtime
    for child in path.rglob("*"):
        try:
            newest = max(newest, child.stat().st_mtime)
        except OSError:
            continue
    return newest
