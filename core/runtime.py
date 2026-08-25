"""Transactional ownership of provider files written for one CLI attempt."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RuntimeLease:
    root: Path
    _before: dict[Path, bytes | None] = field(default_factory=dict)

    def capture(self, paths: list[Path]) -> None:
        root = self.root.resolve()
        for candidate in paths:
            path = candidate.resolve()
            if path != root and root not in path.parents:
                raise PermissionError(f"runtime path escapes workspace: {candidate}")
            if path in self._before:
                continue
            self._before[path] = path.read_bytes() if path.is_file() else None

    def restore(self) -> None:
        root = self.root.resolve()
        for path, content in reversed(list(self._before.items())):
            if content is None:
                if path.is_file() or path.is_symlink():
                    path.unlink()
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            parent = path.parent
            while parent != root and root in parent.parents:
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent
        self._before.clear()

