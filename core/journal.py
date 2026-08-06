"""What a run changed, in a form that can be replayed or undone.

The honest scope first, because it is narrower than the deepagents equivalent
and pretending otherwise would be the actual bug: a CLI agent's own tools run
inside a process we do not instrument. What is observable is (a) calls that
went through our MCP servers and (b) files in the workspace, before and after.

That is enough to replay the effects without re-invoking a model, and enough to
put the files back. It is *not* enough to reproduce a run that was given
command-exec permission and reached outside — so a journal that saw that
records the fact and reports itself as partial rather than claiming a fidelity
it does not have.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

#: Inside the workspace: the file snapshots it references are only meaningful
#: next to the files themselves.
_JOURNAL_DIRNAME = ".processgpt-journal"
_ENTRIES_FILENAME = "entries.jsonl"
_SNAPSHOT_DIRNAME = "snapshots"


@dataclass
class Entry:
    """One replayable change."""

    #: ``tool`` (an MCP call) or ``file`` (a workspace mutation).
    kind: str
    at: float = field(default_factory=time.time)
    tool: str = ""
    arguments: Any = None
    path: str = ""
    change: str = ""
    #: Snapshot ids, not contents: the journal stays readable and the bytes
    #: live next to it.
    before: str | None = None
    after: str | None = None
    #: True when the effect leaves what we can observe — an MCP call that wrote
    #: to another system, for instance.
    external: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


class Journal:
    """Append-only record for one run."""

    def __init__(self, workspace_path: Path) -> None:
        self.root = workspace_path / _JOURNAL_DIRNAME
        self.entries_path = self.root / _ENTRIES_FILENAME
        self.snapshots = self.root / _SNAPSHOT_DIRNAME
        self._out_of_scope: list[str] = []

    # -- writing ---------------------------------------------------------

    def capture_baseline(self, workspace_path: Path) -> int:
        """Snapshot what was already there, before the agent touches anything.

        Without this, undo is wrong in a specific and damaging way: file
        changes are learned about *after* they happen, so the first time a
        pre-existing file is edited there is no recorded "before" — and an undo
        reading that as "did not exist" deletes a file the run only modified.

        Returns how many files were captured.
        """
        captured = 0
        for path in sorted(workspace_path.rglob("*")):
            if not path.is_file() or self._is_internal(path, workspace_path):
                continue
            digest = self._snapshot(path)
            if digest is None:
                continue
            self._append(
                Entry(kind="baseline", path=_relative(path, workspace_path), after=digest)
            )
            captured += 1
        return captured

    def record_tool(self, tool: str, arguments: Any, *, external: bool = True) -> None:
        self._append(Entry(kind="tool", tool=tool, arguments=arguments, external=external))

    def record_file(self, path: Path, change: str, *, workspace_path: Path) -> None:
        """Snapshot a file change, taking ``before`` from the previous entry.

        A creation has no before; a delete has no after. Both are recorded so
        undo knows which direction to move.
        """
        relative = _relative(path, workspace_path)
        before = self._latest_after(relative)
        after = self._snapshot(path) if path.is_file() else None
        self._append(
            Entry(kind="file", path=relative, change=change, before=before, after=after)
        )

    def note_out_of_scope(self, detail: str) -> None:
        """Record that something happened we cannot replay."""
        if detail not in self._out_of_scope:
            self._out_of_scope.append(detail)
        self._append(Entry(kind="out_of_scope", tool=detail, external=True))

    # -- reading ---------------------------------------------------------

    def entries(self) -> list[Entry]:
        if not self.entries_path.is_file():
            return []
        out: list[Entry] = []
        for line in self.entries_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                try:
                    out.append(Entry(**data))
                except TypeError:
                    continue
        return out

    @property
    def replayable(self) -> bool:
        """False when the run did something outside what was recorded."""
        return not any(e.kind == "out_of_scope" for e in self.entries())

    def limitations(self) -> list[str]:
        return [e.tool for e in self.entries() if e.kind == "out_of_scope"]

    # -- replay / undo ---------------------------------------------------

    def replay_files(self, workspace_path: Path) -> list[str]:
        """Re-apply recorded file states in order. No model is involved.

        Returns the paths touched. Raises on the first failure, having said how
        far it got — a half-applied replay the caller knows about beats one it
        does not.
        """
        applied: list[str] = []
        for entry in self.entries():
            if entry.kind != "file":
                continue
            target = workspace_path / entry.path
            if entry.after is None:
                target.unlink(missing_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(self._read_snapshot(entry.after))
            applied.append(entry.path)
        return applied

    def undo_files(self, workspace_path: Path) -> tuple[list[str], list[str]]:
        """Walk the file entries backwards. Returns (restored, irreversible)."""
        restored: list[str] = []
        irreversible = self.limitations()

        for entry in reversed(self.entries()):
            if entry.kind == "tool" and entry.external:
                irreversible.append(f"{entry.tool} 호출은 되돌릴 수 없습니다")
                continue
            if entry.kind != "file":
                continue
            target = workspace_path / entry.path
            if entry.before is None:
                target.unlink(missing_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(self._read_snapshot(entry.before))
            restored.append(entry.path)
        return restored, irreversible

    # -- internals -------------------------------------------------------

    def _append(self, entry: Entry) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.entries_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.as_dict(), ensure_ascii=False) + "\n")

    def _snapshot(self, path: Path) -> str | None:
        try:
            data = path.read_bytes()
        except OSError:
            return None
        self.snapshots.mkdir(parents=True, exist_ok=True)
        # Content-addressed, so an unchanged file costs nothing to record twice
        # and a reverted edit reuses the original blob.
        import hashlib

        digest = hashlib.sha256(data).hexdigest()[:32]
        blob = self.snapshots / digest
        if not blob.exists():
            blob.write_bytes(data)
        return digest

    def _read_snapshot(self, digest: str) -> bytes:
        return (self.snapshots / digest).read_bytes()

    def _latest_after(self, relative: str) -> str | None:
        """The last known content of ``relative``, baseline included.

        The baseline is what makes "modified" distinguishable from "created"
        for a file the run did not make.
        """
        for entry in reversed(self.entries()):
            if entry.kind in ("file", "baseline") and entry.path == relative:
                return entry.after
        return None

    @staticmethod
    def _is_internal(path: Path, workspace_path: Path) -> bool:
        """Our own bookkeeping is not the run's work product."""
        try:
            relative = path.resolve().relative_to(workspace_path.resolve())
        except (ValueError, OSError):
            return False
        return relative.parts and relative.parts[0].startswith(".processgpt")


def _relative(path: Path, workspace_path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(workspace_path).resolve()))
    except (ValueError, OSError):
        return str(path)
