"""What a chat conversation keeps beyond the Pod that served it.

Chat runs on one runner Pod per conversation, and the session router reclaims
a Pod that has been idle past its TTL — its disk goes with it. Two things have
to outlive that, and the SDK already has a home for each, the same homes the
codex orchestration uses:

* **The conversation**: the workspace and the CLI's own session transcript
  (under ``.agent-home/<cli>/``), so the next turn on a fresh Pod continues
  instead of starting over. Kept in the SDK session archive — the private
  ``agent-sessions`` bucket — restored before a turn and saved after it. The
  disk is a cache; storage is the original.
* **The outputs**: files the user should be able to download. Whatever the
  turn put in ``outputs/`` is collected by the SDK artifact collector, stored
  privately through Memento and handed to the browser as signed links on the
  final event (``pdfFiles`` → ``done.files`` and ``chats.messages``). Not
  served from the workspace: ``/runs/{id}/download`` is not a chat path, so
  the router sends it to the shared Deployment, which never had this Pod's
  disk — not even while the Pod is alive.

The archive is not a download mechanism. Its bucket is private and nothing
serves it to a browser; it exists so the *agent* gets its files back.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cliagents import registry
from processgpt_agent_sdk import ArchivePart, SessionArchive
from processgpt_agent_sdk.artifacts import (
    ArtifactCollector,
    MementoArtifactStore,
    download_file,
    snapshot,
)

from .settings import AGENT_TYPE, settings
from .workspace import Workspace

logger = logging.getLogger(__name__)

#: The directory whose contents are the turn's deliverables — the SDK artifact
#: convention. Anything elsewhere in the workspace is working material.
OUTPUTS_DIR = "outputs"

#: Where each CLI keeps its config and sessions inside a run's workspace
#: (``core.bridge.runtime_env``).
_AGENT_HOME = ".agent-home"

#: Top-level names inside a CLI home that never go to storage.
_HOME_EXCLUDE = (
    # Credentials, should a CLI ever write them here. A private bucket is
    # still not where a login belongs.
    ".credentials.json",
    "auth.json",
    # Rewritten on every turn, and can carry a tenant's MCP credentials.
    "config.toml",
    "agents",
    "skills",
    "commands",
    "plugins",
    # Account and cache state the CLI rebuilds on its own. Claude Code's
    # `.claude.json` carries the signed-in account; resuming needs only the
    # transcript under `projects/`.
    ".claude.json",
    "backups",
    ".tmp",
    "tmp",
    "cache",
    "log",
    "statsig",
    "shell-snapshots",
)


def archive_parts(workspace: Workspace) -> list[ArchivePart]:
    """The directories that make up one conversation, in restore order.

    The workspace comes first. Restore skips any part whose directory already
    holds files, and the CLI homes live *inside* the workspace — restored
    first, they would make the workspace look present and leave it empty.
    """
    home = workspace.path / _AGENT_HOME
    return [
        ArchivePart("workspace", workspace.path, exclude=(_AGENT_HOME,)),
        *(
            ArchivePart(f"home-{agent_id}", home / agent_id, exclude=_HOME_EXCLUDE)
            for agent_id in registry.ids()
        ),
    ]


def _archive() -> SessionArchive:
    return SessionArchive(AGENT_TYPE)


async def restore(workspace: Workspace, *, tenant_id: str, conversation_id: str) -> int:
    """Bring a conversation back onto this Pod's disk. Files restored.

    A failure starts the turn without its history rather than not at all; the
    SDK archive logs what it could not read.
    """
    try:
        return await _archive().restore(
            tenant_id, conversation_id, archive_parts(workspace)
        )
    except Exception:  # noqa: BLE001 - an answer without context beats no answer
        logger.warning(
            "session restore failed; continuing without it | conversation=%s",
            conversation_id,
            exc_info=True,
        )
        return 0


async def save(workspace: Workspace, *, tenant_id: str, conversation_id: str) -> int:
    """Store the conversation so a reclaimed Pod does not take it along.

    Never raises: the turn's answer has already been sent and stored, and a
    failed upload must not turn it into a failed turn.
    """
    try:
        return await _archive().save(tenant_id, conversation_id, archive_parts(workspace))
    except Exception:  # noqa: BLE001 - see above
        logger.warning("session save failed | conversation=%s", conversation_id, exc_info=True)
        return 0


# -- outputs -----------------------------------------------------------------


def outputs_dir(workspace: Workspace) -> Path:
    return workspace.path / OUTPUTS_DIR


def before_turn(workspace: Workspace) -> dict:
    """What ``outputs/`` holds before the turn, so only its own files count."""
    return snapshot(outputs_dir(workspace))


@dataclass
class Published:
    """The outcome of handing a turn's outputs to the user."""

    #: Download records, in the shape ``done.files`` / ``pdfFiles`` carry.
    files: list[dict[str, Any]] = field(default_factory=list)
    #: Files the turn produced that did not get a link.
    unlinked: list[str] = field(default_factory=list)
    #: Why, when the whole publication failed.
    reason: str = ""


async def publish(
    workspace: Workspace,
    *,
    tenant_id: str,
    conversation_id: str,
    before: dict,
) -> Published:
    """Store this turn's outputs and return links to them."""
    directory = outputs_dir(workspace)
    produced = _changed(directory, before)
    if not produced:
        return Published()

    if not settings.memento_base_url:
        return Published(unlinked=produced, reason="MEMENTO_BASE_URL is not set")

    collector = ArtifactCollector(MementoArtifactStore(settings.memento_base_url))
    try:
        records = await collector.collect(
            directory,
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            before=before,
            # The CLI reads its files from disk; nothing needs them indexed,
            # and indexing is what has stalled turns for minutes elsewhere.
            index=False,
        )
    except Exception as exc:  # noqa: BLE001 - reported to the user instead
        logger.warning("output publication failed | conversation=%s", conversation_id, exc_info=True)
        return Published(unlinked=produced, reason=f"{type(exc).__name__}: {exc}")

    linked = {str(r.get("file_name") or "") for r in records}
    return Published(
        files=[download_file(r) for r in records],
        unlinked=[name for name in produced if name not in linked],
    )


def _changed(directory: Path, before: dict) -> list[str]:
    """Names of non-empty files this turn added to or changed in ``directory``.

    The same test the SDK collector applies, keyed the way its snapshot is,
    so "produced" and "linked" are counted over the same files.
    """
    if not directory.is_dir():
        return []
    names: list[str] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        stat = path.stat()
        if stat.st_size <= 0:
            continue
        if before.get(str(path.resolve()).casefold()) == (stat.st_size, stat.st_mtime_ns):
            continue
        names.append(path.name)
    return names
