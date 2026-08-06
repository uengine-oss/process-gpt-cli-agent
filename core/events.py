"""Normalised exec events → the events ProcessGPT's UI already knows.

The browser has a renderer for `tool_start`, `tool_end`, `plan_*` and text
chunks, built for the deepagents orchestration. Reusing those names is the
difference between "the new agent type streams like the old one" and "someone
writes a second SSE client".

Two rules:

* An event with nothing to show produces nothing. A stream of empty bubbles is
  worse than a quiet one.
* An event we do not recognise still produces something. The whole point of
  passing unknown CLI events through the library is lost if the last hop drops
  them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cliagents import ExecEvent, ExecEventKind

from .settings import settings


@dataclass(frozen=True)
class UiEvent:
    """One SSE payload, in the shape the existing chat client parses."""

    type: str
    data: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"type": self.type, **self.data}


def translate(event: ExecEvent, *, workspace=None) -> list[UiEvent]:
    """Map one exec event onto zero or more UI events."""
    kind = event.kind

    if kind is ExecEventKind.ASSISTANT_TEXT:
        return [UiEvent("text", {"content": event.text})] if event.text else []

    if kind is ExecEventKind.THINKING:
        # Shown, but distinctly: reasoning is context, not the answer.
        return [UiEvent("thinking", {"content": event.text})] if event.text else []

    if kind is ExecEventKind.TOOL_START:
        return [
            UiEvent(
                "tool_start",
                {
                    "tool": event.tool or "tool",
                    "tool_use_id": event.tool_use_id,
                    "input": _truncate_value(event.tool_input),
                },
            )
        ]

    if kind is ExecEventKind.TOOL_END:
        return [
            UiEvent(
                "tool_end",
                {
                    "tool": event.tool or "tool",
                    "tool_use_id": event.tool_use_id,
                    "output": _truncate_value(event.tool_output),
                    "is_error": event.is_error,
                },
            )
        ]

    if kind is ExecEventKind.PLAN:
        return [UiEvent("plan_update", {"items": event.tool_output or []})]

    if kind is ExecEventKind.FILE_CHANGE:
        return [_file_event(event, workspace)]

    if kind is ExecEventKind.PERMISSION_REQUEST:
        return [
            UiEvent(
                "permission_request",
                {
                    "tool": event.tool or "",
                    "detail": event.text,
                    "tool_use_id": event.tool_use_id,
                },
            )
        ]

    if kind is ExecEventKind.USAGE:
        return [UiEvent("usage", {"usage": event.usage or {}})]

    if kind is ExecEventKind.ERROR:
        return [UiEvent("error", {"content": event.text, "error": event.text})]

    if kind is ExecEventKind.RUN_START:
        return [UiEvent("run_start", {"session_id": event.session_id, "model": event.text})]

    if kind is ExecEventKind.RESULT:
        # The caller decides what a result means for the work item; the stream
        # only needs to stop showing progress.
        return []

    # UNKNOWN: whatever the CLI said, still said. Text if it had any, the raw
    # payload otherwise, so a new event type is visible rather than absent.
    if event.text:
        return [UiEvent("agent_log", {"content": event.text})]
    if event.raw:
        return [UiEvent("agent_log", {"raw": _truncate_value(event.raw)})]
    return []


def _file_event(event: ExecEvent, workspace) -> UiEvent:
    """A file artifact, with a preview small enough to send."""
    path = event.path or ""
    relative = workspace.relative(path) if workspace and path else path
    content = event.text or ""
    truncated = False
    if len(content) > settings.file_preview_max_bytes:
        content = content[: settings.file_preview_max_bytes]
        truncated = True
    return UiEvent(
        "file_artifact",
        {
            "path": relative,
            "absolute_path": path,
            "operation": event.change or "modified",
            "content": content,
            "truncated": truncated,
        },
    )


def _truncate_value(value: Any) -> Any:
    """Keep a tool payload legible without shipping a megabyte of it."""
    if isinstance(value, str) and len(value) > settings.file_preview_max_bytes:
        return value[: settings.file_preview_max_bytes] + "\n… (생략됨)"
    return value
