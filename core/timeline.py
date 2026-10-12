"""Exec events → the work-item timeline the screen already draws.

The work-item monitor (vue3 ``agentEventTimeline``) builds its cards from rows
of the ``events`` table, and it already knows how to draw a tool call
(``tool_usage_started`` / ``tool_usage_finished``, paired by ``tool_name``), a
plan (a ``write_todos`` start whose ``args.todos`` is ``[{content, status}]``),
a question (``human_asked``) and a failure (``error``). deepagents writes those
rows; until now cli-agent wrote none of them, so a running CLI agent showed a
spinner and nothing else.

:mod:`core.events` is the *chat* stream (SSE chunks). This module is the
*stored* timeline: what survives a refresh and a crash. Both are fed from the
same exec events.

Three rules:

* Only shapes the screen already reads. No cli-agent-only card.
* A tool that started is eventually finished — by its own end event, by the
  refusal that replaced it, or, after a crash, by the run that took over
  (:class:`OpenTools`). Otherwise the screen spins on it forever.
* Payloads are cut short. The table is a timeline, not a transcript.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cliagents import ExecEvent, ExecEventKind

#: Kept next to the session file: the tools this work item's runs started and
#: have not finished yet. A run that dies leaves its entries behind.
OPEN_TOOLS_FILENAME = ".processgpt-open-tools.json"

#: The plan list the screen draws (``getLatestTodos``).
PLAN_TOOL = "write_todos"
#: Claude Code's plan tool. Its input is already ``{todos: [{content, status}]}``.
CLAUDE_PLAN_TOOL = "TodoWrite"

#: ``events.crew_type`` of this agent's rows — the work item's ``agent_orch``,
#: as deepagents writes ``deepagents``. The work card is drawn from it.
CREW_TYPE = "cliagents"
#: The final-result card. The screen offers "채택" (adopt into the form) only on
#: cards of this type, so it is kept apart from the work card.
RESULT_CREW_TYPE = "result"

#: Long enough to read, short enough not to bloat the events table.
MAX_FIELD_CHARS = 4000

INTERRUPTED_RESULT = "직전 실행이 중단되어 결과를 받지 못했습니다. 이어받은 실행이 계속 진행합니다."

#: Claude Code's subagent tools ("Task" before 2.x, "Agent" after). A call to
#: one is another agent doing a piece of the work, not a tool result: it gets
#: its own card, between the work card and the final result.
SUBAGENT_TOOLS = ("Agent", "Task")


@dataclass(frozen=True)
class TimelineEvent:
    """One row for the events table: its ``event_type`` and ``data``."""

    event_type: str
    data: dict[str, Any] = field(default_factory=dict)
    #: The card this row belongs to, as a suffix of the run's card id
    #: (``<job>:<card>``). ``None``: the run's own work card.
    card: str | None = None

    def text(self) -> str:
        return json.dumps(self.data, ensure_ascii=False, default=str)


def translate(event: ExecEvent) -> list[TimelineEvent]:
    """Map one exec event onto zero or more timeline rows."""
    kind = event.kind

    if kind is ExecEventKind.TOOL_START and event.tool in SUBAGENT_TOOLS:
        return [_subagent_started(event.tool_use_id, event.tool_input)]

    if kind is ExecEventKind.TOOL_END and event.tool in SUBAGENT_TOOLS:
        output = event.tool_output if event.tool_output is not None else event.text
        return [_subagent_finished(event.tool_use_id, output, is_error=event.is_error)]

    if kind is ExecEventKind.TOOL_START:
        name = _tool_name(event.tool)
        args = event.tool_input
        if name == PLAN_TOOL:
            args = {"todos": _todos((args or {}).get("todos") if isinstance(args, dict) else None)}
        return [
            TimelineEvent(
                "tool_usage_started",
                {
                    "tool": name,
                    "tool_name": name,
                    "tool_use_id": event.tool_use_id,
                    "args": _cut(args),
                    "query": _query(name, args),
                },
            )
        ]

    if kind is ExecEventKind.TOOL_END:
        name = _tool_name(event.tool)
        output = event.tool_output if event.tool_output is not None else event.text
        return [_finished(name, event.tool_use_id, output, is_error=event.is_error)]

    if kind is ExecEventKind.PERMISSION_REQUEST:
        # The CLI reports the refusal *instead of* the tool's end. Close the
        # tool with the refusal, so the card shows why it stopped.
        name = _tool_name(event.tool)
        return [_finished(name, event.tool_use_id, f"권한 거부: {event.text}", is_error=True)]

    if kind is ExecEventKind.PLAN:
        # Codex reports its plan as a list of ``{text, completed}``; the screen
        # reads the latest ``write_todos`` start. Every update is a new start
        # and its own end, so no plan entry is ever left open.
        todos = _todos(event.tool_output)
        if not todos:
            return []
        return [
            TimelineEvent("tool_usage_started", {"tool": PLAN_TOOL, "tool_name": PLAN_TOOL, "args": {"todos": todos}}),
            TimelineEvent("tool_usage_finished", {"tool": PLAN_TOOL, "tool_name": PLAN_TOOL, "result": ""}),
        ]

    return []


def failure_data(message: str) -> dict[str, Any]:
    """The body of a failed run, in the keys the error card reads.

    The screen shows ``friendly`` (then ``message``) on its error card; a body
    with only ``error`` showed "오류가 발생했습니다" and nothing more. The card
    title and description come from ``name`` and ``goal`` — the same values the
    SDK writes for its own error events, so every failure card reads alike.
    """
    return {
        "name": "시스템 오류 알림",
        "role": "CLI 코딩 에이전트",
        "goal": "오류 원인과 대처 안내를 전달합니다.",
        "agent_profile": "/images/chat-icon.png",
        "friendly": message,
        "message": message,
        "error": message,
    }


def job_id(task_id: str, row: dict[str, Any] | None) -> str:
    """The card this run belongs to.

    One card per stretch of work between people: a run that pauses for an
    answer, or whose draft is sent back, ends its card, and the run after the
    answer opens the next one — so the screen reads "work → question → work".
    A crash reclaim does not touch ``feedback``, so it lands on the same card
    and its tools continue the list where the dead run stopped.

    The first stretch keeps the bare task id, which is what earlier versions
    wrote; items already on screen keep their card.
    """
    entries = _feedback_entries((row or {}).get("feedback"))
    return task_id if not entries else f"{task_id}:{len(entries)}"


def result_job_id(job: str) -> str:
    """The final-result card that follows a work card."""
    return f"{job}:result"


def description_of(row: dict[str, Any] | None) -> str:
    """The activity's Description, for the work card.

    ``todolist.query`` is the whole prompt completion built — Description,
    Instruction and the input JSON — and showing it on the card dumped all of
    that on screen. ``todolist.description`` holds the Description alone; when it
    is empty, take the ``[Description]`` section of the prompt.
    """
    row = row or {}
    text = str(row.get("description") or "").strip()
    if text:
        return text
    query = str(row.get("query") or "")
    marker = "[Description]"
    start = query.find(marker)
    if start < 0:
        return ""
    lines: list[str] = []
    for line in query[start + len(marker):].splitlines():
        if line.strip().startswith("[") and line.strip().endswith("]") and lines:
            break
        lines.append(line)
    return "\n".join(lines).strip()


class OpenTools:
    """Tools started and not yet finished, persisted in the workspace.

    In memory this would die with the process — exactly the case it is for.
    """

    def __init__(self, workspace: Path):
        self.path = Path(workspace) / OPEN_TOOLS_FILENAME

    def load(self) -> dict[str, str]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return {}
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}

    def _save(self, entries: dict[str, str]) -> None:
        if not entries:
            self.path.unlink(missing_ok=True)
            return
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def track(self, rows: list[TimelineEvent]) -> None:
        """Note starts and ends as they are recorded."""
        entries = self.load()
        before = dict(entries)
        for row in rows:
            key = row.data.get("tool_use_id")
            if not key:
                continue
            if row.event_type in ("tool_usage_started", "task_started"):
                # A subagent card is remembered by its card, so it is closed as a card.
                entries[str(key)] = f"card:{row.card}" if row.card else str(row.data.get("tool_name") or "tool")
            elif row.event_type in ("tool_usage_finished", "task_completed"):
                entries.pop(str(key), None)
        if entries != before:
            self._save(entries)

    def close_interrupted(self) -> list[TimelineEvent]:
        """End what a dead run left open, and forget it.

        Called when a run starts. A fresh workspace has nothing here; a run
        that took over from a crashed one finds the tools that were running
        when the process died.
        """
        entries = self.load()
        self._save({})
        return [
            _subagent_finished(key, INTERRUPTED_RESULT, is_error=True)
            if name.startswith("card:")
            else _finished(name, key, INTERRUPTED_RESULT, is_error=True)
            for key, name in entries.items()
        ]


# --- helpers ----------------------------------------------------------------


def _finished(name: str, tool_use_id: str | None, output: Any, *, is_error: bool) -> TimelineEvent:
    return TimelineEvent(
        "tool_usage_finished",
        {
            "tool": name,
            "tool_name": name,
            "tool_use_id": tool_use_id,
            "result": _cut(_as_text(output)),
            "is_error": bool(is_error),
        },
    )


def _subagent_card(tool_use_id: str | None) -> str:
    return f"agent:{tool_use_id or 'unknown'}"


def _subagent_started(tool_use_id: str | None, args: Any) -> TimelineEvent:
    """A subagent's own card: what it was asked to do, and who."""
    args = args if isinstance(args, dict) else {}
    agent = str(args.get("subagent_type") or "").strip()
    what = str(args.get("description") or "").strip()
    return TimelineEvent(
        "task_started",
        {
            "name": what or agent or "서브에이전트 작업",
            "role": "서브에이전트",
            "goal": what,
            "task_description": f"서브에이전트 {agent} 에게 위임" if agent else "서브에이전트에게 위임",
            "tool_use_id": tool_use_id,
        },
        card=_subagent_card(tool_use_id),
    )


def _subagent_finished(tool_use_id: str | None, output: Any, *, is_error: bool) -> TimelineEvent:
    text = _cut(readable(_as_text(output)))
    return TimelineEvent(
        "task_completed",
        {"result": f"오류: {text}" if is_error and text else text, "tool_use_id": tool_use_id},
        card=_subagent_card(tool_use_id),
    )


#: What Claude Code appends to a subagent's answer for its own use — the id to
#: resume that subagent with, and token accounting. Not for a person.
_CLAUDE_TRAILER = re.compile(r"(?:^|\n)agentId: \S+[^\n]*|<usage>.*?</usage>", flags=re.S)


def readable(text: str) -> str:
    """A subagent's answer as markdown a person can read on the card.

    Subagents often answer with JSON — the parent asked for the form — wrapped
    in a ```json fence. Shown as is, the card was one long line of quotes and
    literal ``\\n``. A JSON object becomes one section per key (the value's own
    markdown kept); anything else is returned without the fence.
    """
    body = _CLAUDE_TRAILER.sub("", text or "").strip()
    fenced = re.fullmatch(r"```[a-zA-Z0-9_-]*\s*\n(.*?)\n?```", body, flags=re.S)
    if fenced:
        body = fenced.group(1).strip()
    try:
        parsed = json.loads(body)
    except ValueError:
        return body
    if not isinstance(parsed, dict) or not parsed:
        return body
    sections = []
    for key, value in parsed.items():
        if isinstance(value, str):
            shown = value.strip()
        else:
            shown = "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```"
        sections.append(f"**{key}**\n\n{shown}")
    return "\n\n".join(sections)


def _tool_name(tool: str | None) -> str:
    name = (tool or "").strip() or "tool"
    return PLAN_TOOL if name == CLAUDE_PLAN_TOOL else name


def _todos(items: Any) -> list[dict[str, str]]:
    """``[{content, status}]`` from Claude's todos or Codex's plan items."""
    todos: list[dict[str, str]] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        content = item.get("content") or item.get("text") or item.get("activeForm") or ""
        status = item.get("status")
        if status not in ("pending", "in_progress", "completed"):
            status = "completed" if item.get("completed") else "pending"
        if content:
            todos.append({"content": str(content), "status": status})
    return todos


#: Which argument says what a tool is doing, for the one-line status.
_QUERY_KEYS = ("description", "command", "file_path", "path", "pattern", "url", "query", "prompt")


def _query(name: str, args: Any) -> str | None:
    if name == PLAN_TOOL or not isinstance(args, dict):
        return None
    for key in _QUERY_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            line = value.strip().splitlines()[0]
            return line if len(line) <= 120 else line[:117] + "..."
    return None


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        # Claude Code tool results: [{type: text, text: ...}, ...]
        parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in value]
        if all(isinstance(p, str) for p in parts) and any(parts):
            return "\n".join(p for p in parts if p)
    return json.dumps(value, ensure_ascii=False, default=str)


def _cut(value: Any) -> Any:
    if isinstance(value, str):
        return value if len(value) <= MAX_FIELD_CHARS else value[:MAX_FIELD_CHARS] + "\n… (생략됨)"
    if isinstance(value, dict):
        text = json.dumps(value, ensure_ascii=False, default=str)
        if len(text) <= MAX_FIELD_CHARS:
            return value
        return {k: _cut(v) if isinstance(v, str) else v for k, v in value.items()}
    return value


def _feedback_entries(raw: Any) -> list:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return []
    return raw if isinstance(raw, list) else []
