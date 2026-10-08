"""Why this run is starting, and what to start it from.

The platform hands every run a reason (``extras["resume"]``, set by the SDK):
a new item, a crash reclaim, a human's answer, or a revision. A CLI agent's
continuity is the pair (session id, workspace), so each reason maps to one
choice of prompt and session:

- reclaim       resume the session saved at run start, ask to continue
- human answer  resume the session the run paused in, with the answer verbatim
- anything else what this service did before the reason existed

Only the dictionary is read, never an SDK symbol, so an older SDK that sends
no reason leaves this service behaving exactly as it did.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from core import hitl

#: Written the moment the CLI announces its session, not when the run ends:
#: a run killed halfway is exactly the one that needs it (2026-10-06 E3 —
#: saved at completion only, a reclaim repeated finished steps 2/2).
SESSION_FILENAME = ".processgpt-session.json"

RECLAIM = "reclaim"
HUMAN_ANSWER = "human_answer"
REVISION = "revision"

#: Same wording as the SDK's standard continuation prompt. Kept here rather
#: than imported so a service on an older SDK still starts.
CONTINUE_PROMPT = (
    "직전 실행이 중단되었습니다. 지금까지의 대화와 작업 공간을 확인하고, "
    "이미 완료한 단계는 다시 하지 말고 완료되지 않은 단계부터 이어서 수행하세요."
)


def remember_session(workspace_path: Path, session_id: str) -> None:
    if not session_id or saved_session(workspace_path) == session_id:
        return
    path = workspace_path / SESSION_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"session_id": session_id}), encoding="utf-8")


def saved_session(workspace_path: Path) -> str:
    try:
        data = json.loads((workspace_path / SESSION_FILENAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    value = data.get("session_id") if isinstance(data, dict) else None
    return value.strip() if isinstance(value, str) else ""


@dataclass
class StartPlan:
    prompt: str
    #: CLI session to resume; None starts a new one.
    session_id: str | None
    #: Said to the user when continuity was lost.
    notice: str = ""


def _resume(extras: dict[str, Any]) -> tuple[str, str]:
    value = extras.get("resume")
    if not isinstance(value, dict):
        return "", ""
    return str(value.get("kind") or ""), str(value.get("answer_raw") or "")


def _last_feedback(row: dict[str, Any]) -> str:
    raw = row.get("feedback")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return ""
    if not isinstance(raw, list) or not raw:
        return ""
    last = raw[-1]
    content = last.get("content") if isinstance(last, dict) else last
    return content.strip() if isinstance(content, str) else ""


def _legacy_answer(row: dict[str, Any], extras: dict[str, Any]) -> str:
    """Keys an earlier design expected the platform to send. Nothing sends
    them today; still honoured so a caller that does keeps working."""
    for key in ("human_answer", "feedback_answer", "user_answer"):
        value = row.get(key) or extras.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def human_answer(
    workspace_path: Path, row: dict[str, Any], extras: dict[str, Any], *, workspace_exists: bool
) -> str:
    """The reply to a question this run asked earlier, verbatim, if there is one.

    An answer saved by an older screen carries no kind and reaches us as a
    revision. A pending pause in the workspace settles it: the last run stopped
    to ask, so what came back is the answer.
    """
    kind, raw = _resume(extras)
    if kind == HUMAN_ANSWER:
        return (raw or _last_feedback(row)).strip()
    if kind == REVISION and workspace_exists and hitl.recall(workspace_path) is not None:
        return (raw or _last_feedback(row)).strip()
    return _legacy_answer(row, extras)


def session_of(row: dict[str, Any], extras: dict[str, Any]) -> str | None:
    """The CLI session this conversation is already using, if any."""
    for source in (row, extras):
        value = source.get("cliagents_session_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    draft = row.get("draft") or row.get("output")
    if isinstance(draft, str) and draft.strip().startswith("{"):
        try:
            parsed = json.loads(draft)
        except json.JSONDecodeError:
            return None
        value = parsed.get("cliagents_session_id") if isinstance(parsed, dict) else None
        return value if isinstance(value, str) and value.strip() else None
    return None


def plan_start(
    workspace_path: Path,
    *,
    row: dict[str, Any],
    extras: dict[str, Any],
    workspace_exists: bool,
    fresh_prompt: Callable[[], str],
    previous_summary: str = "",
) -> StartPlan:
    """Choose the prompt and session for this run from why it is running."""
    kind, _ = _resume(extras)

    if kind == RECLAIM:
        session = saved_session(workspace_path) if workspace_exists else ""
        if session:
            # The session already holds the task. Repeating it reads as a new
            # request, and the model starts over (E3, codex E4 C1).
            return StartPlan(prompt=CONTINUE_PROMPT, session_id=session)
        return StartPlan(prompt=fresh_prompt(), session_id=session_of(row, extras))

    answer = human_answer(workspace_path, row, extras, workspace_exists=workspace_exists)
    if answer:
        plan = hitl.plan_resume(
            workspace_path,
            answer,
            workspace_exists=workspace_exists,
            previous_summary=previous_summary,
        )
        # The pause is answered. Left behind, it would make the next revision
        # look like another answer. A crash from here on is a reclaim, and the
        # session saved at run start covers it.
        hitl.clear(workspace_path)
        return StartPlan(
            prompt=plan.prompt,
            session_id=plan.session_id or None,
            notice=f"이전 실행을 이어갈 수 없어 새로 시작합니다. ({plan.reason})" if plan.restarted else "",
        )

    return StartPlan(prompt=fresh_prompt(), session_id=session_of(row, extras))
