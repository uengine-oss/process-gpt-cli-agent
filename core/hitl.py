"""Pausing for a human, and picking up where the pause happened.

The deepagents orchestration suspends a graph and reloads it from a
checkpointer. A CLI agent has no graph to suspend: the process ends. What
survives is the pair (session id, workspace) — enough for the CLI to resume its
own conversation with its own history and its own files.

So a pause is written down, not held in memory. The service can restart, the
pod can move, and the answer that arrives tomorrow still lands in the run that
asked the question.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: Lives inside the workspace, so a run's pause travels with its files and is
#: swept by the same retention rule. Nothing to garbage-collect separately.
_STATE_FILENAME = ".processgpt-pending.json"


@dataclass
class PendingRequest:
    """A question the run is waiting on."""

    run_id: str
    agent_id: str
    session_id: str
    #: What the agent wanted to do, in the user's terms.
    question: str
    tool: str = ""
    asked_at: float = field(default_factory=time.time)
    #: Fingerprint of the question, so the same block asked twice does not
    #: produce two notifications for one decision.
    fingerprint: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def fingerprint(tool: str, question: str) -> str:
    """Identity of a request, for duplicate suppression.

    Tool plus the first line: the tail of a permission message often carries a
    retry count or a timestamp, and matching on the whole string would make
    every repeat look new.
    """
    head = (question or "").strip().splitlines()
    return f"{tool}::{head[0][:200] if head else ''}"


def state_path(workspace_path: Path) -> Path:
    return workspace_path / _STATE_FILENAME


def remember(workspace_path: Path, request: PendingRequest) -> bool:
    """Record a pause. Returns False when this question is already pending.

    The caller uses the return value to decide whether to notify anyone, which
    is the whole of duplicate suppression.
    """
    existing = recall(workspace_path)
    if existing and existing.fingerprint == request.fingerprint:
        return False

    path = state_path(workspace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(request.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def recall(workspace_path: Path) -> PendingRequest | None:
    path = state_path(workspace_path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        return PendingRequest(**data)
    except TypeError:
        # Written by an older shape of this service. Treat as no pause rather
        # than crash a run that is otherwise fine.
        return None


def clear(workspace_path: Path) -> None:
    try:
        state_path(workspace_path).unlink()
    except OSError:
        pass


@dataclass
class ResumePlan:
    """How to continue after a human answered."""

    #: Session to resume, empty when the run has to start over.
    session_id: str
    prompt: str
    #: True when continuity was lost and the user must be told so.
    restarted: bool = False
    reason: str = ""


def plan_resume(
    workspace_path: Path,
    answer: str,
    *,
    workspace_exists: bool,
    previous_summary: str = "",
) -> ResumePlan:
    """Decide how to act on ``answer``.

    Silently starting from scratch is the failure this guards against: the user
    approved one step and would get a whole run repeated, with the side effects
    that implies.
    """
    pending = recall(workspace_path) if workspace_exists else None

    if pending and pending.session_id:
        return ResumePlan(
            session_id=pending.session_id,
            prompt=(
                f"담당자 응답: {answer}\n\n"
                "이 응답을 반영해 중단된 지점부터 작업을 계속하세요. "
                "이미 완료한 작업은 다시 하지 마세요."
            ),
        )

    reason = (
        "이전 작업 공간이 정리되어 이어갈 수 없습니다."
        if not workspace_exists
        else "이전 세션 정보를 찾을 수 없어 이어갈 수 없습니다."
    )
    prompt = f"담당자 응답: {answer}\n"
    if previous_summary:
        prompt += f"\n## 이전 진행 요약\n{previous_summary}\n"
    prompt += "\n위 맥락을 참고해 작업을 새로 진행하세요."

    return ResumePlan(session_id="", prompt=prompt, restarted=True, reason=reason)
