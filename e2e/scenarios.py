"""What each E2E scenario seeds and what it must be true of afterwards.

Kept as data, separate from the runner, so the acceptance criteria are readable
without following execution code — and so the same definitions can drive both
the live suite and the offline checks below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Scenario:
    id: str
    description: str
    #: The todolist row to seed, minus ids the runner fills in.
    work_item: dict[str, Any]
    #: Human-readable acceptance criteria, asserted by the runner.
    expects: list[str] = field(default_factory=list)
    #: True when the scenario needs a real model call (and so real credits).
    spends_tokens: bool = True


BASIC = Scenario(
    id="basic",
    description="A work item is claimed, run on a CLI agent, and stored",
    work_item={
        "agent_orch": "cliagents",
        "agent_mode": "COMPLETE",
        "activity_name": "요약 보고서 작성",
        "query": "report.md 파일을 만들고 '완료'라고만 답하세요.",
        "agent_config": {"agent_cli": "claude-code", "permission": "workspace_write"},
    },
    expects=[
        "the work item reaches DONE",
        "a result is stored on the work item",
        "report.md exists in the run's workspace",
        "the stored result carries a session id, so the run can be continued",
    ],
)

BOTH_CLIS = Scenario(
    id="both-clis",
    description="The same work item on both CLIs produces the same contract",
    work_item={
        "agent_orch": "cliagents",
        "agent_mode": "COMPLETE",
        "activity_name": "동일 업무 교차 검증",
        "query": "note.txt 파일에 'hi' 를 쓰고 '완료'라고만 답하세요.",
        "agent_config": {"permission": "workspace_write"},  # cli filled per run
    },
    expects=[
        "both runs emit run_start, tool_end and a result event",
        "both runs produce note.txt",
        "both store a result and a session id",
        "neither run's events require the caller to know which CLI ran",
    ],
)

HITL = Scenario(
    id="hitl",
    description="A run that needs permission pauses, and answering resumes it",
    work_item={
        "agent_orch": "cliagents",
        "agent_mode": "DRAFT",
        "activity_name": "승인이 필요한 작업",
        # read_only guarantees the agent hits a wall it cannot work around.
        "query": "output.txt 파일을 생성하세요.",
        "agent_config": {"agent_cli": "claude-code", "permission": "read_only"},
    },
    expects=[
        "the work item goes to an input-required state, not FAILED",
        "a pending request is recorded in the workspace and survives a restart",
        "the same question asked twice notifies once",
        "answering resumes the recorded session rather than starting a new one",
    ],
)

MISSING_CLI = Scenario(
    id="missing-cli",
    description="A CLI that is not installed fails loudly and runs nothing else",
    work_item={
        "agent_orch": "cliagents",
        "agent_mode": "COMPLETE",
        "activity_name": "미설치 CLI 실행 시도",
        "query": "무엇이든 해보세요.",
        "agent_config": {"agent_cli": "gemini-cli"},
    },
    expects=[
        "the work item fails",
        "the failure names the CLI and how to install it",
        "no other CLI process was started",
    ],
    # No model is ever reached, so this one is free and should run in CI.
    spends_tokens=False,
)

DRAFT_VS_COMPLETE = Scenario(
    id="draft-vs-complete",
    description="agent_mode decides whether the run may close the work item",
    work_item={
        "agent_orch": "cliagents",
        "agent_mode": "DRAFT",
        "activity_name": "초안 모드 확인",
        "query": "'초안입니다' 라고만 답하세요.",
        "agent_config": {"agent_cli": "claude-code", "permission": "read_only"},
    },
    expects=[
        "in DRAFT the result is stored but the item is not closed",
        "in COMPLETE the same run closes the item",
    ],
)

ALL = [BASIC, BOTH_CLIS, HITL, MISSING_CLI, DRAFT_VS_COMPLETE]


def by_id(scenario_id: str) -> Scenario:
    for scenario in ALL:
        if scenario.id == scenario_id:
            return scenario
    raise KeyError(f"unknown scenario {scenario_id!r}; have {[s.id for s in ALL]}")
