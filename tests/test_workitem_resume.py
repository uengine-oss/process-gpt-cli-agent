"""워크아이템 재개 — 크래시 재점유와 사람 답변 뒤 CLI 세션으로 이어 간다.

스펙: infra/process-gpt/openspec/changes/workitem-resume-default/specs/cli-agent_workitem-session-resume
  RE-1.3, RE-1.4, RE-1.5, RE-2.1, RE-2.2, RE-3.1

2026-10-06 실측
- E3: 세션 ID 를 완료 때만 저장해서, 재점유되면 새 세션으로 처음부터 — 완료된 step1 2/2 재실행.
  실행 시작 직후 저장한 세션으로 --resume + 이어서 지시면 0/2.
- E3-H: 사람이 답해도 답을 읽는 키(human_answer 등)를 아무도 만들지 않아, 저장해 둔
  세션을 쓰지 않고 답 원문 대신 요약본으로 새로 시작했다.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from cliagents import ExecEvent, ExecEventKind

from core import hitl
from core import resume as resume_mod
from core.resume import plan_start, remember_session, saved_session

ORIGINAL = "1) step1 기록 2) sleep 90 후 step2 3) step3"


def _fresh(_row=None, _extras=None) -> str:
    return ORIGINAL


def _plan(ws: Path, *, resume: dict | None = None, row: dict | None = None, exists: bool = True, extras: dict | None = None):
    extras = dict(extras or {})
    if resume is not None:
        extras["resume"] = resume
    return plan_start(
        ws,
        row=row or {},
        extras=extras,
        workspace_exists=exists,
        fresh_prompt=lambda: _fresh(),
    )


def _pause(ws: Path, session: str = "SESSION-123") -> None:
    hitl.remember(ws, hitl.PendingRequest(
        run_id="todo-1", agent_id="claude-code", session_id=session,
        question="Bash 실행 권한이 필요합니다", fingerprint=hitl.fingerprint("claude-code", "Bash"),
    ))


# ---------------------------------------------------------------------------
# RE-1.4 세션 ID 는 실행 시작 직후 저장된다
# ---------------------------------------------------------------------------

def test_re_1_4_session_is_remembered_and_read_back(tmp_path):
    assert saved_session(tmp_path) == ""
    remember_session(tmp_path, "S-1")
    assert saved_session(tmp_path) == "S-1"
    remember_session(tmp_path, "")  # 빈 값은 덮어쓰지 않는다
    assert saved_session(tmp_path) == "S-1"
    (tmp_path / resume_mod.SESSION_FILENAME).write_text("깨짐", encoding="utf-8")
    assert saved_session(tmp_path) == ""


def test_re_1_4_stream_saves_session_on_the_first_event_before_the_run_ends(tmp_path, monkeypatch):
    """세션 ID 를 알리는 첫 이벤트 직후, 실행이 끝나기 전에 이미 작업 공간에 있다."""
    import executor as ex
    from core.journal import Journal
    from core.workspace import Workspace

    seen_during_run: list[str] = []

    async def fake_astream(provider, request, env=None, timeout=None):
        yield ExecEvent(kind=ExecEventKind.RUN_START, session_id="SESSION-EARLY")
        # 여기서 파드가 죽는다고 치자 — 그래도 세션은 남아 있어야 한다
        seen_during_run.append(saved_session(tmp_path))
        yield ExecEvent(kind=ExecEventKind.RESULT, text="끝", session_id="SESSION-EARLY")

    monkeypatch.setattr(ex, "astream", fake_astream)
    workspace = Workspace(path=tmp_path, run_id="todo-1")

    class _Q:
        async def enqueue_event(self, *_a, **_k):
            pass

    class _Ctx:
        metadata: dict[str, Any] = {}

    final, session, paused = asyncio.run(ex.CliAgentExecutor()._stream(
        provider=None, request=None, env=None, context=_Ctx(), event_queue=_Q(),
        task_id="todo-1", context_id="c", workspace=workspace, journal=Journal(tmp_path),
    ))
    assert seen_during_run == ["SESSION-EARLY"]
    assert session == "SESSION-EARLY"


# ---------------------------------------------------------------------------
# RE-1.3 / RE-1.5 크래시 재점유
# ---------------------------------------------------------------------------

def test_re_1_3_reclaim_resumes_the_saved_session_with_a_continue_instruction(tmp_path):
    remember_session(tmp_path, "SESSION-EARLY")
    plan = _plan(tmp_path, resume={"kind": "reclaim", "attempt": 2, "key": "todo-1", "answer_raw": ""})
    assert plan.session_id == "SESSION-EARLY"
    assert "중단" in plan.prompt and "다시 하지" in plan.prompt and "이어서" in plan.prompt
    assert ORIGINAL not in plan.prompt  # 세션에 이미 있는 지시를 다시 넣으면 처음부터 한다(E3)
    assert plan.notice == ""


def test_re_1_5_reclaim_without_a_saved_session_starts_over_with_the_original_task(tmp_path):
    plan = _plan(tmp_path, resume={"kind": "reclaim", "attempt": 2})
    assert plan.session_id is None
    assert plan.prompt == ORIGINAL


def test_re_1_5_reclaim_after_the_workspace_was_swept_starts_over(tmp_path):
    remember_session(tmp_path, "SESSION-EARLY")
    plan = _plan(tmp_path, resume={"kind": "reclaim", "attempt": 2}, exists=False)
    assert plan.session_id is None
    assert plan.prompt == ORIGINAL


# ---------------------------------------------------------------------------
# RE-2 사람 답변
# ---------------------------------------------------------------------------

def test_re_2_1_human_answer_resumes_the_paused_session_with_the_raw_answer(tmp_path):
    """E3-H 와 같은 입력: pending 파일 + SDK 가 실어 준 원문 답 + 요약본."""
    _pause(tmp_path)
    answer = "허용합니다. 계속 진행하세요"
    plan = _plan(
        tmp_path,
        resume={"kind": "human_answer", "attempt": 1, "key": "todo-1", "answer_raw": answer},
        extras={"summarized_feedback": "## 이전 결과에 대한 피드백 — 요약본"},
        row={"feedback": [{"content": answer, "kind": "human_answer"}], "draft": None},
    )
    assert plan.session_id == "SESSION-123"
    assert answer in plan.prompt
    assert "요약본" not in plan.prompt
    assert "다시 하지" in plan.prompt
    # 답을 쓴 뒤에는 pending 이 남지 않는다 — 다음 반려를 답변으로 오인하지 않게
    assert hitl.recall(tmp_path) is None


def test_re_2_1_answer_falls_back_to_the_last_feedback_entry(tmp_path):
    _pause(tmp_path)
    plan = _plan(
        tmp_path,
        resume={"kind": "human_answer", "attempt": 1, "answer_raw": ""},
        row={"feedback": [{"content": "B안으로", "kind": "human_answer"}]},
    )
    assert plan.session_id == "SESSION-123"
    assert "B안으로" in plan.prompt


def test_re_2_2_kindless_answer_with_a_pending_pause_is_a_human_answer(tmp_path):
    """구버전 화면은 kind 를 남기지 않는다 → SDK 는 revision. pending 이 있으면 답이다."""
    _pause(tmp_path)
    plan = _plan(tmp_path, resume={"kind": "revision", "attempt": 1, "answer_raw": "허용"})
    assert plan.session_id == "SESSION-123"
    assert "허용" in plan.prompt


def test_human_answer_without_a_pause_says_it_restarted(tmp_path):
    plan = _plan(tmp_path, resume={"kind": "human_answer", "answer_raw": "허용"})
    assert plan.session_id is None
    assert "허용" in plan.prompt
    assert plan.notice  # 이어 갈 수 없음을 사용자에게 알린다


# ---------------------------------------------------------------------------
# RE-3.1 신호가 없거나 신규·반려면 이전 동작
# ---------------------------------------------------------------------------

def test_re_3_1_fresh_and_legacy_keep_the_previous_prompt(tmp_path):
    remember_session(tmp_path, "SESSION-EARLY")  # 신규 실행은 남은 세션을 쓰지 않는다
    for resume in (None, {"kind": "fresh", "attempt": 1}):
        plan = _plan(tmp_path, resume=resume)
        assert plan.prompt == ORIGINAL
        assert plan.session_id is None


def test_re_3_1_revision_keeps_the_previous_session_rule(tmp_path):
    plan = _plan(
        tmp_path,
        resume={"kind": "revision", "answer_raw": "표로 다시"},
        row={"draft": '{"cliagents_session_id": "S-DONE"}'},
    )
    assert plan.prompt == ORIGINAL
    assert plan.session_id == "S-DONE"


def test_legacy_answer_keys_still_resume(tmp_path):
    _pause(tmp_path)
    plan = _plan(tmp_path, row={"human_answer": "예"})
    assert plan.session_id == "SESSION-123"
    assert "예" in plan.prompt


def test_executor_is_wired_to_the_start_plan():
    """선택·스킬·런타임 준비까지 띄우지 않고, 실행이 시작 계획을 거치는지만 지킨다."""
    import inspect

    import executor as ex

    src = inspect.getsource(ex.CliAgentExecutor._run)
    assert "resume.plan_start(" in src
    assert "resume_session = start.session_id" in src
    assert "_human_answer" not in src  # E3-H 결함: 아무도 만들지 않는 키만 읽던 경로


# ---------------------------------------------------------------------------
# RE-2.0 권한 때문에 막히면 사람에게 묻고 멈춘다 (2026-10-07 kind e2e 에서 발견)
#
# 배포 이미지의 Claude Code 2.1.216 은 거부를 "This command requires approval" 로 돌려주는데
# cliagents 의 거부 표식에 이 문구가 없어 권한 요청으로 인식되지 않았다. 게다가 모델은 거부 뒤
# 설명 한 줄을 남기므로 "결과 없음일 때만 멈춤" 규칙에도 걸리지 않아, 작업은 폼 계약 위반으로
# FAILED 가 됐다 — 사람에게 물을 기회가 없었다.
# ---------------------------------------------------------------------------

def test_re_2_0_requires_approval_tool_error_is_a_permission_refusal():
    refused = ExecEvent(kind=ExecEventKind.TOOL_END, tool="Bash", text="This command requires approval", is_error=True)
    assert hitl.permission_refusal(refused) == "This command requires approval"
    assert hitl.permission_refusal(ExecEvent(kind=ExecEventKind.PERMISSION_REQUEST, text="권한")) == "권한"
    # 일반 도구 오류는 거부가 아니다
    assert hitl.permission_refusal(ExecEvent(kind=ExecEventKind.TOOL_END, tool="Bash", text="exit 1", is_error=True)) is None
    assert hitl.permission_refusal(ExecEvent(kind=ExecEventKind.TOOL_END, tool="Bash", text="This command requires approval", is_error=False)) is None


def test_re_2_0_refusal_without_a_usable_result_pauses():
    # 거부 + 설명만 있고 폼 결과가 비었다 → 사람이 정해야 한다
    assert hitl.should_pause(refused=True, final_text="권한 때문에 실행하지 못했습니다", contract_met=False) is True
    assert hitl.should_pause(refused=True, final_text="", contract_met=True) is True
    # 거부를 우회해 결과를 냈다 → 저장(이전 동작)
    assert hitl.should_pause(refused=True, final_text='{"result": "ok"}', contract_met=True) is False
    # 거부가 없으면 멈추지 않는다
    assert hitl.should_pause(refused=False, final_text="", contract_met=False) is False


def test_re_2_0_stream_reports_the_requires_approval_refusal_as_a_pause(tmp_path, monkeypatch):
    import executor as ex
    from core.journal import Journal
    from core.workspace import Workspace

    async def fake_astream(provider, request, env=None, timeout=None):
        yield ExecEvent(kind=ExecEventKind.RUN_START, session_id="S-1")
        yield ExecEvent(kind=ExecEventKind.TOOL_START, tool="Bash", session_id="S-1")
        yield ExecEvent(kind=ExecEventKind.TOOL_END, tool="Bash", text="This command requires approval", is_error=True, session_id="S-1")
        yield ExecEvent(kind=ExecEventKind.RESULT, text="권한 때문에 실행하지 못했습니다.", session_id="S-1")

    monkeypatch.setattr(ex, "astream", fake_astream)

    class _Q:
        async def enqueue_event(self, *_a, **_k):
            pass

    class _Ctx:
        metadata: dict[str, Any] = {}

    _final, session, paused = asyncio.run(ex.CliAgentExecutor()._stream(
        provider=None, request=None, env=None, context=_Ctx(), event_queue=_Q(),
        task_id="t", context_id="c", workspace=Workspace(path=tmp_path, run_id="t"), journal=Journal(tmp_path),
    ))
    assert paused == "This command requires approval"
    assert session == "S-1"


def test_rs_6_1_pause_is_recorded_as_a_human_asked_event_the_screen_can_answer(tmp_path):
    """화면의 질문 카드는 events.event_type='human_asked' 로만 그려지고 data.question 을 보인다.
    'human_input_required' 는 저장소 enum 에 없어 이벤트가 저장되지도 않았다(2026-10-07 e2e)."""
    import json as _json

    import executor as ex
    from core.workspace import Workspace

    sent: list = []

    class _Q:
        async def enqueue_event(self, event):
            sent.append(event)

    class _Ctx:
        metadata: dict[str, Any] = {}

    asyncio.run(ex.CliAgentExecutor()._pause(
        context=_Ctx(), event_queue=_Q(), task_id="t", context_id="c",
        workspace=Workspace(path=tmp_path, run_id="t"), agent_id="claude-code",
        session_id="S-1", question="This command requires approval",
    ))
    # 질문 앞에서 이번 구간의 카드를 닫고(본문 없음 — 채택할 결과가 아니다), 질문 카드를 연다
    close, event = sent
    assert dict(close.metadata.items())["event_type"] == "task_completed"
    assert close.status.message.parts[0].text == "{}"  # 빈 문자열이면 SDK 가 메시지 원본을 저장했다
    meta = dict(event.metadata.items())
    assert meta["event_type"] == "human_asked"
    # 질문 카드는 작업 카드와 다른 id — 같으면 화면이 작업 카드의 도구 목록을 질문 카드에도 그린다
    assert meta["job_id"] == f"{dict(close.metadata.items())['job_id']}:ask"
    body = _json.loads(event.status.message.parts[0].text)
    assert body["question"] == "This command requires approval"
    assert body["type"] == "text"
    assert hitl.recall(tmp_path).session_id == "S-1"
