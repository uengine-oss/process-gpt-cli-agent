"""업무 화면 타임라인 — cli-agent 실행 과정을 events 테이블에 남긴다.

화면(vue3 agentEventTimeline)은 events 행으로 카드를 그린다. 도구 호출은
tool_usage_started/finished(tool_name 으로 짝), 계획은 write_todos 의 args.todos,
질문은 human_asked, 실패는 error 의 friendly. deepagents 는 이 행들을 남겼지만
cli-agent 는 시작·완료·질문만 남겨, 실행 중에는 빈 카드만 보였다(2026-10-08).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from cliagents import ExecEvent, ExecEventKind

from core import timeline
from core.timeline import OpenTools, job_id, translate


def _bash_start(cmd="ls -la", tool_use_id="T1"):
    return ExecEvent(kind=ExecEventKind.TOOL_START, tool="Bash", tool_input={"command": cmd}, tool_use_id=tool_use_id)


def _bash_end(out="a.txt", tool_use_id="T1", is_error=False):
    return ExecEvent(kind=ExecEventKind.TOOL_END, tool="Bash",
                     tool_output=[{"type": "text", "text": out}], tool_use_id=tool_use_id, is_error=is_error)


# ---------------------------------------------------------------------------
# 이벤트 모양 — 화면이 이미 읽는 것만
# ---------------------------------------------------------------------------

def test_tool_start_is_a_tool_usage_started_with_name_args_and_a_one_line_query():
    [row] = translate(_bash_start("python make.py\n# 두 번째 줄"))
    assert row.event_type == "tool_usage_started"
    assert row.data["tool_name"] == "Bash"
    assert row.data["args"] == {"command": "python make.py\n# 두 번째 줄"}
    assert row.data["query"] == "python make.py"  # "Bash 도구 사용 중: python make.py"


def test_tool_end_carries_the_result_text_the_card_shows():
    [row] = translate(_bash_end("결과 줄"))
    assert row.event_type == "tool_usage_finished"
    assert row.data["tool_name"] == "Bash"
    assert row.data["result"] == "결과 줄"  # 화면은 data.result 를 "결과 보기" 로 펼친다
    assert row.data["is_error"] is False


def test_claude_todowrite_becomes_the_plan_list():
    todos = [{"content": "자료 수집", "status": "completed", "activeForm": "자료 수집 중"},
             {"content": "초안 작성", "status": "in_progress", "activeForm": "초안 작성 중"}]
    [row] = translate(ExecEvent(kind=ExecEventKind.TOOL_START, tool="TodoWrite",
                                tool_input={"todos": todos}, tool_use_id="P1"))
    assert row.data["tool_name"] == "write_todos"
    assert row.data["args"]["todos"] == [
        {"content": "자료 수집", "status": "completed"},
        {"content": "초안 작성", "status": "in_progress"},
    ]


def test_codex_plan_becomes_the_plan_list_and_never_stays_open():
    rows = translate(ExecEvent(kind=ExecEventKind.PLAN,
                               tool_output=[{"text": "읽기", "completed": True}, {"text": "쓰기", "completed": False}]))
    assert [r.event_type for r in rows] == ["tool_usage_started", "tool_usage_finished"]
    assert rows[0].data["args"]["todos"] == [{"content": "읽기", "status": "completed"},
                                            {"content": "쓰기", "status": "pending"}]


def test_permission_refusal_closes_the_tool_it_replaced():
    [row] = translate(ExecEvent(kind=ExecEventKind.PERMISSION_REQUEST, tool="Bash",
                                text="This command requires approval", tool_use_id="T1", is_error=True))
    assert row.event_type == "tool_usage_finished"
    assert row.data["result"].startswith("권한 거부")
    assert row.data["tool_use_id"] == "T1"


def test_text_and_usage_are_not_stored():
    assert translate(ExecEvent(kind=ExecEventKind.ASSISTANT_TEXT, text="생각 중")) == []
    assert translate(ExecEvent(kind=ExecEventKind.USAGE, usage={"x": 1})) == []


def test_long_results_are_cut():
    [row] = translate(_bash_end("x" * (timeline.MAX_FIELD_CHARS + 500)))
    assert len(row.data["result"]) < timeline.MAX_FIELD_CHARS + 50


def test_failure_body_has_the_key_the_error_card_shows():
    body = timeline.failure_data("결과가 형식과 맞지 않습니다")
    assert body["friendly"] == "결과가 형식과 맞지 않습니다"  # 없으면 "오류가 발생했습니다" 만 보였다


# ---------------------------------------------------------------------------
# 카드 — 사람 답변·반려마다 새 카드, 크래시 재점유는 같은 카드
# ---------------------------------------------------------------------------

def test_job_id_is_the_task_until_a_person_answers_then_one_card_per_stretch():
    assert job_id("todo-1", {}) == "todo-1"
    assert job_id("todo-1", {"feedback": None}) == "todo-1"
    assert job_id("todo-1", {"feedback": [{"content": "답", "kind": "human_answer"}]}) == "todo-1:1"
    assert job_id("todo-1", {"feedback": json.dumps([{}, {}])}) == "todo-1:2"


# ---------------------------------------------------------------------------
# 크래시 — 죽은 실행이 남긴 도구는 이어받은 실행이 닫는다
# ---------------------------------------------------------------------------

def test_open_tools_survive_the_process_and_are_closed_by_the_next_run(tmp_path):
    OpenTools(tmp_path).track(translate(_bash_start(tool_use_id="T9")))
    # 여기서 프로세스가 죽었다 — 새 프로세스가 같은 작업 공간을 연다
    rows = OpenTools(tmp_path).close_interrupted()
    assert [(r.event_type, r.data["tool_name"], r.data["tool_use_id"]) for r in rows] == [
        ("tool_usage_finished", "Bash", "T9")
    ]
    assert rows[0].data["result"] == timeline.INTERRUPTED_RESULT
    assert OpenTools(tmp_path).close_interrupted() == []  # 한 번만


def test_finished_tools_are_not_left_open(tmp_path):
    tools = OpenTools(tmp_path)
    tools.track(translate(_bash_start()))
    tools.track(translate(_bash_end()))
    assert tools.load() == {}
    assert not (tmp_path / timeline.OPEN_TOOLS_FILENAME).exists()


# ---------------------------------------------------------------------------
# executor 연결 — 실제 _stream 이 이벤트 큐로 보낸다
# ---------------------------------------------------------------------------

class _Q:
    def __init__(self):
        self.sent: list = []

    async def enqueue_event(self, event):
        self.sent.append(event)

    def rows(self):
        out = []
        for e in self.sent:
            meta = dict(e.metadata.items())
            text = e.status.message.parts[0].text
            out.append((meta["event_type"], meta["job_id"], json.loads(text) if text else None))
        return out


class _Ctx:
    metadata: dict[str, Any] = {}


def _run_stream(monkeypatch, tmp_path, events, *, record=True, job="todo-1", errors=None, die_after=None):
    import executor as ex
    from core.journal import Journal
    from core.workspace import Workspace

    async def fake_astream(provider, request, env=None, timeout=None):
        for i, e in enumerate(events):
            if die_after is not None and i == die_after:
                raise RuntimeError("프로세스가 죽었다")
            yield e

    monkeypatch.setattr(ex, "astream", fake_astream)
    q = _Q()

    async def go():
        ex._JOB_ID.set(job)
        return await ex.CliAgentExecutor()._stream(
            provider=None, request=None, env=None, context=_Ctx(), event_queue=q,
            task_id="todo-1", context_id="c", workspace=Workspace(path=tmp_path, run_id="todo-1"),
            journal=Journal(tmp_path), record=record, errors=errors,
        )

    try:
        asyncio.run(go())
    except RuntimeError:
        pass
    return q


def test_stream_stores_tool_calls_on_this_runs_card(monkeypatch, tmp_path):
    q = _run_stream(monkeypatch, tmp_path, [
        ExecEvent(kind=ExecEventKind.RUN_START, session_id="S"),
        _bash_start("ls"),
        _bash_end("a.txt"),
        ExecEvent(kind=ExecEventKind.RESULT, text="끝"),
    ], job="todo-1:1")
    assert [(t, j) for t, j, _ in q.rows()] == [
        ("tool_usage_started", "todo-1:1"),
        ("tool_usage_finished", "todo-1:1"),
    ]
    assert q.rows()[1][2]["result"] == "a.txt"


def test_chat_runs_do_not_write_the_work_item_timeline(monkeypatch, tmp_path):
    q = _run_stream(monkeypatch, tmp_path, [_bash_start(), _bash_end()], record=False)
    assert q.sent == []


def test_reclaim_closes_the_tool_the_dead_run_left_spinning(monkeypatch, tmp_path):
    # 1회차: 도구를 시작한 직후 죽는다
    first = _run_stream(monkeypatch, tmp_path, [_bash_start("python long.py", "T1"), _bash_end()], die_after=1)
    assert [t for t, _, _ in first.rows()] == ["tool_usage_started"]
    # 2회차(재점유): 시작하자마자 그 도구를 닫고, 자기 도구는 다시 하지 않은 채 이어 간다
    second = _run_stream(monkeypatch, tmp_path, [_bash_start("python step3.py", "T2"), _bash_end("ok", "T2")])
    rows = second.rows()
    assert rows[0][0] == "tool_usage_finished" and rows[0][2]["tool_use_id"] == "T1"
    assert rows[0][2]["result"] == timeline.INTERRUPTED_RESULT
    assert [r[2]["tool_use_id"] for r in rows[1:]] == ["T2", "T2"]


def test_cli_errors_are_collected_for_the_failure_text(monkeypatch, tmp_path):
    errors: list[str] = []
    _run_stream(monkeypatch, tmp_path, [ExecEvent(kind=ExecEventKind.ERROR, text="Not logged in · Please run /login")],
                errors=errors)
    assert errors == ["Not logged in · Please run /login"]


def test_fail_event_shows_the_cause_on_the_error_card():
    import executor as ex

    q = _Q()

    async def go():
        ex._JOB_ID.set("t:1")
        await ex.CliAgentExecutor()._fail(q, task_id="t", context_id="c", text="실행 제한 시간을 초과했습니다")

    asyncio.run(go())
    [(event_type, job, body)] = q.rows()
    assert event_type == "error"
    assert body["friendly"] == "실행 제한 시간을 초과했습니다"
    # 이번 실행의 카드 id — 화면이 그 카드를 "진행중" 대신 실패로 닫는다
    assert job == "t:1"


def test_notice_is_stored_with_a_type_the_store_accepts():
    import executor as ex

    q = _Q()
    asyncio.run(ex.CliAgentExecutor()._notice(_Ctx(), q, task_id="t", context_id="c", text="일부 스킬을 불러오지 못했습니다"))
    [(event_type, _, body)] = q.rows()
    assert event_type == "task_working"  # 'notice' 는 events.event_type enum 에 없어 거절됐다
    assert body["query"] == "일부 스킬을 불러오지 못했습니다"


# ---------------------------------------------------------------------------
# 카드 구성 — deepagents 와 같게: 작업 카드(cliagents) + 최종 결과 카드(result)
# ---------------------------------------------------------------------------

def _crew_rows(q):
    out = []
    for e in q.sent:
        meta = dict(e.metadata.items())
        if "event_type" not in meta:  # 결과 아티팩트
            continue
        text = e.status.message.parts[0].text
        out.append((meta["event_type"], meta["job_id"], meta["crew_type"], json.loads(text) if text else None))
    return out


def test_description_is_the_activity_description_not_the_whole_prompt():
    query = "[Description]\n에이전트 수행\n\n[Instruction]\n세 단계를 하라\n\n[InputData]\n{\"a\": 1}"
    assert timeline.description_of({"description": "지출결의서 작성", "query": query}) == "지출결의서 작성"
    assert timeline.description_of({"query": query}) == "에이전트 수행"
    assert timeline.description_of({"query": "그냥 지시"}) == ""


def test_work_card_is_the_agent_orch_with_only_the_description():
    import executor as ex

    class _Provider:
        display_name = "Claude Code"

    q = _Q()

    async def go():
        ex._JOB_ID.set("t")
        await ex.CliAgentExecutor()._started(
            q, task_id="t", context_id="c", provider=_Provider(),
            row={"activity_name": "에이전트 수행", "description": "에이전트 수행",
                 "query": "[Description]\n에이전트 수행\n\n[Instruction]\n긴 지시\n\n[InputData]\n{\"x\": 1}"},
        )

    asyncio.run(go())
    [(event_type, job, crew, body)] = _crew_rows(q)
    assert (event_type, job, crew) == ("task_started", "t", "cliagents")
    assert body["task_description"] == "에이전트 수행"  # 지시문·입력 JSON 이 카드에 쏟아지지 않는다


def test_answer_goes_on_its_own_result_card_and_the_work_card_closes_empty(tmp_path):
    import executor as ex
    from core.journal import Journal

    class _Outcome:
        payload = {"form": {"result": "step1"}}
        outputs = {"result": "step1"}
        raw_text = "step1"

    q = _Q()

    async def go():
        ex._JOB_ID.set("t:1")
        await ex.CliAgentExecutor()._complete(
            context=_Ctx(), event_queue=q, task_id="t", context_id="c",
            outcome=_Outcome(), session_id="S", journal=Journal(tmp_path),
        )

    asyncio.run(go())
    rows = _crew_rows(q)
    assert [(t, j, c) for t, j, c, _ in rows] == [
        ("task_completed", "t:1", "cliagents"),      # 작업 카드: 결과 없이 닫힘(채택 버튼 없음)
        ("task_started", "t:1:result", "result"),    # 최종 결과 카드
        ("task_completed", "t:1:result", "result"),
    ]
    assert rows[0][3] == {}
    assert rows[2][3]["form"] == {"result": "step1"}  # 채택하면 폼에 들어갈 값


def test_every_row_but_the_result_card_is_crew_type_cliagents(tmp_path):
    import executor as ex
    from core.workspace import Workspace

    q = _Q()

    async def go():
        ex._JOB_ID.set("t")
        e = ex.CliAgentExecutor()
        await e._notice(_Ctx(), q, task_id="t", context_id="c", text="알림")
        await e._pause(context=_Ctx(), event_queue=q, task_id="t", context_id="c",
                       workspace=Workspace(path=tmp_path, run_id="t"), agent_id="claude-code",
                       session_id="S", question="허용할까요?")
        await e._fail(q, task_id="t", context_id="c", text="실패")
        await e._timeline(q, task_id="t", context_id="c", rows=translate(_bash_start()))

    asyncio.run(go())
    assert {c for _, _, c, _ in _crew_rows(q)} == {"cliagents"}
