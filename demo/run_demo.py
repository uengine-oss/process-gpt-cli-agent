"""Run the demo for real and record what happened.

Everything this writes to ``transcript.json`` comes from an actual run: a real
work item shape, the real selection and prompt code, a real ``claude``/``codex``
process, and the real outcome parsing. Nothing is staged.

That constraint is the point. A screencast of a mocked pipeline proves the
mock works, and this pipeline's whole risk is in the parts a mock would replace
— what the CLI actually emits, whether the file actually lands, whether the
answer actually parses into the form the process asked for.

    python -m demo.run_demo                 # all scenes
    python -m demo.run_demo --scene refusal # just the free one

The rendering step (``demo/record.mjs``) reads the transcript. Splitting them
means the video can be re-cut without spending tokens again.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from cliagents import ExecEventKind, Surface, registry

from core import prompt as prompt_builder
from core import skills
from core.availability import snapshot
from core.journal import Journal
from core.outcome import interpret
from core.runner import astream
from core.selection import CliSelectionError, require_runnable, resolve
from core.workspace import for_run

HERE = Path(__file__).parent
TRANSCRIPT = HERE / "transcript.json"

#: Cheap and quick — the demo is about the pipeline, not about model quality.
DEMO_MODEL = "haiku"

#: A work item shaped the way the ProcessGPT engine delivers one, with a form
#: attached so the output contract is exercised rather than described.
WORK_ITEM: dict[str, Any] = {
    "id": "demo-expense-report",
    "tenant_id": "demo",
    "agent_orch": "cliagents",
    "agent_mode": "COMPLETE",
    "activity_name": "출장 경비 정산 검토",
    "query": (
        "expense-review.md 파일에 아래 3건의 경비 검토 결과를 표로 작성하세요.\n"
        "- 2026-08-01 KTX 서울→부산 59,800원\n"
        "- 2026-08-01 숙박 128,000원\n"
        "- 2026-08-02 식대 32,000원\n"
        "그 다음 지정된 JSON 형식으로만 최종 답을 제출하세요."
    ),
    "agent_config": {"agent_cli": "claude-code", "model": DEMO_MODEL, "permission": "workspace_write"},
}

FORM_FIELDS = [
    {"key": "total_amount", "label": "총 금액", "type": "number"},
    {"key": "verdict", "label": "검토 결과", "type": "text"},
]


def _step(scene: list[dict], kind: str, **payload: Any) -> None:
    scene.append({"kind": kind, "at": round(time.time(), 3), **payload})
    label = payload.get("title") or payload.get("text") or payload.get("tool") or ""
    print(f"  [{kind}] {str(label)[:90]}")


async def scene_availability() -> dict:
    """What this machine can actually delegate to."""
    print("\n=== scene: availability")
    steps: list[dict] = []
    _step(steps, "narration", title="이 서버가 실행할 수 있는 CLI 에이전트를 확인합니다",
          detail="GET /agents — 설치 여부와 설치 방법을 함께 돌려줍니다")
    _step(steps, "command", text="curl -s $CLI_AGENT/agents | jq")

    agents = [a.as_dict() for a in snapshot(refresh=True, check_auth=False)]
    _step(steps, "json", title="응답", payload={"agents": agents})
    return {"id": "availability", "title": "1. 어떤 CLI 를 쓸 수 있나", "steps": steps}


async def scene_delegate(agent_id: str) -> dict:
    """A work item, delegated to a real CLI, start to finish."""
    print(f"\n=== scene: delegate ({agent_id})")
    steps: list[dict] = []

    row = {**WORK_ITEM, "agent_config": {**WORK_ITEM["agent_config"], "agent_cli": agent_id}}
    extras = {"form_fields": FORM_FIELDS, "users": [{"name": "김담당"}]}

    _step(steps, "narration", title="업무 하나를 CLI 에이전트에게 위임합니다",
          detail=f"agent_orch=cliagents · CLI={agent_id} · 권한=workspace_write")
    _step(steps, "workitem", payload={
        "activity_name": row["activity_name"],
        "query": row["query"],
        "form_fields": [f["key"] for f in FORM_FIELDS],
        "agent_config": row["agent_config"],
    })

    selection = resolve(row)
    provider = require_runnable(selection)
    _step(steps, "note", text=f"선택 확정: {selection.agent_id} / 모델 {selection.model} / 권한 {selection.permission.value}")

    workspace = for_run(f"demo-{agent_id}", tenant_id="demo")
    shutil.rmtree(workspace.path, ignore_errors=True)
    workspace.path.mkdir(parents=True, exist_ok=True)
    _step(steps, "note", text=f"실행 전용 작업 공간: {workspace.path}")

    journal = Journal(workspace.path)
    journal.capture_baseline(workspace.path)

    bundle, failures = skills.build_bundle(instructions=_instructions(), skill_names=[])
    provisioned = skills.provision(provider, bundle, workspace.path, failures)
    _step(steps, "files", title=f"{provider.display_name} 관례로 주입된 파일",
          payload=[p for p in provisioned.paths][:6])

    text = prompt_builder.build(row, extras, workdir=str(workspace.path))
    _step(steps, "prompt", title="에이전트에게 전달되는 프롬프트", text=text)

    request = selection.to_request(text, str(workspace.path))
    _step(steps, "command", text=" ".join(provider.exec_argv(request)[:8]) + " …")
    _step(steps, "narration", title="실행", detail="CLI 가 내보내는 이벤트를 공통 형식으로 정규화해 받습니다")

    final_text = ""
    session_id = None
    started = time.time()
    async for event in astream(provider, request, timeout=600):
        if event.session_id:
            session_id = event.session_id
        if event.kind is ExecEventKind.RESULT:
            final_text = event.text or final_text
        if event.kind in (ExecEventKind.TOOL_START, ExecEventKind.TOOL_END, ExecEventKind.FILE_CHANGE):
            _step(steps, "event", event_kind=event.kind.value, tool=event.tool,
                  path=workspace.relative(event.path) if event.path else None,
                  change=event.change)
        elif event.kind is ExecEventKind.ASSISTANT_TEXT and event.text.strip():
            _step(steps, "event", event_kind="assistant_text", text=event.text)
        if event.kind is ExecEventKind.FILE_CHANGE and event.path:
            path = Path(event.path)
            if workspace.contains(path):
                journal.record_file(path, event.change or "modified", workspace_path=workspace.path)

    elapsed = round(time.time() - started, 1)
    _step(steps, "note", text=f"실행 완료 · {elapsed}초 · session={session_id}")

    produced = [workspace.relative(p) for p in workspace.files()
                if not workspace.relative(p).startswith(".")]
    _step(steps, "files", title="에이전트가 만든 산출물", payload=produced)

    report = workspace.path / "expense-review.md"
    if report.is_file():
        _step(steps, "filebody", title="expense-review.md",
              text=report.read_text(encoding="utf-8")[:1200])

    outcome = interpret(final_text, FORM_FIELDS)
    _step(steps, "outcome", title="폼 계약 검증",
          payload={
              "contract_met": outcome.contract_met,
              "outputs": outcome.outputs,
              "mismatch_reason": outcome.mismatch_reason,
          })
    _step(steps, "narration", title="워크아이템에 저장",
          detail="agent_mode=COMPLETE 이므로 업무가 완료 처리됩니다"
                 if outcome.contract_met else
                 "형식이 맞지 않으면 저장하지 않고 실패로 보고합니다")

    return {"id": f"delegate-{agent_id}", "title": f"2. 업무 위임 — {provider.display_name}",
            "steps": steps, "workspace": str(workspace.path)}


async def scene_refusal() -> dict:
    """The rule that matters most operationally, and it costs nothing to show."""
    print("\n=== scene: refusal")
    steps: list[dict] = []
    _step(steps, "narration", title="설치되지 않은 CLI 를 지정하면",
          detail="다른 에이전트로 조용히 대체하지 않습니다")

    row = {**WORK_ITEM, "agent_config": {"agent_cli": "gemini-cli"}}
    _step(steps, "workitem", payload={"agent_config": row["agent_config"]})
    try:
        require_runnable(resolve(row))
        _step(steps, "note", text="(예상과 다름: 대체 실행이 일어났습니다)")
    except CliSelectionError as exc:
        _step(steps, "failure", title="업무 실패 — 안내와 함께", text=str(exc))
        _step(steps, "narration", title="왜 이렇게 하나",
              detail="아무 에이전트나 대신 돌리면, 사용자가 고르지 않은 이름으로 결과물이 남습니다")
    return {"id": "refusal", "title": "3. 없는 CLI 는 대체하지 않는다", "steps": steps}


async def scene_undo(workspace_path: str) -> dict:
    """Undo, scoped to what was actually observable."""
    print("\n=== scene: undo")
    steps: list[dict] = []
    workspace = Path(workspace_path)
    journal = Journal(workspace)

    _step(steps, "narration", title="되돌리기",
          detail="실행이 만든 파일을 실행 전 상태로 돌립니다. 모델은 다시 호출하지 않습니다")

    before = sorted(p.name for p in workspace.glob("*.md"))
    _step(steps, "files", title="되돌리기 전", payload=before)

    restored, irreversible = journal.undo_files(workspace)
    after = sorted(p.name for p in workspace.glob("*.md"))
    _step(steps, "json", title="undo 결과",
          payload={"restored": restored, "irreversible": irreversible,
                   "fully_replayable": journal.replayable})
    _step(steps, "files", title="되돌리기 후", payload=after)
    _step(steps, "narration", title="범위는 정직하게",
          detail="기록 대상은 MCP 호출과 작업 공간 파일뿐입니다. 그 밖은 '부분 재실행 불가' 로 표시합니다")
    return {"id": "undo", "title": "4. 되돌리기", "steps": steps}


def _instructions() -> str:
    return (
        "# ProcessGPT 업무 에이전트\n\n"
        "당신은 ProcessGPT 업무 프로세스 안에서 실행되는 에이전트입니다.\n"
        "- 작업 디렉터리 밖의 파일을 수정하지 마세요.\n"
        "- 확실하지 않은 값을 지어내지 마세요.\n"
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", help="record one scene only")
    parser.add_argument("--cli", default="claude-code", help="which CLI to delegate to")
    args = parser.parse_args()

    installed = [p.id for p in registry.for_surface(Surface.EXEC) if p.detect(refresh=True).installed]
    if args.scene != "refusal" and args.cli not in installed:
        print(f"{args.cli} is not installed here (have: {installed or 'none'})", file=sys.stderr)
        return 2

    scenes = []
    if args.scene in (None, "availability"):
        scenes.append(await scene_availability())
    if args.scene in (None, "delegate"):
        delegated = await scene_delegate(args.cli)
        scenes.append(delegated)
    if args.scene in (None, "refusal"):
        scenes.append(await scene_refusal())
    if args.scene in (None, "delegate") and scenes:
        workspace = next((s["workspace"] for s in scenes if s.get("workspace")), None)
        if workspace:
            scenes.append(await scene_undo(workspace))

    TRANSCRIPT.write_text(
        json.dumps({"recorded_at": time.time(), "cli": args.cli, "scenes": scenes},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\ntranscript → {TRANSCRIPT} ({sum(len(s['steps']) for s in scenes)} steps)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
