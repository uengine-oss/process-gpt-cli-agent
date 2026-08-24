"""One work item, or one chat turn, from claim to stored result.

The shape follows the deepagents executor because the platform expects it to:
the same `AgentExecutor` contract, the same UI event names, the same
draft/complete semantics. What differs is what sits in the middle — a CLI
process with its own tools and its own context, rather than a graph we drive
step by step.

The consequences of that difference are all in one direction: less is
observable, so more has to be said out loud. A run that could not use its tools
says so. A run that lost its session says so. A run that produced prose where
the form wanted fields fails instead of storing a shrug.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from a2a.helpers import new_text_artifact_update_event, new_text_status_update_event
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import TaskState
from cliagents import ExecEventKind
from google.protobuf.json_format import ParseDict
from processgpt_agent_sdk import emit_chunk_json, is_chat_request
from typing_extensions import override

from core import activity, bridge, hitl, skills
from core import events as ui_events
from core import prompt as prompt_builder
from core.journal import Journal
from core.outcome import interpret
from core.runner import ConcurrencyLimit, RunTimeoutError, astream
from core.selection import CliSelectionError, require_runnable, resolve
from core.settings import settings
from core.stream_registry import registry as stream_registry
from core.workspace import Workspace, for_run

logger = logging.getLogger(__name__)

#: Shared across every run in this process, so the cap is a property of the
#: container rather than of one work item.
limiter = ConcurrencyLimit(settings.max_concurrent_runs)


class CliAgentExecutor(AgentExecutor):
    """Runs a work item on the CLI agent the work item asked for."""

    @override
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        row: dict[str, Any] = dict(getattr(context, "row", {}) or {})
        extras: dict[str, Any] = dict(getattr(context, "metadata", {}) or {})
        task_id = str(context.task_id or "")
        context_id = str(context.context_id or "")
        chat = is_chat_request(context)

        run_id = task_id or context_id or "run"
        tenant_id = str(row.get("tenant_id") or extras.get("tenant_id") or "")
        workspace = for_run(run_id, tenant_id=tenant_id)

        try:
            await self._run(
                context=context,
                event_queue=event_queue,
                row=row,
                extras=extras,
                task_id=task_id,
                context_id=context_id,
                workspace=workspace,
                chat=chat,
            )
        except CliSelectionError as exc:
            # Never substitute another agent: the work item named one, and
            # running a different one produces work under a name nobody chose.
            await self._fail(
                event_queue,
                task_id=task_id,
                context_id=context_id,
                text=(
                    f"{exc}\n\n선택된 CLI 에이전트를 이 환경에서 실행할 수 없어 "
                    "업무를 진행하지 않았습니다. 다른 에이전트로 대체 실행하지 않습니다."
                ),
            )
            raise
        except RunTimeoutError as exc:
            await self._fail(
                event_queue,
                task_id=task_id,
                context_id=context_id,
                text=f"실행 제한 시간을 초과했습니다({exc}). 지금까지의 산출물은 보존됩니다.",
            )
            raise

    async def _run(
        self,
        *,
        context: RequestContext,
        event_queue: EventQueue,
        row: dict[str, Any],
        extras: dict[str, Any],
        task_id: str,
        context_id: str,
        workspace: Workspace,
        chat: bool,
    ) -> None:
        # The designer's choices for this activity — which CLI, which skills —
        # live in the process definition, not on the work item row. Read once,
        # and treat them as defaults the row may override: a value put on the
        # item deliberately should beat one inherited from the definition.
        declared = await asyncio.to_thread(activity.for_work_item, row)

        selection = resolve({"agent_config": declared.agent_config, **row, **extras})
        provider = require_runnable(selection)

        await self._started(
            event_queue,
            task_id=task_id,
            context_id=context_id,
            row=row,
            provider=provider,
        )

        journal = Journal(workspace.path)
        journal.capture_baseline(workspace.path)

        # --- what the agent can read -----------------------------------
        bundle, skill_failures = skills.build_bundle(
            instructions=_instructions(row, extras, workspace),
            skill_names=_named_skills(row, extras) or declared.skills,
            git_skills=_git_skills(row, extras),
        )
        provisioned = skills.provision(provider, bundle, workspace.path, skill_failures)
        if provisioned.failed:
            await self._notice(
                context,
                event_queue,
                task_id=task_id,
                context_id=context_id,
                text="일부 스킬을 불러오지 못했습니다: "
                + ", ".join(f"{k} ({v})" for k, v in provisioned.failed.items()),
            )

        # --- what the agent can do -------------------------------------
        servers = bridge.processgpt_servers(tenant_mcp=extras.get("tenant_mcp"))
        wiring = bridge.install(provider, workspace.path, servers)
        if wiring.failed:
            await self._notice(
                context,
                event_queue,
                task_id=task_id,
                context_id=context_id,
                text="일부 도구를 연결하지 못해 해당 도구 없이 진행합니다: "
                + ", ".join(wiring.failed),
            )

        # --- what the agent is asked ------------------------------------
        answer = _human_answer(row, extras)
        if answer:
            plan = hitl.plan_resume(
                workspace.path,
                answer,
                workspace_exists=workspace.exists,
                previous_summary=str(row.get("draft") or "")[:2000],
            )
            text = plan.prompt
            resume_session = plan.session_id or None
            if plan.restarted:
                await self._notice(
                    context,
                    event_queue,
                    task_id=task_id,
                    context_id=context_id,
                    text=f"이전 실행을 이어갈 수 없어 새로 시작합니다. ({plan.reason})",
                )
        else:
            text = prompt_builder.build(row, extras, workdir=str(workspace.path))
            resume_session = _session_of(row, extras)

        request = selection.to_request(
            text, str(workspace.path), resume_session=resume_session
        )

        # --- run --------------------------------------------------------
        stream_registry.start(workspace.run_id)
        async with limiter:
            final_text, session_id, paused = await self._stream(
                provider=provider,
                request=request,
                env=wiring.env or None,
                context=context,
                event_queue=event_queue,
                task_id=task_id,
                context_id=context_id,
                workspace=workspace,
                journal=journal,
            )

        # Whatever happened next — paused, stored, failed — this stream is
        # over, and a listener waiting on it should be released rather than
        # left hanging until its own timeout.
        stream_registry.finish(workspace.run_id)

        if paused is not None and not final_text:
            # A refusal with nothing to show is a real block: the agent could
            # not get past it, so a person has to decide.
            await self._pause(
                context=context,
                event_queue=event_queue,
                task_id=task_id,
                context_id=context_id,
                workspace=workspace,
                agent_id=selection.agent_id,
                session_id=session_id or "",
                question=paused,
            )
            return

        if paused is not None:
            # A refusal *and* an answer means the agent tried something it was
            # not allowed, then worked around it. Parking the item here would
            # discard finished work and wait for a decision nobody needs to
            # make — but the refusal is still worth saying out loud, because it
            # is why the answer may be thinner than it should be.
            await self._notice(
                context,
                event_queue,
                task_id=task_id,
                context_id=context_id,
                text=f"실행 중 허용되지 않은 동작이 있었습니다(결과는 그대로 저장합니다): {paused}",
            )

        # --- store ------------------------------------------------------
        outcome = interpret(final_text, extras.get("form_fields"))
        if not outcome.contract_met:
            await self._fail(
                event_queue,
                task_id=task_id,
                context_id=context_id,
                text=(
                    f"결과가 요구된 출력 형식과 맞지 않습니다: {outcome.mismatch_reason}\n\n"
                    f"에이전트 응답:\n{outcome.raw_text}"
                ),
            )
            return

        await self._complete(
            context=context,
            event_queue=event_queue,
            task_id=task_id,
            context_id=context_id,
            outcome=outcome,
            session_id=session_id,
            journal=journal,
        )

    async def _stream(
        self,
        *,
        provider,
        request,
        env,
        context: RequestContext,
        event_queue: EventQueue,
        task_id: str,
        context_id: str,
        workspace: Workspace,
        journal: Journal,
    ) -> tuple[str, str | None, str | None]:
        """Forward progress to the UI. Returns (final text, session, pause)."""
        final_text = ""
        session_id: str | None = None
        pause_reason: str | None = None
        streamed: list[str] = []

        async for event in astream(
            provider, request, env=env, timeout=settings.run_timeout_seconds
        ):
            if event.session_id:
                session_id = event.session_id

            if event.kind is ExecEventKind.RESULT:
                final_text = event.text or final_text
            elif event.kind is ExecEventKind.ASSISTANT_TEXT:
                streamed.append(event.text)
            elif event.kind is ExecEventKind.PERMISSION_REQUEST:
                # First refusal wins: the rest of the run is the agent trying
                # to work around a wall it cannot get past.
                pause_reason = pause_reason or (event.text or "권한이 필요합니다")
            elif event.kind is ExecEventKind.FILE_CHANGE and event.path:
                self._record_file(journal, workspace, event)
            elif event.kind is ExecEventKind.TOOL_END and event.tool and "/" in event.tool:
                journal.record_tool(event.tool, event.tool_input, external=True)

            for ui_event in ui_events.translate(event, workspace=workspace):
                payload = ui_event.as_dict()
                await emit_chunk_json(context, payload)
                # Also to the registry: a client that reconnects mid-run
                # replays from there rather than watching a dead spinner.
                stream_registry.publish(workspace.run_id, payload)

        return (final_text or "".join(streamed)).strip(), session_id, pause_reason

    @staticmethod
    def _record_file(journal: Journal, workspace: Workspace, event) -> None:
        """Journal a file change, or record that it left what we can replay."""
        from pathlib import Path

        path = Path(event.path)
        if not workspace.contains(path):
            journal.note_out_of_scope(f"워크스페이스 밖 경로 변경: {event.path}")
            return
        journal.record_file(path, event.change or "modified", workspace_path=workspace.path)

    # -- outcomes --------------------------------------------------------

    async def _started(
        self,
        event_queue: EventQueue,
        *,
        task_id: str,
        context_id: str,
        row: dict[str, Any],
        provider,
    ) -> None:
        """Announce the run, so the monitor has a card to fill in later.

        The work-item panel builds its timeline from `task_started` and applies
        `task_completed` only to a job it already knows about. Skip this and the
        finished result is stored correctly while the screen still says the job
        is queued — a run that worked, reported as one that never began.
        """
        await self._status(
            event_queue,
            task_id=task_id,
            context_id=context_id,
            state=TaskState.TASK_STATE_SUBMITTED,
            text=json.dumps(
                {
                    "goal": (row.get("activity_name") or "").strip() or "업무 수행",
                    "name": provider.display_name,
                    "role": "CLI 코딩 에이전트",
                    "task_description": (row.get("query") or "").strip(),
                },
                ensure_ascii=False,
            ),
            event_type="task_started",
            # Matches the completion event, so the card renders the answer text
            # rather than a raw payload dump.
            crew_type="result",
        )

    async def _complete(
        self,
        *,
        context: RequestContext,
        event_queue: EventQueue,
        task_id: str,
        context_id: str,
        outcome,
        session_id: str | None,
        journal: Journal,
    ) -> None:
        payload = dict(outcome.payload)
        if session_id:
            # Persisted so the next turn, or a human's answer tomorrow, can
            # resume this conversation instead of starting a new one.
            payload["cliagents_session_id"] = session_id
        if not journal.replayable:
            payload["replay_limited"] = journal.limitations()

        body = json.dumps(payload, ensure_ascii=False)
        await self._status(
            event_queue,
            task_id=task_id,
            context_id=context_id,
            state=TaskState.TASK_STATE_COMPLETED,
            text=body,
            event_type="task_completed",
            crew_type="result",
        )

        # `last_chunk` means "the run is over", not "the item is closed". The
        # platform decides output-vs-draft from the item's own `agent_mode`;
        # sending False here only tells it the run is still going, so a draft
        # run stays IN_PROGRESS forever with a finished answer sitting in it.
        artifact = new_text_artifact_update_event(
            task_id=task_id,
            context_id=context_id,
            name="assistant_response",
            text=body if outcome.outputs else outcome.raw_text,
            last_chunk=True,
        )
        ParseDict({"role": "assistant"}, artifact.metadata)
        await event_queue.enqueue_event(artifact)

    async def _pause(
        self,
        *,
        context: RequestContext,
        event_queue: EventQueue,
        task_id: str,
        context_id: str,
        workspace: Workspace,
        agent_id: str,
        session_id: str,
        question: str,
    ) -> None:
        """Stop and wait for a person, without calling it done or failed."""
        request = hitl.PendingRequest(
            run_id=workspace.run_id,
            agent_id=agent_id,
            session_id=session_id,
            question=question,
            fingerprint=hitl.fingerprint(agent_id, question),
        )
        is_new = hitl.remember(workspace.path, request)

        if is_new:
            await emit_chunk_json(
                context,
                {"type": "human_input_required", "question": question, "agent": agent_id},
            )

        await self._status(
            event_queue,
            task_id=task_id,
            context_id=context_id,
            state=TaskState.TASK_STATE_INPUT_REQUIRED,
            text=question,
            event_type="human_input_required",
            crew_type="agent",
        )

    async def _fail(
        self, event_queue: EventQueue, *, task_id: str, context_id: str, text: str
    ) -> None:
        await event_queue.enqueue_event(
            new_text_status_update_event(
                task_id=task_id,
                context_id=context_id,
                state=TaskState.TASK_STATE_FAILED,
                text=json.dumps({"error": text}, ensure_ascii=False),
            )
        )

    async def _notice(
        self,
        context: RequestContext,
        event_queue: EventQueue,
        *,
        task_id: str,
        context_id: str,
        text: str,
    ) -> None:
        """Tell the user about a degradation without failing the run."""
        logger.warning("run %s degraded: %s", task_id, text)
        await emit_chunk_json(context, {"type": "notice", "content": text})
        await self._status(
            event_queue,
            task_id=task_id,
            context_id=context_id,
            state=TaskState.TASK_STATE_WORKING,
            text=text,
            event_type="notice",
            crew_type="agent",
        )

    async def _status(
        self,
        event_queue: EventQueue,
        *,
        task_id: str,
        context_id: str,
        state,
        text: str,
        event_type: str,
        crew_type: str,
    ) -> None:
        event = new_text_status_update_event(
            task_id=task_id, context_id=context_id, state=state, text=text
        )
        ParseDict(
            {"event_type": event_type, "job_id": task_id, "crew_type": crew_type},
            event.metadata,
        )
        await event_queue.enqueue_event(event)

    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Stop the run and leave the work item claimable again.

        The child process is terminated by the stream teardown when the
        surrounding task is cancelled; this reports the state change.
        """
        task_id = str(context.task_id or "")
        context_id = str(context.context_id or "")
        logger.info("cancel requested for task %s", task_id)
        await event_queue.enqueue_event(
            new_text_status_update_event(
                task_id=task_id,
                context_id=context_id,
                state=TaskState.TASK_STATE_CANCELED,
                text="작업이 취소되었습니다.",
            )
        )


# --- reading the work item ------------------------------------------------


def _instructions(row: dict[str, Any], extras: dict[str, Any], workspace: Workspace) -> str:
    """The always-on project instruction file for this run."""
    lines = [
        "# ProcessGPT 업무 에이전트",
        "",
        "당신은 ProcessGPT 업무 프로세스 안에서 실행되는 에이전트입니다.",
        "- 작업 디렉터리 밖의 파일을 읽거나 수정하지 마세요.",
        # Without this the agent goes looking for a skill in the machine's home
        # directory, gets refused (that path is outside the workspace), and
        # stops — with a copy of the same skill sitting in the project.
        "- 이 업무에 필요한 스킬과 참고 문서는 모두 작업 디렉터리 안에 이미 제공되어 있습니다. "
        "홈 디렉터리(`~/.claude` 등)를 찾아보지 마세요.",
        "- 확실하지 않은 값을 지어내지 말고, 모르면 모른다고 결과에 적으세요.",
        "- 연결된 MCP 도구가 있으면 추측 대신 도구로 확인하세요.",
    ]
    activity = (row.get("activity_name") or "").strip()
    if activity:
        lines += ["", f"현재 업무: {activity}"]
    tenant = (row.get("tenant_id") or "").strip()
    if tenant:
        lines += [f"테넌트: {tenant}"]

    # A skill that generates process artifacts documents their file names but
    # not where this run's files belong — that is a per-run value only the
    # service holds. Said once here rather than repeated in every skill.
    lines += ["", prompt_builder.artifact_paths(str(workspace.path), workspace.run_id)]
    return "\n".join(lines)


def _named_skills(row: dict[str, Any], extras: dict[str, Any]) -> list[str]:
    for source in (row, extras):
        raw = source.get("skills") or source.get("agent_skills")
        if isinstance(raw, str) and raw.strip():
            return [s.strip() for s in raw.split(",") if s.strip()]
        if isinstance(raw, list):
            return [str(s).strip() for s in raw if str(s).strip()]
    for agent in extras.get("agents") or []:
        if isinstance(agent, dict) and agent.get("skills"):
            raw = agent["skills"]
            if isinstance(raw, str):
                return [s.strip() for s in raw.split(",") if s.strip()]
    return []


def _git_skills(row: dict[str, Any], extras: dict[str, Any]) -> dict[str, str]:
    raw = row.get("git_skills") or extras.get("git_skills")
    if isinstance(raw, str) and raw.strip():
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {str(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}


def _human_answer(row: dict[str, Any], extras: dict[str, Any]) -> str:
    """The reply to a question this run asked earlier, if there is one."""
    for key in ("human_answer", "feedback_answer", "user_answer"):
        value = row.get(key) or extras.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _session_of(row: dict[str, Any], extras: dict[str, Any]) -> str | None:
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


