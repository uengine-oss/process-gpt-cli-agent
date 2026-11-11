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
import contextvars
import json
import logging
from typing import Any

from a2a.helpers import new_text_artifact_update_event, new_text_status_update_event
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import TaskState
from cliagents import ArtifactKind, ExecEventKind
from google.protobuf.json_format import ParseDict
from processgpt_agent_sdk import emit_chunk_json, is_chat_request
from processgpt_agent_sdk.artifacts import strip_local_paths
from typing_extensions import override

from core import activity, bridge, chat_session, hitl, resume, skills, subagents
from core import events as ui_events
from core import timeline
from core import prompt as prompt_builder
from core.availability import snapshot as cli_snapshot
from core.journal import Journal
from core.outcome import interpret
from core.runner import ConcurrencyLimit, RunTimeoutError, astream
from core.runtime import RuntimeLease
from core.selection import CliNotChosenError, CliSelectionError, chat_choice, require_runnable, resolve
from core.settings import settings
from core.stream_registry import registry as stream_registry
from core.workspace import Workspace, for_run

logger = logging.getLogger(__name__)

#: Shared across every run in this process, so the cap is a property of the
#: container rather than of one work item.
limiter = ConcurrencyLimit(settings.max_concurrent_runs)

#: Runs this process is executing — chat turns and work items alike, from the
#: first line of ``execute`` to the last byte archived. Wider than the
#: limiter, which only counts live CLI processes: a Pod still uploading a
#: finished turn's session is not idle either.
_active_runs = 0


def active_runs() -> int:
    """How many runs are in progress here. ``/health`` reports busy on it."""
    return _active_runs

#: The timeline card this run writes to (``timeline.job_id``). A context
#: variable, not an attribute: one executor serves concurrent runs.
_JOB_ID: contextvars.ContextVar[str] = contextvars.ContextVar("cli_agent_job_id", default="")


class CliAgentExecutor(AgentExecutor):
    """Runs a work item on the CLI agent the work item asked for."""

    @override
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        global _active_runs
        _active_runs += 1
        try:
            await self._execute(context, event_queue)
        finally:
            _active_runs -= 1

    async def _execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        row: dict[str, Any] = dict(getattr(context, "row", {}) or {})
        extras: dict[str, Any] = dict(getattr(context, "metadata", {}) or {})
        task_id = str(context.task_id or "")
        context_id = str(context.context_id or "")
        chat = is_chat_request(context)
        _JOB_ID.set(timeline.job_id(task_id, row))

        if chat and not context.get_user_input().strip():
            await self._chat_error(
                context,
                code="empty_message",
                text="보낼 메시지가 비어 있어 CLI 에이전트를 실행하지 않았습니다.",
                agent_id="",
            )
            return

        run_id = task_id or context_id or "run"
        tenant_id = str(row.get("tenant_id") or extras.get("tenant_id") or "")
        workspace = for_run(run_id, tenant_id=tenant_id)
        runtime_lease = RuntimeLease(workspace.path)

        # A chat's workspace is the conversation's, and this Pod may be a new
        # one. Restore before anything writes here: a directory that already
        # holds files is taken to be current and is not restored.
        archive = chat
        if chat:
            await chat_session.restore(workspace, tenant_id=tenant_id, conversation_id=context_id)

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
                runtime_lease=runtime_lease,
            )
        except CliSelectionError as exc:
            # Nothing ran, so there is nothing new to keep.
            archive = False
            if chat:
                # The chat stream ignores status events, so `_fail` would end
                # the turn in silence. Say it on the stream, with a code the
                # screen can act on, and end the turn there.
                await self._refuse_chat(context, exc)
                return
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
        finally:
            # Native agent definitions can contain tenant MCP credentials.
            # Restore/remove every managed runtime file after every attempt;
            # sessions and business artifacts are deliberately not managed.
            runtime_lease.restore()
            # After the lease, so what is archived is the conversation and not
            # this turn's provisioning. Also after a stop or a timeout: the
            # next turn should continue from where this one was cut off.
            if archive:
                await chat_session.save(workspace, tenant_id=tenant_id, conversation_id=context_id)

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
        runtime_lease: RuntimeLease,
    ) -> None:
        # The designer's choices for this activity — which CLI, which skills —
        # live in the process definition, not on the work item row. Read once,
        # and treat them as defaults the row may override: a value put on the
        # item deliberately should beat one inherited from the definition.
        declared = await asyncio.to_thread(activity.for_work_item, row)

        # A chat names its CLI in the request (`metadata.agent_cli`) or not at
        # all; the configured default is a work-item fallback, not a chat one.
        #
        # A chat request is written by whoever is chatting, so it may choose the
        # CLI and model but not the permission: `command_exec` gives the CLI a
        # shell, and the shell inherits this pod's environment. Chat runs at the
        # configured default permission; widening it is the work item designer's
        # call (the activity definition), not the chat user's.
        selection = resolve(
            {"agent_config": declared.agent_config, **row, **(chat_choice(extras) if chat else extras)},
            require_choice=chat,
        )
        provider = require_runnable(selection)
        message = context.get_user_input() if chat else ""

        await self._started(
            event_queue,
            task_id=task_id,
            context_id=context_id,
            row=row,
            provider=provider,
        )

        journal = Journal(workspace.path)
        journal.capture_baseline(workspace.path)

        # Resolve capabilities before writing anything. MCP is attached to
        # native subagents, never to the parent CLI session.
        tenant_servers = bridge.processgpt_servers(tenant_mcp=extras.get("tenant_mcp"))
        requested_skills = _named_skills(row, extras) or declared.skills
        runtime_agents = subagents.prepare(
            list(extras.get("agents") or []),
            tenant_servers=tenant_servers,
            fallback_skills=requested_skills,
            fallback_tools=declared.tools,
        )
        selected_skills = (
            subagents.selected_skill_names(runtime_agents)
            if extras.get("agents")
            else requested_skills
        )

        # --- what the agent can read -----------------------------------
        bundle, skill_failures = skills.build_bundle(
            instructions=(
                _instructions(row, extras, workspace, chat=chat)
                + "\n\n"
                + subagents.delegation_instructions(runtime_agents)
            ),
            skill_names=selected_skills,
            git_skills=_git_skills(row, extras),
            tenant_id=str(row.get("tenant_id") or extras.get("tenant_id") or ""),
        )
        runtime_agents = subagents.preload_discovered_skills(
            runtime_agents,
            [artifact.name for artifact in bundle.of_kind(ArtifactKind.SKILL)],
        )
        subagents.provision_definitions(
            provider, bundle, runtime_agents, workspace.path, runtime_lease
        )
        provisioned = skills.provision(
            provider, bundle, workspace.path, skill_failures, lease=runtime_lease
        )
        if provisioned.failed:
            await self._notice(
                context,
                event_queue,
                task_id=task_id,
                context_id=context_id,
                text="일부 스킬을 불러오지 못했습니다: "
                + ", ".join(f"{k} ({v})" for k, v in provisioned.failed.items()),
            )

        # Codex keeps session state under CODEX_HOME. Relocate that home per
        # workspace; provider.exec_env merges this override into os.environ so
        # deployment API keys are retained.
        runtime_env = bridge.runtime_env(provider, workspace.path)

        # --- what the agent is asked ------------------------------------
        if chat:
            # The user's words, on the session this conversation is already
            # on — which a restored archive brings back on a fresh Pod.
            start = resume.plan_chat(workspace.path, message, agent_id=selection.agent_id)
        else:
            # Why this run is starting (SDK: new / crash reclaim / human
            # answer / revision) picks the prompt and the session to continue.
            start = resume.plan_start(
                workspace.path,
                row=row,
                extras=extras,
                workspace_exists=workspace.exists,
                fresh_prompt=lambda: prompt_builder.build(row, extras, workdir=str(workspace.path)),
                previous_summary=str(row.get("draft") or "")[:2000],
            )
        text = start.prompt
        resume_session = start.session_id
        if start.notice:
            await self._notice(
                context,
                event_queue,
                task_id=task_id,
                context_id=context_id,
                text=start.notice,
            )

        request = selection.to_request(
            text, str(workspace.path), resume_session=resume_session
        )

        # --- run --------------------------------------------------------
        if chat:
            # What `outputs/` held before, so only this turn's files are
            # offered for download.
            chat_session.outputs_dir(workspace).mkdir(exist_ok=True)
            outputs_before = chat_session.before_turn(workspace)

        cli_errors: list[str] = []
        stream_registry.start(workspace.run_id)
        try:
            async with limiter:
                final_text, session_id, paused = await self._stream(
                    provider=provider,
                    request=request,
                    env=runtime_env or None,
                    context=context,
                    event_queue=event_queue,
                    task_id=task_id,
                    context_id=context_id,
                    workspace=workspace,
                    journal=journal,
                    record=not chat,
                    chat=chat,
                    errors=cli_errors,
                )
        finally:
            # Whatever happened next — paused, stored, failed, stopped — this
            # stream is over, and a listener waiting on it should be released
            # rather than left hanging until its own timeout.
            stream_registry.finish(workspace.run_id)

        outcome = interpret(final_text, extras.get("form_fields"))
        if hitl.should_pause(refused=paused is not None, final_text=final_text, contract_met=outcome.contract_met):
            # A refusal with nothing usable to show is a real block: the agent
            # could not get past it, so a person has to decide.
            await self._pause(
                context=context,
                event_queue=event_queue,
                task_id=task_id,
                context_id=context_id,
                workspace=workspace,
                agent_id=selection.agent_id,
                session_id=session_id or "",
                question=paused or "",
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
        if not outcome.contract_met:
            await self._fail(
                event_queue,
                task_id=task_id,
                context_id=context_id,
                text=(
                    f"결과가 요구된 출력 형식과 맞지 않습니다: {outcome.mismatch_reason}\n\n"
                    f"에이전트 응답:\n{outcome.raw_text}"
                    # The CLI's own error ("Not logged in", a crashed tool) is
                    # usually the real cause; without it the card only says
                    # the answer had the wrong shape.
                    + ("\n\nCLI 오류:\n" + "\n".join(cli_errors) if cli_errors else "")
                ),
            )
            return

        files: list[dict[str, Any]] = []
        if chat:
            published = await chat_session.publish(
                workspace,
                tenant_id=str(extras.get("tenant_id") or ""),
                conversation_id=context_id,
                before=outputs_before,
            )
            files = published.files
            if published.unlinked:
                # Said in the answer itself: a `notice` event is not drawn by
                # the chat screen, and "the agent made a file you cannot get"
                # is exactly what must not pass in silence.
                note = (
                    "다음 산출물은 내려받기 링크를 만들지 못했습니다: "
                    + ", ".join(published.unlinked)
                    + (f" ({published.reason})" if published.reason else "")
                )
                await self._notice(
                    context, event_queue, task_id=task_id, context_id=context_id, text=note
                )
                outcome.raw_text = f"{outcome.raw_text}\n\n> {note}".strip()

        await self._complete(
            context=context,
            event_queue=event_queue,
            task_id=task_id,
            context_id=context_id,
            outcome=outcome,
            session_id=session_id,
            journal=journal,
            files=files,
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
        record: bool = True,
        chat: bool = False,
        errors: list[str] | None = None,
    ) -> tuple[str, str | None, str | None]:
        """Forward progress to the UI. Returns (final text, session, pause).

        Two audiences: the chat stream (SSE chunks, gone on refresh) and, for a
        work item (``record``), the stored timeline the work-item screen draws.
        CLI error messages are appended to ``errors`` for the failure text.

        In ``chat``, the answer's text goes out as SDK ``token`` chunks: that
        is what the chat screen draws as the answer, and what the SDK keeps
        for the snapshot a reattaching client (``/chat/stream/attach``) gets
        first. Sent as anything else, a reattach shows an empty answer.
        """
        final_text = ""
        session_id: str | None = None
        pause_reason: str | None = None
        streamed: list[str] = []
        open_tools = timeline.OpenTools(workspace.path)

        if record:
            # A run that died mid-tool left that tool open on the card. This
            # run took over; say so on the tool instead of spinning forever.
            await self._timeline(
                event_queue, task_id=task_id, context_id=context_id,
                rows=open_tools.close_interrupted(),
            )

        async for event in astream(
            provider, request, env=env, timeout=settings.run_timeout_seconds
        ):
            if event.session_id:
                if event.session_id != session_id:
                    # Saved now, not at completion: a run killed halfway is the
                    # one a reclaim has to continue.
                    resume.remember_session(
                        workspace.path, event.session_id, agent_id=getattr(provider, "id", "")
                    )
                session_id = event.session_id

            if event.kind is ExecEventKind.RESULT:
                final_text = event.text or final_text
            elif event.kind is ExecEventKind.ASSISTANT_TEXT:
                streamed.append(event.text)
            elif (refusal := hitl.permission_refusal(event)) is not None:
                # First refusal wins: the rest of the run is the agent trying
                # to work around a wall it cannot get past.
                pause_reason = pause_reason or refusal
            elif event.kind is ExecEventKind.FILE_CHANGE and event.path:
                self._record_file(journal, workspace, event)
            elif event.kind is ExecEventKind.TOOL_END and event.tool and "/" in event.tool:
                journal.record_tool(event.tool, event.tool_input, external=True)
            elif event.kind is ExecEventKind.ERROR and event.text and errors is not None:
                errors.append(event.text)

            if record:
                rows = timeline.translate(event)
                # Persist before sending: if the process dies right after a
                # start is stored, the next run must know to close it.
                open_tools.track(rows)
                await self._timeline(event_queue, task_id=task_id, context_id=context_id, rows=rows)

            for ui_event in ui_events.translate(event, workspace=workspace):
                payload = ui_event.as_dict()
                if chat and ui_event.type == "text":
                    payload = {"type": "token", "content": ui_event.data.get("content", "")}
                await emit_chunk_json(context, payload)
                # Also to the registry: a client that reconnects mid-run
                # replays from there rather than watching a dead spinner.
                stream_registry.publish(workspace.run_id, payload)

        return (final_text or "".join(streamed)).strip(), session_id, pause_reason

    async def _timeline(
        self, event_queue: EventQueue, *, task_id: str, context_id: str, rows: list
    ) -> None:
        """Store timeline rows — tool calls and plan on this run's card, a
        subagent on a card of its own (``<job>:agent:<call id>``)."""
        job = _JOB_ID.get() or task_id
        for row in rows:
            await self._status(
                event_queue,
                task_id=task_id,
                context_id=context_id,
                state=TaskState.TASK_STATE_WORKING,
                text=row.text(),
                event_type=row.event_type,
                crew_type=timeline.CREW_TYPE,
                job_id=f"{job}:{row.card}" if row.card else None,
            )

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
                    # The activity's Description, not ``query`` — that is the
                    # whole prompt with the input JSON.
                    "task_description": timeline.description_of(row),
                },
                ensure_ascii=False,
            ),
            event_type="task_started",
            # The work card: progress and tool calls. The answer goes on its
            # own result card (_complete), where it can be adopted.
            crew_type=timeline.CREW_TYPE,
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
        files: list[dict[str, Any]] | None = None,
    ) -> None:
        payload = dict(outcome.payload)
        if session_id:
            # Persisted so the next turn, or a human's answer tomorrow, can
            # resume this conversation instead of starting a new one.
            payload["cliagents_session_id"] = session_id
        if not journal.replayable:
            payload["replay_limited"] = journal.limitations()

        body = json.dumps(payload, ensure_ascii=False)
        # Close the work card without a result (``{}``), then put the answer on
        # its own result card — as deepagents does. "채택" is offered only there,
        # so the work card is never adopted into the form.
        await self._status(
            event_queue,
            task_id=task_id,
            context_id=context_id,
            state=TaskState.TASK_STATE_WORKING,
            text="{}",
            event_type="task_completed",
            crew_type=timeline.CREW_TYPE,
        )
        result_job = timeline.result_job_id(_JOB_ID.get() or task_id)
        await self._status(
            event_queue,
            task_id=task_id,
            context_id=context_id,
            state=TaskState.TASK_STATE_WORKING,
            text=json.dumps(
                {"goal": "최종 결과를 반환합니다.", "name": "최종 결과 반환", "role": "최종 결과 반환"},
                ensure_ascii=False,
            ),
            event_type="task_started",
            crew_type=timeline.RESULT_CREW_TYPE,
            job_id=result_job,
        )
        await self._status(
            event_queue,
            task_id=task_id,
            context_id=context_id,
            state=TaskState.TASK_STATE_COMPLETED,
            text=body,
            event_type="task_completed",
            crew_type=timeline.RESULT_CREW_TYPE,
            job_id=result_job,
        )

        # `last_chunk` means "the run is over", not "the item is closed". The
        # platform decides output-vs-draft from the item's own `agent_mode`;
        # sending False here only tells it the run is still going, so a draft
        # run stays IN_PROGRESS forever with a finished answer sitting in it.
        text = body if outcome.outputs else outcome.raw_text
        metadata: dict[str, Any] = {"role": "assistant"}
        if files:
            # `pdfFiles` is the SDK's name for a turn's downloads: it rides on
            # `done.files` and is stored on the chat message, so the links
            # survive a reload and the Pod that made the files.
            metadata["pdfFiles"] = files
            # The agent names its files by the path it saw. Point those at
            # the links instead of at a disk the user cannot reach.
            text = strip_local_paths(text, files)

        artifact = new_text_artifact_update_event(
            task_id=task_id,
            context_id=context_id,
            name="assistant_response",
            text=text,
            last_chunk=True,
        )
        ParseDict(metadata, artifact.metadata)
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

        # End this stretch's card before the question. The answer starts the
        # next card (``timeline.job_id``), so the screen reads work → question →
        # work. An empty body (``{}``): the card says "done" without a result
        # box, so nothing here can be adopted into the form by mistake. Not an
        # empty string — the SDK stores the whole message object for that.
        await self._status(
            event_queue,
            task_id=task_id,
            context_id=context_id,
            state=TaskState.TASK_STATE_WORKING,
            text="{}",
            event_type="task_completed",
            crew_type=timeline.CREW_TYPE,
        )

        await self._status(
            event_queue,
            task_id=task_id,
            context_id=context_id,
            state=TaskState.TASK_STATE_INPUT_REQUIRED,
            # The work-item screen draws its question card from a `human_asked`
            # event and reads `question`/`type` from its data. Any other event
            # type is not in the store's enum and is never saved.
            text=json.dumps(
                {"question": question, "type": "text", "agent": agent_id, "role": "CLI 코딩 에이전트"},
                ensure_ascii=False,
            ),
            event_type="human_asked",
            crew_type=timeline.CREW_TYPE,
            # Its own card id: the screen lists a card's tool calls under every
            # card sharing that id, so a question on the work card's id showed
            # the same tools twice. The answer copies this id.
            job_id=f"{_JOB_ID.get() or task_id}:ask",
        )

    async def _fail(
        self, event_queue: EventQueue, *, task_id: str, context_id: str, text: str
    ) -> None:
        # The error card shows `friendly`; `error` alone showed only "오류가
        # 발생했습니다". On this run's card id, so the screen closes that card as
        # failed instead of leaving it "in progress" next to the error.
        await self._status(
            event_queue,
            task_id=task_id,
            context_id=context_id,
            state=TaskState.TASK_STATE_FAILED,
            text=json.dumps(timeline.failure_data(text), ensure_ascii=False),
            event_type="error",
            crew_type=timeline.CREW_TYPE,
        )

    async def _refuse_chat(self, context: RequestContext, exc: CliSelectionError) -> None:
        """End a chat turn whose CLI cannot run, and say what to do about it."""
        if isinstance(exc, CliNotChosenError):
            await self._chat_error(
                context,
                code="cli_not_selected",
                text=(
                    "이 대화에 사용할 CLI 에이전트가 선택되지 않았습니다. "
                    "Claude Code 나 Codex 중 하나를 골라 다시 보내 주세요. "
                    "임의로 다른 에이전트를 대신 실행하지 않습니다."
                ),
                agent_id="",
            )
            return
        await self._chat_error(
            context,
            code="cli_unavailable",
            text=(
                f"{exc}\n\n선택한 CLI 에이전트를 이 환경에서 실행할 수 없어 답하지 않았습니다. "
                "다른 에이전트로 대체 실행하지 않습니다."
            ),
            agent_id=exc.agent_id,
        )

    async def _chat_error(
        self, context: RequestContext, *, code: str, text: str, agent_id: str
    ) -> None:
        """An error event the chat screen shows, with what can run here.

        ``available`` lists the CLIs installed in this deployment — the same
        answer ``GET /agents`` gives — so the screen can offer a choice
        instead of only a failure.
        """
        logger.warning("chat turn refused (%s): %s", code, text.splitlines()[0])
        available = [a.agent_id for a in cli_snapshot(refresh=False) if a.installed]
        await emit_chunk_json(
            context,
            {
                "type": "error",
                "code": code,
                "error": text,
                "content": text,
                "agent": agent_id,
                "available": available,
            },
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
        # `notice` is not in the store's event_type enum — those rows were
        # rejected. `task_working` is, and the card lists it with `query`.
        await self._status(
            event_queue,
            task_id=task_id,
            context_id=context_id,
            state=TaskState.TASK_STATE_WORKING,
            text=json.dumps({"query": text, "info": text}, ensure_ascii=False),
            event_type="task_working",
            crew_type=timeline.CREW_TYPE,
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
        job_id: str | None = None,
    ) -> None:
        event = new_text_status_update_event(
            task_id=task_id, context_id=context_id, state=state, text=text
        )
        ParseDict(
            {"event_type": event_type, "job_id": job_id or _JOB_ID.get() or task_id, "crew_type": crew_type},
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


def _instructions(
    row: dict[str, Any], extras: dict[str, Any], workspace: Workspace, *, chat: bool = False
) -> str:
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

    if chat:
        # The only files a chat user can download are the ones collected from
        # here (core.chat_session). A file left anywhere else is made and then
        # unreachable — the screen gets no link for it.
        outputs = workspace.path / chat_session.OUTPUTS_DIR
        lines += [
            "",
            "## 사용자에게 전달할 파일",
            f"사용자가 내려받아야 하는 최종 파일은 `{outputs}/` 에 저장하세요. "
            "이 폴더에 둔 파일만 내려받기 링크로 전달됩니다. "
            "중간 작업 파일은 다른 곳에 두어도 됩니다.",
        ]

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
    return []


def _git_skills(row: dict[str, Any], extras: dict[str, Any]) -> dict[str, str]:
    raw = row.get("git_skills") or extras.get("git_skills")
    if isinstance(raw, str) and raw.strip():
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {str(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}
