"""Chat served from one runner Pod per conversation.

What the session router depends on, and what a reclaimed Pod must not take
with it: the standard chat routes, a truthful ``busy``, no work-item polling in
a runner, a refusal (not a substitution) when a chat names no CLI, answers the
reattach snapshot can see, and a conversation — files, session, download
links — that outlives the Pod's disk.
"""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("starlette", reason="requires the processgpt extra")
pytest.importorskip("httpx", reason="requires the processgpt extra")

from cliagents import ExecEvent, ExecEventKind, Permission, registry  # noqa: E402
from processgpt_agent_sdk import (  # noqa: E402
    ChatEventQueue,
    ChatRequest,
    ChatRequestContext,
    ProcessGPTAgentServer,
    SessionArchive,
    get_run_registry,
)
from processgpt_agent_sdk.artifacts import StoredArtifact  # noqa: E402
from processgpt_agent_sdk.chat_mode import ChatStreamer  # noqa: E402
from processgpt_agent_sdk.chat_registry import RunRecordingQueue  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

import executor as ex  # noqa: E402
import server as server_module  # noqa: E402
from core import chat_session, credentials, resume, selection  # noqa: E402
from core import workspace as workspace_module  # noqa: E402
from core.settings import Settings, settings  # noqa: E402
from core.workspace import Workspace  # noqa: E402

# --- HTTP surface -----------------------------------------------------------


@pytest.fixture
def client():
    server = ProcessGPTAgentServer(
        agent_executor=ex.CliAgentExecutor(), agent_type="cliagents", tenant_auth=True
    )
    return TestClient(server_module.build_app(server))


def test_the_standard_chat_routes_are_mounted_and_guarded(client):
    """The screen calls these paths for every chat runtime. With tenant auth
    on, a caller without a token is turned away before anything runs."""
    for path in ("/chat/stream", "/chat/stream/attach", "/chat/stop", "/chat/steer"):
        response = client.post(path, json={"conversation_id": "c1", "tenant_id": "acme"})
        assert response.status_code == 401, path


def test_steering_is_refused_as_unsupported_rather_than_ignored(client, monkeypatch):
    """No `steer()` on this executor: the SDK says so (501) instead of
    accepting a redirection that would never reach the CLI."""
    import processgpt_agent_sdk.chat_sse as chat_sse

    async def _authorized(request):
        request.state.tenant_id = "acme"
        request.state.user_id = "u1"

    async def _room_tenant(_conversation_id):
        return "acme"

    monkeypatch.setattr(chat_sse, "authorize_tenant", _authorized)
    monkeypatch.setattr(chat_sse, "fetch_chat_room_tenant_id", _room_tenant)

    response = client.post(
        "/chat/steer", json={"conversation_id": "c1", "tenant_id": "acme", "message": "방향 바꿔"}
    )
    assert response.status_code == 501
    assert response.json()["reason"] == "unsupported"


def test_health_keeps_its_fields_and_is_idle_when_nothing_runs(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["agent_type"] == "cliagents"
    assert {"runs_in_flight", "max_concurrent_runs", "process_polling"} <= body.keys()
    assert body["busy"] is False


def test_health_is_busy_while_a_chat_turn_runs(client, monkeypatch):
    class _Inflight:
        def active_conversation_ids(self):
            return ["conv-1"]

    monkeypatch.setattr(server_module, "get_inflight_registry", lambda: _Inflight())
    body = client.get("/health").json()
    assert body["busy"] is True
    assert body["active_turns"] == 1


def test_health_is_busy_while_a_work_item_or_an_upload_runs(client, monkeypatch):
    """Not only chat: a work item, or the session upload after a turn, is
    also something a reclaim would cut short."""
    monkeypatch.setattr(server_module, "active_runs", lambda: 1)
    assert client.get("/health").json()["busy"] is True


# --- polling switch -----------------------------------------------------------


def test_the_polling_switch_reads_false_and_defaults_to_on(monkeypatch):
    monkeypatch.delenv("CLIAGENTS_PROCESS_POLLING", raising=False)
    assert Settings().process_polling is True
    monkeypatch.setenv("CLIAGENTS_PROCESS_POLLING", "false")
    assert Settings().process_polling is False


@pytest.mark.parametrize("polling", [False, True])
def test_a_runner_pod_serves_http_without_claiming_work_items(monkeypatch, polling):
    """A runner that claimed a work item would take it down with it when the
    router reclaims the Pod for being idle."""
    import uvicorn

    started: list[str] = []

    async def _serve(_self, *args, **kwargs):
        started.append("http")

    async def _poll(_self):
        started.append("poller")

    async def _sweep():
        await asyncio.Event().wait()

    monkeypatch.setattr(server_module, "settings", dataclasses.replace(settings, process_polling=polling))
    monkeypatch.setattr(server_module.skill_store, "seed_system_skills", lambda: 0)
    monkeypatch.setattr(server_module, "_sweep_workspaces", _sweep)
    monkeypatch.setattr(uvicorn.Server, "serve", _serve)
    monkeypatch.setattr(ProcessGPTAgentServer, "run", _poll)

    asyncio.run(server_module.main())

    assert "http" in started
    assert ("poller" in started) is polling


# --- choosing the CLI in chat ---------------------------------------------------


def test_a_chat_that_names_no_cli_is_refused_not_given_the_default():
    with pytest.raises(selection.CliNotChosenError) as caught:
        selection.resolve({"tenant_id": "acme"}, require_choice=True)
    assert "claude-code" in caught.value.hint and "codex" in caught.value.hint


def test_a_chat_names_its_cli_model_and_permission_in_metadata():
    chosen = selection.resolve(
        {"agent_cli": "codex", "agent_model": "gpt-5", "agent_permission": "read-only"},
        require_choice=True,
    )
    assert (chosen.agent_id, chosen.model, chosen.permission) == (
        "codex",
        "gpt-5",
        Permission.READ_ONLY,
    )
    nested = selection.resolve({"agent_config": {"agentCli": "claude-code"}}, require_choice=True)
    assert nested.agent_id == "claude-code"


def test_a_work_item_still_inherits_the_configured_default():
    assert selection.resolve({}).agent_id == settings.default_agent_id


# --- a chat turn, end to end through the SDK's chat queue ----------------------


def _chat(tmp_path, monkeypatch, *, metadata: dict[str, Any], message: str = "한 문장으로 답해 줘"):
    """A real SDK chat context and queue around this executor."""
    monkeypatch.setattr(
        workspace_module, "settings", dataclasses.replace(settings, workspace_root=tmp_path)
    )
    monkeypatch.setattr(ex, "require_runnable", lambda chosen: registry.get(chosen.agent_id))
    # Chat runs only on a company key (core.credentials.require_company_key);
    # these tests are about the turn, so the tenant has one.
    monkeypatch.setattr(credentials, "_fetch", lambda _tenant, _provider: "sk-test-company-key")

    calls: dict[str, int] = {"restore": 0, "save": 0}

    async def _restore(*_a, **_k):
        calls["restore"] += 1
        return 0

    async def _save(*_a, **_k):
        calls["save"] += 1
        return 0

    monkeypatch.setattr(chat_session, "restore", _restore)
    monkeypatch.setattr(chat_session, "save", _save)

    request = ChatRequest(
        message=message, tenant_id="acme", conversation_id="conv-1", metadata=metadata
    )
    out: asyncio.Queue = RunRecordingQueue("conv-1")
    context = ChatRequestContext(request, streamer=ChatStreamer(out))
    queue = ChatEventQueue(out, request=request)
    return context, queue, out, calls


def _drain(out: asyncio.Queue) -> list[dict]:
    events: list[dict] = []
    while not out.empty():
        item = out.get_nowait()
        if item.get("event") == "message":
            events.append(item["data"])
    return events


def test_a_chat_without_a_cli_ends_with_an_error_the_screen_can_act_on(tmp_path, monkeypatch):
    context, queue, out, calls = _chat(tmp_path, monkeypatch, metadata={})
    ran: list[Any] = []

    async def _never(*args, **kwargs):
        ran.append(args)
        yield  # pragma: no cover

    monkeypatch.setattr(ex, "astream", _never)

    asyncio.run(ex.CliAgentExecutor().execute(context, queue))

    events = _drain(out)
    assert [e["type"] for e in events] == ["error"]
    assert events[0]["code"] == "cli_not_selected"
    assert "선택" in events[0]["content"]
    assert isinstance(events[0]["available"], list)
    assert ran == []  # nothing ran in its place
    assert calls["save"] == 0  # and nothing new was archived
    assert ex.active_runs() == 0


def test_an_empty_message_runs_nothing_and_touches_no_storage(tmp_path, monkeypatch):
    context, queue, out, calls = _chat(
        tmp_path, monkeypatch, metadata={"agent_cli": "claude-code"}, message="  "
    )

    asyncio.run(ex.CliAgentExecutor().execute(context, queue))
    events = _drain(out)

    assert [(e["type"], e["code"]) for e in events] == [("error", "empty_message")]
    assert calls == {"restore": 0, "save": 0}


def test_a_chat_turn_streams_tokens_the_reattach_snapshot_can_see(tmp_path, monkeypatch):
    """The SDK keeps `token` chunks for the snapshot a reattaching client gets
    first. Sent as `text`, a reload mid-answer would show an empty bubble."""
    context, queue, out, calls = _chat(tmp_path, monkeypatch, metadata={"agent_cli": "claude-code"})
    seen: dict[str, Any] = {}

    async def _fake(provider, request, env=None, timeout=None):
        seen["prompt"] = request.prompt
        seen["resume"] = request.resume_session
        seen["busy"] = ex.active_runs()
        yield ExecEvent(kind=ExecEventKind.RUN_START, session_id="S-1")
        yield ExecEvent(kind=ExecEventKind.ASSISTANT_TEXT, text="안녕", session_id="S-1")
        yield ExecEvent(kind=ExecEventKind.ASSISTANT_TEXT, text="하세요.", session_id="S-1")
        subscription = await get_run_registry().subscribe("conv-1")
        seen["snapshot"] = subscription[1] if subscription else None
        yield ExecEvent(kind=ExecEventKind.RESULT, text="안녕하세요.", session_id="S-1")

    monkeypatch.setattr(ex, "astream", _fake)

    async def _turn():
        await get_run_registry().start_run("conv-1", out_queue=out)
        await ex.CliAgentExecutor().execute(context, queue)

    asyncio.run(_turn())

    events = _drain(out)
    assert seen["prompt"] == "한 문장으로 답해 줘"  # the user's words, verbatim
    assert seen["resume"] is None
    assert seen["busy"] == 1
    assert seen["snapshot"] == "안녕하세요."
    assert "text" not in {e["type"] for e in events}
    assert events[-1] == {"type": "done", "content": "안녕하세요."}
    assert calls == {"restore": 1, "save": 1}

    # The next turn continues the same CLI session.
    assert resume.saved_session(tmp_path / "acme" / "conv-1", agent_id="claude-code") == "S-1"


def test_the_next_turn_resumes_the_session_but_not_another_clis(tmp_path):
    resume.remember_session(tmp_path, "S-claude", agent_id="claude-code")
    assert resume.plan_chat(tmp_path, " 계속 ", agent_id="claude-code").session_id == "S-claude"
    assert resume.plan_chat(tmp_path, "계속", agent_id="codex").session_id is None
    assert resume.plan_chat(tmp_path, " 계속 ", agent_id="claude-code").prompt == "계속"


def test_a_chat_turn_hands_its_outputs_over_as_links(tmp_path, monkeypatch):
    """`pdfFiles` on the final event becomes `done.files` and is stored on the
    chat message — the links outlive the Pod that made the files."""
    context, queue, out, _calls = _chat(tmp_path, monkeypatch, metadata={"agent_cli": "codex"})
    link = {"url": "https://signed/report.md", "fileUrl": "https://signed/report.md", "name": "report.md"}

    async def _publish(workspace, **_kwargs):
        return chat_session.Published(files=[link])

    async def _fake(provider, request, env=None, timeout=None):
        yield ExecEvent(
            kind=ExecEventKind.RESULT, text="만들었습니다: /workspace/acme/conv-1/outputs/report.md"
        )

    monkeypatch.setattr(chat_session, "publish", _publish)
    monkeypatch.setattr(ex, "astream", _fake)

    asyncio.run(ex.CliAgentExecutor().execute(context, queue))

    done = _drain(out)[-1]
    assert done["type"] == "done"
    assert done["files"] == [link]
    assert "[report.md](https://signed/report.md)" in done["content"]
    assert "/workspace/" not in done["content"]


def test_an_output_without_a_link_is_said_in_the_answer(tmp_path, monkeypatch):
    context, queue, out, _calls = _chat(tmp_path, monkeypatch, metadata={"agent_cli": "claude-code"})

    async def _publish(workspace, **_kwargs):
        return chat_session.Published(unlinked=["report.md"], reason="MEMENTO_BASE_URL is not set")

    async def _fake(provider, request, env=None, timeout=None):
        yield ExecEvent(kind=ExecEventKind.RESULT, text="완료")

    monkeypatch.setattr(chat_session, "publish", _publish)
    monkeypatch.setattr(ex, "astream", _fake)

    asyncio.run(ex.CliAgentExecutor().execute(context, queue))

    done = _drain(out)[-1]
    assert "report.md" in done["content"] and "링크" in done["content"]


# --- what survives the Pod ------------------------------------------------------


class _Bucket:
    """Supabase Storage, as much of it as the SDK archive touches."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def upload(self, key, body, options=None):
        self.objects[key] = bytes(body)

    def download(self, key):
        return self.objects[key]

    def list(self, prefix):
        prefix = prefix.rstrip("/") + "/"
        entries: dict[str, dict] = {}
        for key in self.objects:
            if not key.startswith(prefix):
                continue
            head, _, rest = key[len(prefix):].partition("/")
            entries[head] = {"name": head, "id": None if rest else "object"}
        return list(entries.values())


def _write(root: Path, relative: str, text: str = "x") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_a_reclaimed_pods_conversation_comes_back_on_a_new_one(tmp_path, monkeypatch):
    bucket = _Bucket()
    monkeypatch.setattr(SessionArchive, "_store", lambda self: bucket)

    old = Workspace(path=tmp_path / "old" / "acme" / "conv-1", run_id="conv-1")
    _write(old.path, "notes.md", "메모")
    _write(old.path, "outputs/report.md", "보고서")
    _write(old.path, resume.SESSION_FILENAME, '{"session_id": "S-1", "agent_id": "claude-code"}')
    _write(old.path, ".agent-home/claude-code/projects/p/S-1.jsonl", "transcript")
    _write(old.path, ".agent-home/claude-code/.credentials.json", "secret")
    _write(old.path, ".agent-home/claude-code/.claude.json", '{"oauthAccount": {}}')
    _write(old.path, ".agent-home/claude-code/backups/.claude.json.backup.1", "{}")
    _write(old.path, ".agent-home/claude-code/agents/worker.md", "TOKEN=secret")
    _write(old.path, ".agent-home/codex/sessions/rollout.jsonl", "transcript")
    _write(old.path, ".agent-home/codex/auth.json", "secret")
    _write(old.path, ".agent-home/codex/config.toml", "TOKEN=secret")

    saved = asyncio.run(chat_session.save(old, tenant_id="acme", conversation_id="conv-1"))
    # notes, report, session record, and one transcript per CLI.
    assert saved == 5
    stored = {key.removeprefix("cliagents/acme/conv-1/") for key in bucket.objects}
    assert stored == {
        "workspace/notes.md",
        "workspace/outputs/report.md",
        f"workspace/{resume.SESSION_FILENAME}",
        "home-claude-code/projects/p/S-1.jsonl",
        "home-codex/sessions/rollout.jsonl",
    }

    new = Workspace(path=tmp_path / "new" / "acme" / "conv-1", run_id="conv-1")
    new.path.mkdir(parents=True)
    asyncio.run(chat_session.restore(new, tenant_id="acme", conversation_id="conv-1"))

    assert (new.path / "notes.md").read_text(encoding="utf-8") == "메모"
    assert (new.path / "outputs/report.md").read_text(encoding="utf-8") == "보고서"
    assert resume.saved_session(new.path, agent_id="claude-code") == "S-1"
    assert (new.path / ".agent-home/claude-code/projects/p/S-1.jsonl").is_file()
    assert (new.path / ".agent-home/codex/sessions/rollout.jsonl").is_file()
    assert not (new.path / ".agent-home/claude-code/.credentials.json").exists()


def test_a_failed_restore_starts_the_turn_rather_than_failing_it(tmp_path, monkeypatch):
    def _down(self):
        raise RuntimeError("storage unreachable")

    monkeypatch.setattr(SessionArchive, "_store", _down)
    ws = Workspace(path=tmp_path, run_id="conv-1")
    assert asyncio.run(chat_session.restore(ws, tenant_id="acme", conversation_id="conv-1")) == 0
    assert asyncio.run(chat_session.save(ws, tenant_id="acme", conversation_id="conv-1")) == 0


def test_only_this_turns_outputs_get_links(tmp_path, monkeypatch):
    stored: list[str] = []

    class _Store:
        def __init__(self, _base_url):
            pass

        async def put(self, *, tenant_id, conversation_id, path, content_type, index=True):
            assert index is False
            stored.append(path.name)
            return StoredArtifact(file_id=f"artifacts/{path.name}", url=f"https://signed/{path.name}")

    monkeypatch.setattr(chat_session, "settings", dataclasses.replace(settings, memento_base_url="http://memento"))
    monkeypatch.setattr(chat_session, "MementoArtifactStore", _Store)

    ws = Workspace(path=tmp_path, run_id="conv-1")
    _write(ws.path, "outputs/earlier.md", "지난 턴")
    before = chat_session.before_turn(ws)
    _write(ws.path, "outputs/Report.md", "이번 턴")
    _write(ws.path, "outputs/script.py", "print(1)")

    published = asyncio.run(
        chat_session.publish(ws, tenant_id="acme", conversation_id="conv-1", before=before)
    )
    assert stored == ["Report.md"]
    assert [f["name"] for f in published.files] == ["Report.md"]
    assert published.files[0]["url"] == "https://signed/Report.md"
    # Produced but not a deliverable format: named, not silently dropped.
    assert published.unlinked == ["script.py"]


def test_without_memento_outputs_are_named_as_unlinked(tmp_path, monkeypatch):
    monkeypatch.setattr(chat_session, "settings", dataclasses.replace(settings, memento_base_url=""))
    ws = Workspace(path=tmp_path, run_id="conv-1")
    before = chat_session.before_turn(ws)
    _write(ws.path, "outputs/report.md", "x")

    published = asyncio.run(
        chat_session.publish(ws, tenant_id="acme", conversation_id="conv-1", before=before)
    )
    assert published.files == []
    assert published.unlinked == ["report.md"]
    assert "MEMENTO_BASE_URL" in published.reason


def test_a_chat_cannot_widen_its_own_permission():
    # command_exec gives the CLI a shell, and the shell inherits the pod's
    # environment. The chat user may pick the CLI and model, not this.
    from core.selection import chat_choice, resolve
    from core.settings import settings

    extras = {
        "agent_cli": "codex",
        "agent_permission": "command_exec",
        "metadata": {"agentPermission": "command_exec", "agent_model": "gpt-x"},
        "agent_config": {"permission": "command_exec"},
    }
    picked = chat_choice(extras)
    assert "agent_permission" not in picked
    assert "agentPermission" not in picked["metadata"]
    assert "permission" not in picked["agent_config"]
    selection = resolve(picked, require_choice=True)
    assert selection.agent_id == "codex"
    assert selection.model == "gpt-x"
    assert selection.permission == settings.default_permission
