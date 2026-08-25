"""The decisions that would be wrong silently: selection, isolation, pausing,
containment, replay scope."""

from __future__ import annotations

import dataclasses
import os
import time

import pytest
from cliagents import ExecEvent, ExecEventKind, Permission, registry

from core import bridge, events, hitl, journal, selection, subagents, workspace
from core.settings import settings


def _with_settings(monkeypatch, module, **overrides):
    """Point one module at a modified copy of the settings.

    Settings are frozen in production — configuration that mutates at runtime
    is a class of bug nobody wants — so a test swaps the binding rather than
    the fields.
    """
    monkeypatch.setattr(module, "settings", dataclasses.replace(settings, **overrides))

# --- selection ------------------------------------------------------------


def test_a_missing_cli_choice_falls_back_to_the_configured_default():
    chosen = selection.resolve({"activity_name": "x"})
    assert chosen.agent_id == settings.default_agent_id
    assert chosen.permission is Permission.WORKSPACE_WRITE


def test_the_cli_choice_is_read_from_a_nested_settings_blob_too():
    """The UI writes it next to the orchestration on one path, inside a config
    object on another. Reading only one makes the other silently do nothing."""
    chosen = selection.resolve({"agent_config": '{"agent_cli": "codex", "model": "gpt-5"}'})
    assert chosen.agent_id == "codex"
    assert chosen.model == "gpt-5"


def test_default_permission_is_the_safe_one_not_the_convenient_one():
    chosen = selection.resolve({"agent_permission": "nonsense-value"})
    assert chosen.permission is Permission.WORKSPACE_WRITE


def test_permission_aliases_from_the_ui_are_honoured():
    assert selection.resolve({"permission": "read-only"}).permission is Permission.READ_ONLY
    assert selection.resolve({"permission": "full"}).permission is Permission.COMMAND_EXEC


def test_an_unknown_cli_refuses_instead_of_substituting():
    with pytest.raises(selection.CliSelectionError) as caught:
        selection.require_runnable(selection.Selection("gemini-cli", None, Permission.READ_ONLY))
    assert caught.value.agent_id == "gemini-cli"
    assert "available" in caught.value.hint


def test_a_missing_cli_reports_how_to_install_it(monkeypatch):
    provider = registry.get("codex")
    monkeypatch.setattr(provider, "resolve_executable", _raise_not_installed(provider))

    with pytest.raises(selection.CliSelectionError) as caught:
        selection.require_runnable(selection.Selection("codex", None, Permission.READ_ONLY))
    assert "npm install" in caught.value.hint


def _raise_not_installed(provider):
    from cliagents import AgentNotInstalledError

    def _raise(**_kwargs):
        raise AgentNotInstalledError(provider.id, provider.executable, provider.install_hint)

    return _raise


# --- workspace containment ------------------------------------------------


def test_a_workspace_refuses_paths_that_climb_out(tmp_path):
    ws = workspace.Workspace(path=tmp_path / "run", run_id="run")
    ws.path.mkdir()

    assert ws.contains(ws.path / "a.txt")
    assert not ws.contains(tmp_path / "elsewhere.txt")
    with pytest.raises(PermissionError):
        ws.resolve_within("../elsewhere.txt")


def test_containment_follows_symlinks_rather_than_text(tmp_path):
    """A path that *looks* inside but resolves outside is outside."""
    ws = workspace.Workspace(path=tmp_path / "run", run_id="run")
    ws.path.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("other tenant's draft")
    (ws.path / "link.txt").symlink_to(outside)

    assert not ws.contains(ws.path / "link.txt")


def test_run_directories_are_separated_per_tenant(tmp_path, monkeypatch):
    _with_settings(monkeypatch, workspace, workspace_root=tmp_path)
    a = workspace.for_run("todo-1", tenant_id="tenant-a")
    b = workspace.for_run("todo-1", tenant_id="tenant-b")
    assert a.path != b.path
    assert not a.contains(b.path)


def test_a_run_id_cannot_smuggle_a_path_out_of_the_root(tmp_path, monkeypatch):
    _with_settings(monkeypatch, workspace, workspace_root=tmp_path)
    ws = workspace.for_run("../../etc", tenant_id="t")
    assert tmp_path.resolve() in ws.path.resolve().parents


def test_the_sweep_keeps_recent_runs_and_removes_stale_ones(tmp_path, monkeypatch):
    _with_settings(monkeypatch, workspace, workspace_root=tmp_path, workspace_retention_hours=1)

    fresh = workspace.for_run("fresh", tenant_id="t")
    stale = workspace.for_run("stale", tenant_id="t")
    (fresh.path / "a.txt").write_text("a")
    (stale.path / "a.txt").write_text("a")

    old = time.time() - 7200
    for path in (stale.path, stale.path / "a.txt"):
        os.utime(path, (old, old))

    removed = workspace.sweep()
    assert stale.path in removed
    assert fresh.path.exists()


# --- MCP isolation --------------------------------------------------------


def test_a_globally_scoped_agent_gets_a_private_config_home(tmp_path):
    provider = registry.get("codex")
    env = bridge._isolate_config_home(provider, tmp_path)
    assert env["CODEX_HOME"].startswith(str(tmp_path))


def test_a_project_scoped_agent_needs_no_isolation(tmp_path):
    provider = registry.get("claude-code")
    assert bridge._isolate_config_home(provider, tmp_path) == {}


def test_claude_runtime_uses_an_isolated_user_config_to_avoid_project_trust(tmp_path):
    provider = registry.get("claude-code")
    env = bridge.runtime_env(provider, tmp_path)
    assert env["CLAUDE_CONFIG_DIR"].startswith(str(tmp_path))


def test_an_unisolatable_global_agent_refuses_rather_than_share_tools(tmp_path):
    """Sharing a config home between tenants would leak tools and credentials."""

    class _GlobalStranger:
        id = "stranger"
        mcp_scope = bridge.ConfigScope.USER_GLOBAL

    with pytest.raises(bridge.BridgeIsolationError):
        bridge._isolate_config_home(_GlobalStranger(), tmp_path)


def test_tenant_mcp_config_is_read_as_data():
    servers = bridge.processgpt_servers(
        tenant_mcp={"mcpServers": {"office": {"command": "npx", "args": ["office-mcp"]}}},
        extra_env={"TENANT": "acme"},
    )
    assert [s.name for s in servers] == ["office"]
    assert servers[0].env["TENANT"] == "acme"


def test_each_native_subagent_gets_only_its_declared_tenant_server():
    servers = bridge.processgpt_servers(
        tenant_mcp={
            "mcpServers": {
                "office": {"command": "office"},
                "hr": {"command": "hr"},
            }
        }
    )
    agents = subagents.prepare(
        [{"name": "Reviewer", "tools": "office", "skills": "expense-policy"}],
        tenant_servers=servers,
        fallback_skills=[],
        fallback_tools=[],
    )
    assert [server.name for server in agents[0].servers] == ["office"]
    assert agents[0].skills == ["expense-policy"]


def test_no_explicit_agent_creates_an_activity_worker():
    servers = bridge.processgpt_servers(
        tenant_mcp={"mcpServers": {"office": {"command": "office"}}}
    )
    agents = subagents.prepare(
        [], tenant_servers=servers, fallback_skills=["policy"], fallback_tools=[]
    )
    assert [agent.name for agent in agents] == ["processgpt-worker"]
    assert [server.name for server in agents[0].servers] == ["office"]


def test_synthesized_worker_preloads_default_discovered_skills():
    agents = subagents.prepare(
        [], tenant_servers=[], fallback_skills=[], fallback_tools=[]
    )
    updated = subagents.preload_discovered_skills(agents, ["system", "tenant-policy"])
    assert updated[0].skills == ["system", "tenant-policy"]


# --- pausing for a human --------------------------------------------------


def test_the_same_question_asked_twice_notifies_once(tmp_path):
    request = hitl.PendingRequest(
        run_id="r",
        agent_id="claude-code",
        session_id="s1",
        question="Bash 실행을 허용하시겠습니까?",
        tool="Bash",
        fingerprint=hitl.fingerprint("Bash", "Bash 실행을 허용하시겠습니까?"),
    )
    assert hitl.remember(tmp_path, request) is True
    assert hitl.remember(tmp_path, request) is False


def test_a_pause_survives_a_restart(tmp_path):
    request = hitl.PendingRequest(
        run_id="r", agent_id="codex", session_id="thread-9", question="계속할까요?"
    )
    hitl.remember(tmp_path, request)

    recalled = hitl.recall(tmp_path)
    assert recalled is not None
    assert recalled.session_id == "thread-9"


def test_an_answer_resumes_the_recorded_session(tmp_path):
    hitl.remember(
        tmp_path,
        hitl.PendingRequest(run_id="r", agent_id="codex", session_id="thread-9", question="?"),
    )
    plan = hitl.plan_resume(tmp_path, "승인합니다", workspace_exists=True)

    assert plan.session_id == "thread-9"
    assert not plan.restarted
    assert "이미 완료한 작업은 다시 하지 마세요" in plan.prompt


def test_a_lost_workspace_restarts_loudly_rather_than_quietly(tmp_path):
    plan = hitl.plan_resume(
        tmp_path, "승인합니다", workspace_exists=False, previous_summary="초안까지 작성함"
    )
    assert plan.restarted
    assert plan.reason
    assert "초안까지 작성함" in plan.prompt


# --- UI translation -------------------------------------------------------


def test_tool_events_keep_the_tool_name_the_ui_shows():
    started = events.translate(
        ExecEvent(kind=ExecEventKind.TOOL_START, tool="Write", tool_use_id="t1")
    )
    assert started[0].type == "tool_start"
    assert started[0].data["tool"] == "Write"


def test_an_unknown_event_still_reaches_the_user():
    produced = events.translate(
        ExecEvent(kind=ExecEventKind.UNKNOWN, raw={"type": "brand.new"}, text="")
    )
    assert produced and produced[0].type == "agent_log"


def test_empty_text_produces_no_bubble():
    assert events.translate(ExecEvent(kind=ExecEventKind.ASSISTANT_TEXT, text="")) == []


def test_a_large_file_preview_is_truncated_and_says_so(monkeypatch, tmp_path):
    _with_settings(monkeypatch, events, file_preview_max_bytes=10)
    ws = workspace.Workspace(path=tmp_path, run_id="r")
    produced = events.translate(
        ExecEvent(
            kind=ExecEventKind.FILE_CHANGE,
            path=str(tmp_path / "big.txt"),
            change="created",
            text="x" * 100,
        ),
        workspace=ws,
    )
    assert produced[0].data["truncated"] is True
    assert len(produced[0].data["content"]) == 10
    assert produced[0].data["path"] == "big.txt"


# --- replay scope ---------------------------------------------------------


def test_file_changes_replay_without_calling_a_model(tmp_path):
    log = journal.Journal(tmp_path)
    target = tmp_path / "a.txt"

    target.write_text("v1")
    log.record_file(target, "created", workspace_path=tmp_path)
    target.write_text("v2")
    log.record_file(target, "modified", workspace_path=tmp_path)

    target.write_text("someone else's edit")
    applied = log.replay_files(tmp_path)

    assert applied == ["a.txt", "a.txt"]
    assert target.read_text() == "v2"


def test_undo_restores_a_pre_existing_file_instead_of_deleting_it(tmp_path):
    """File changes are learned about after the fact, so without a baseline the
    first edit of an existing file looks like a creation — and undo deletes it."""
    target = tmp_path / "a.txt"
    target.write_text("original")

    log = journal.Journal(tmp_path)
    assert log.capture_baseline(tmp_path) == 1

    target.write_text("changed")
    log.record_file(target, "modified", workspace_path=tmp_path)

    restored, _ = log.undo_files(tmp_path)
    assert target.exists(), "undo deleted a file the run only modified"
    assert target.read_text() == "original"
    assert restored


def test_undo_removes_a_file_the_run_created(tmp_path):
    log = journal.Journal(tmp_path)
    target = tmp_path / "new.txt"
    target.write_text("created by the agent")
    log.record_file(target, "created", workspace_path=tmp_path)

    log.undo_files(tmp_path)
    assert not target.exists()


def test_an_external_tool_call_is_named_as_irreversible(tmp_path):
    log = journal.Journal(tmp_path)
    log.record_tool("office/create_document", {"title": "x"}, external=True)

    _restored, irreversible = log.undo_files(tmp_path)
    assert any("office/create_document" in item for item in irreversible)


def test_a_run_that_reached_outside_reports_itself_as_partial(tmp_path):
    log = journal.Journal(tmp_path)
    log.record_file(tmp_path / "a.txt", "created", workspace_path=tmp_path)
    assert log.replayable

    log.note_out_of_scope("워크스페이스 밖 경로 수정")
    assert not log.replayable
    assert log.limitations() == ["워크스페이스 밖 경로 수정"]


# --- activity settings ----------------------------------------------------


def test_an_activitys_skills_and_cli_come_from_the_process_definition():
    """`todolist` has no column for either — a service that reads only the row
    ignores every choice made on the designer screen and quietly uses defaults."""
    from core import activity

    definition = {
        "activities": [
            {"id": "other", "skills": ["nope"]},
            {
                "id": "review",
                "skills": ["hwpx-writer", "expense-policy"],
                "tools": ["office"],
                "agentConfig": {"agent_cli": "codex", "model": "gpt-5"},
            },
        ]
    }

    declared = activity.extract(definition, "review")
    assert declared.skills == ["hwpx-writer", "expense-policy"]
    assert declared.tools == ["office"]
    assert declared.agent_config["agent_cli"] == "codex"


def test_a_freshly_generated_definition_is_read_too():
    """The generation skill writes `elements`; the engine stores `activities`.
    Reading only one leaves the other silently without skills."""
    from core import activity

    declared = activity.extract(
        {"elements": [{"id": "apply", "skills": "a, b"}]}, "apply"
    )
    assert declared.skills == ["a", "b"]


def test_an_unknown_activity_declares_nothing_rather_than_guessing():
    from core import activity

    assert activity.extract({"activities": [{"id": "x"}]}, "y") == activity.Capabilities()
    assert activity.extract(None, "x") == activity.Capabilities()


# --- permission refusals --------------------------------------------------


def test_the_run_instructions_keep_the_agent_inside_its_workspace():
    """The agent went looking for a skill in the machine's home directory, was
    refused (outside the workspace), and stopped — with the same skill already
    provisioned into the project."""
    from pathlib import Path

    import executor
    from core.workspace import Workspace

    text = executor._instructions({}, {}, Workspace(path=Path("/ws/run-1"), run_id="run-1"))

    assert "작업 디렉터리 밖의 파일을 읽거나 수정하지 마세요" in text
    assert ".claude" in text


def test_a_run_announces_itself_so_the_monitor_has_a_card():
    """The panel builds its timeline from `task_started` and applies
    `task_completed` only to a job it already knows. Without the announcement
    the result is stored and the screen still says the job is queued."""
    import asyncio
    import json as _json

    from a2a.helpers import get_message_text
    from a2a.types import TaskStatusUpdateEvent
    from google.protobuf.json_format import MessageToDict

    from executor import CliAgentExecutor

    class _Queue:
        def __init__(self):
            self.events = []

        async def enqueue_event(self, event):
            self.events.append(event)

    class _Provider:
        display_name = "Claude Code"

    queue = _Queue()
    asyncio.run(
        CliAgentExecutor()._started(
            queue,
            task_id="t",
            context_id="c",
            row={"activity_name": "담당 부서 배정", "query": "민원을 분류하세요"},
            provider=_Provider(),
        )
    )

    assert len(queue.events) == 1
    event = queue.events[0]
    assert isinstance(event, TaskStatusUpdateEvent)
    metadata = MessageToDict(event.metadata, preserving_proto_field_name=True)
    assert metadata["event_type"] == "task_started"
    # The card's headline and body come from here.
    body = _json.loads(get_message_text(event.status.message))
    assert body["goal"] == "담당 부서 배정"
    assert body["name"] == "Claude Code"
