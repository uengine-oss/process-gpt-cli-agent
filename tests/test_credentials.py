"""Each CLI runs on its tenant's own API key, and nobody else sees it.

Why it matters: a subscription token shared by every tenant is what the
licence does not allow; a company's own key is. So with a key, the run's CLI
must authenticate with *that* key — the pod's token removed, not just
outranked — and the key must not surface anywhere the run leaves a trace:
logs, the chat stream, the work-item timeline, the journal, the session
archive. Without a key, nothing about the run changes except one line saying
so.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import io
import json
import logging
import os
from pathlib import Path
from typing import Any

import pytest
from cliagents import ExecEvent, ExecEventKind, registry
from postgrest.exceptions import APIError

from core import credentials

KEY = "sk-ant-api03-TENANT-ACME-0123456789abcdef"
POD_TOKEN = "sk-ant-oat01-POD-SUBSCRIPTION-TOKEN"


def _stub_rpc(monkeypatch, answer):
    """Replace the RPC itself; returns the list of (tenant, provider) asked."""
    calls: list[tuple[str, str]] = []

    def _fetch(tenant, provider):
        calls.append((tenant, provider))
        if isinstance(answer, BaseException):
            raise answer
        return answer(tenant, provider) if callable(answer) else answer

    monkeypatch.setattr(credentials, "_fetch", _fetch)
    return calls


# --- what a run is started with -------------------------------------------------


def test_claude_code_with_a_tenant_key_drops_the_pods_token(monkeypatch, tmp_path):
    _stub_rpc(monkeypatch, KEY)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", POD_TOKEN)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "pod-gateway-bearer")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    creds = credentials.for_run("claude-code", "acme")
    assert creds.tenant_key and creds.provider == "anthropic"

    provider = registry.get("claude-code")
    env = creds.bind(provider).exec_env({"CLAUDE_CONFIG_DIR": str(tmp_path)})

    assert env["ANTHROPIC_API_KEY"] == KEY
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in env
    # Outranks ANTHROPIC_API_KEY in Claude Code; left in, the key is ignored.
    assert "ANTHROPIC_AUTH_TOKEN" not in env
    # The run's own overrides and the library's defaults are kept.
    assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path)
    assert env["NO_COLOR"] == "1"
    # Only this run's environment changed — not the process, which every
    # other tenant's run inherits.
    assert os.environ["CLAUDE_CODE_OAUTH_TOKEN"] == POD_TOKEN
    assert "ANTHROPIC_API_KEY" not in os.environ


def test_codex_with_a_tenant_key_gets_both_variables(monkeypatch, tmp_path):
    calls = _stub_rpc(monkeypatch, KEY)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-pod-openai")

    creds = credentials.for_run("codex", "acme")
    env = creds.bind(registry.get("codex")).exec_env({"CODEX_HOME": str(tmp_path)})

    assert calls == [("acme", "openai")]
    assert env["CODEX_API_KEY"] == KEY
    assert env["OPENAI_API_KEY"] == KEY
    assert env["CODEX_HOME"] == str(tmp_path)
    assert os.environ["OPENAI_API_KEY"] == "sk-pod-openai"


def test_without_a_key_the_run_is_exactly_todays(monkeypatch, tmp_path):
    _stub_rpc(monkeypatch, None)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", POD_TOKEN)

    creds = credentials.for_run("claude-code", "acme")
    provider = registry.get("claude-code")

    assert not creds.tenant_key and creds.reason == "not_set"
    assert creds.bind(provider) is provider  # the same object, not a wrapper
    assert creds.secrets == ()
    base = {"CLAUDE_CONFIG_DIR": str(tmp_path)}
    assert creds.bind(provider).exec_env(base) == provider.exec_env(base)
    line = creds.describe()
    assert "설정되지 않아" in line and "기본 자격 증명" in line


def test_a_run_without_a_tenant_or_for_another_cli_asks_nothing(monkeypatch):
    calls = _stub_rpc(monkeypatch, KEY)
    assert credentials.for_run("claude-code", "  ").reason == "no_tenant"
    assert credentials.for_run("gemini", "acme").reason == "unsupported"
    assert calls == []


def test_a_pasted_key_loses_its_trailing_newline(monkeypatch):
    class _Response:
        data = f"  {KEY}\n"

    _fake_client(monkeypatch, lambda: _Response())
    assert credentials.for_run("claude-code", "acme").env["ANTHROPIC_API_KEY"] == KEY


def test_the_key_is_not_in_the_repr():
    creds = credentials.RunCredentials(
        agent_id="claude-code", tenant_id="acme", env={"ANTHROPIC_API_KEY": KEY}
    )
    assert KEY not in repr(creds)
    assert KEY not in repr(creds.bind(registry.get("claude-code")))


# --- the lookup ------------------------------------------------------------------


def test_answers_are_cached_per_tenant_and_provider_then_expire(monkeypatch):
    calls = _stub_rpc(monkeypatch, lambda tenant, _p: KEY if tenant == "acme" else None)
    now = [1000.0]
    clock = lambda: now[0]  # noqa: E731

    assert credentials.lookup("acme", "anthropic", clock=clock) == KEY
    assert credentials.lookup("acme", "anthropic", clock=clock) == KEY
    assert credentials.lookup("acme", "openai", clock=clock) == KEY
    assert calls == [("acme", "anthropic"), ("acme", "openai")]

    now[0] += 61
    assert credentials.lookup("acme", "anthropic", clock=clock) == KEY
    assert calls[-1] == ("acme", "anthropic") and len(calls) == 3


def test_a_tenant_without_a_key_is_not_asked_about_on_every_run(monkeypatch):
    calls = _stub_rpc(monkeypatch, None)
    for _ in range(5):
        assert credentials.for_run("claude-code", "beta").reason == "not_set"
    assert calls == [("beta", "anthropic")]


class _Client:
    def __init__(self, respond):
        self._respond = respond
        self.calls: list[tuple[str, dict]] = []

    def rpc(self, name, params):
        self.calls.append((name, params))
        return self

    def execute(self):
        return self._respond()


def _fake_client(monkeypatch, respond) -> _Client:
    """The real ``_fetch`` against a stand-in for the SDK's Supabase client."""
    import processgpt_agent_sdk.database as database

    client = _Client(respond)
    monkeypatch.setenv("SUPABASE_URL", "http://supabase.invalid")
    monkeypatch.setattr(database, "initialize_db", lambda: None)
    monkeypatch.setattr(database, "get_db_client", lambda: client)
    # conftest stubbed `_fetch`; this test exercises the real one.
    monkeypatch.setattr(credentials, "_fetch", _REAL_FETCH)
    return client


_REAL_FETCH = credentials._fetch


def test_the_rpc_is_called_by_name_with_the_contracts_arguments(monkeypatch):
    class _Response:
        data = KEY

    client = _fake_client(monkeypatch, lambda: _Response())
    assert credentials.lookup("acme", "anthropic") == KEY
    assert client.calls == [
        ("cli_agent_credential_get", {"p_tenant": "acme", "p_provider": "anthropic"})
    ]


def test_a_database_without_the_function_means_no_key_logged_once(monkeypatch, caplog):
    """Production runs before the SQL is applied. That is not an error, and
    it must not cost a log line — or a round trip — per run."""

    def _missing():
        raise APIError(
            {
                "code": "PGRST202",
                "message": "Could not find the function public.cli_agent_credential_get",
            }
        )

    client = _fake_client(monkeypatch, _missing)
    caplog.set_level(logging.INFO, logger="core.credentials")

    first = credentials.for_run("claude-code", "acme")
    again = credentials.for_run("claude-code", "acme")
    other = credentials.for_run("codex", "beta")

    assert first.reason == again.reason == other.reason == "unavailable"
    assert not first.tenant_key
    assert "확인하지 못해" in first.describe()
    assert len(client.calls) == 2  # acme cached; beta/openai asked once
    notes = [r for r in caplog.records if "unavailable" in r.getMessage()]
    assert len(notes) == 1


@pytest.mark.parametrize("code", ["42883", "42P01", "3F000"])
def test_a_function_without_its_table_or_vault_is_also_no_key(monkeypatch, code):
    def _missing():
        raise APIError({"code": code, "message": "missing"})

    _fake_client(monkeypatch, _missing)
    assert credentials.for_run("claude-code", "acme").reason == "unavailable"


def test_an_unconfigured_database_is_no_key_without_asking(monkeypatch):
    monkeypatch.setattr(credentials, "_fetch", _REAL_FETCH)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY_URL", raising=False)
    assert credentials.for_run("claude-code", "acme").reason == "unavailable"


def test_a_failed_lookup_runs_on_the_pod_and_asks_again_next_time(monkeypatch, caplog):
    """Not cached: a blip must not keep a tenant that has a key off it."""

    def _denied():
        raise APIError({"code": "42501", "message": "permission denied for function"})

    client = _fake_client(monkeypatch, _denied)
    caplog.set_level(logging.WARNING, logger="core.credentials")

    assert credentials.for_run("claude-code", "acme").reason == "lookup_failed"
    assert credentials.for_run("claude-code", "acme").reason == "lookup_failed"
    assert len(client.calls) == 2
    assert any("42501" in r.getMessage() for r in caplog.records)


# --- the agent printing it ---------------------------------------------------------


def test_events_are_cleaned_wherever_the_key_appears():
    event = ExecEvent(
        kind=ExecEventKind.TOOL_END,
        text=f"ANTHROPIC_API_KEY={KEY}",
        tool="Bash",
        tool_output=[{"type": "text", "text": f"ANTHROPIC_API_KEY={KEY}"}],
        raw={"message": {"content": [{"content": KEY}]}},
    )
    clean = credentials.redact(event, (KEY,))
    assert KEY not in repr(clean)
    assert clean.text == f"ANTHROPIC_API_KEY={credentials.REDACTED}"
    assert clean.tool == "Bash" and clean.kind is ExecEventKind.TOOL_END
    # Nothing to clean: the same object back.
    untouched = ExecEvent(kind=ExecEventKind.RESULT, text="끝")
    assert credentials.redact(untouched, (KEY,)) is untouched
    assert credentials.redact(event, ()) is event


def test_the_workspace_is_scrubbed_but_not_emptied(tmp_path):
    transcript = tmp_path / ".agent-home/claude-code/projects/p/S-1.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text(json.dumps({"content": f"KEY={KEY}"}) + "\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("키 없음", encoding="utf-8")

    changed = credentials.scrub_workspace(tmp_path, (KEY,))

    assert changed == [transcript]
    line = json.loads(transcript.read_text(encoding="utf-8"))  # still valid JSON
    assert line["content"] == f"KEY={credentials.REDACTED}"
    assert (tmp_path / "notes.md").read_text(encoding="utf-8") == "키 없음"


# --- end to end: a run with a key leaves no trace of it ------------------------------


class _FakeCli:
    """A Claude Code child that does the worst thing an agent can do with a
    key: runs ``env``, repeats the key in its answer, and records all of it in
    its session transcript — as the real CLI would."""

    def __init__(self, env: dict[str, str]) -> None:
        self.env = env
        self.returncode = 0
        self.stderr = io.StringIO("")
        printed = f"HOME=/root\nANTHROPIC_API_KEY={env.get('ANTHROPIC_API_KEY', '')}\n"
        lines = [
            {"type": "system", "subtype": "init", "session_id": "S-1", "model": "m"},
            {
                "type": "assistant",
                "session_id": "S-1",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "T1",
                            "name": "Bash",
                            "input": {"command": "env"},
                        }
                    ]
                },
            },
            {
                "type": "user",
                "session_id": "S-1",
                "message": {
                    "content": [{"type": "tool_result", "tool_use_id": "T1", "content": printed}]
                },
            },
            {
                "type": "stream_event",
                "session_id": "S-1",
                "event": {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": f"키는 {KEY} 입니다. "},
                },
            },
            {
                "type": "result",
                "subtype": "success",
                "session_id": "S-1",
                "result": f"키는 {KEY} 입니다.",
            },
        ]
        self.stdout = io.StringIO(
            "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
        )

        home = Path(env["CLAUDE_CONFIG_DIR"])
        transcript = home / "projects" / "p" / "S-1.jsonl"
        transcript.parent.mkdir(parents=True, exist_ok=True)
        transcript.write_text(
            json.dumps({"type": "tool_result", "content": printed}) + "\n", encoding="utf-8"
        )

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        pass

    def kill(self):  # pragma: no cover
        pass


def _real_runner_with_fake_cli(monkeypatch) -> list[dict[str, Any]]:
    """The real runner and ``stream_exec``, with only the process faked."""
    import executor as ex
    from core import runner

    started: list[dict[str, Any]] = []

    def _popen(argv, **kwargs):
        started.append({"argv": argv, "env": kwargs.get("env")})
        return _FakeCli(kwargs["env"])

    monkeypatch.setattr(ex, "astream", functools.partial(runner.astream, popen=_popen))
    monkeypatch.setattr(ex, "require_runnable", lambda chosen: registry.get(chosen.agent_id))
    return started


def _no_key_anywhere(*, logs, events, root: Path, archive: dict[str, bytes] | None = None):
    for record in logs:
        assert KEY not in record.getMessage(), record.getMessage()
        assert KEY not in (record.exc_text or "")
    blob = json.dumps(events, ensure_ascii=False, default=str)
    assert KEY not in blob
    for path in root.rglob("*"):
        if path.is_file():
            assert KEY.encode() not in path.read_bytes(), path
    for name, body in (archive or {}).items():
        assert KEY.encode() not in body, name


def _archive_bucket(monkeypatch) -> dict[str, bytes]:
    from processgpt_agent_sdk import SessionArchive

    objects: dict[str, bytes] = {}

    class _Bucket:
        def upload(self, key, body, options=None):
            objects[key] = bytes(body)

        def download(self, key):
            return objects[key]

        def list(self, prefix):
            return []

    monkeypatch.setattr(SessionArchive, "_store", lambda self: _Bucket())
    return objects


def test_a_chat_turn_with_a_key_leaves_it_nowhere(tmp_path, monkeypatch, caplog):
    from processgpt_agent_sdk import ChatEventQueue, ChatRequest, ChatRequestContext
    from processgpt_agent_sdk.chat_mode import ChatStreamer
    from processgpt_agent_sdk.chat_registry import RunRecordingQueue

    import executor as ex
    from core import workspace as workspace_module
    from core.settings import settings

    caplog.set_level(logging.DEBUG)
    _stub_rpc(monkeypatch, KEY)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", POD_TOKEN)
    monkeypatch.setattr(
        workspace_module, "settings", dataclasses.replace(settings, workspace_root=tmp_path)
    )
    started = _real_runner_with_fake_cli(monkeypatch)
    archive = _archive_bucket(monkeypatch)

    request = ChatRequest(
        message="환경 변수를 보여 줘",
        tenant_id="acme",
        conversation_id="conv-1",
        metadata={"agent_cli": "claude-code"},
    )
    out: asyncio.Queue = RunRecordingQueue("conv-1")
    context = ChatRequestContext(request, streamer=ChatStreamer(out))
    asyncio.run(ex.CliAgentExecutor().execute(context, ChatEventQueue(out, request=request)))

    events = []
    while not out.empty():
        events.append(out.get_nowait())

    # The CLI really was started with the key, and without the pod's token.
    [run] = started
    assert run["env"]["ANTHROPIC_API_KEY"] == KEY
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in run["env"]
    assert "ANTHROPIC_API_KEY" not in os.environ

    # The turn went through, with the key cut out of the answer.
    done = [
        e["data"] for e in events if e.get("event") == "message" and e["data"].get("type") == "done"
    ]
    assert done and credentials.REDACTED in done[-1]["content"]

    # The transcript was archived — cleaned, not dropped.
    transcripts = [k for k in archive if k.endswith("S-1.jsonl")]
    assert transcripts and credentials.REDACTED.encode() in archive[transcripts[0]]

    _no_key_anywhere(logs=caplog.records, events=events, root=tmp_path, archive=archive)


class _WorkItemContext:
    """A work item as the SDK poller hands it to the executor."""

    task_id = "todo-1"
    context_id = "proc-1"

    def __init__(self) -> None:
        self.row = {
            "id": "todo-1",
            "tenant_id": "acme",
            "activity_name": "환경 점검",
            "description": "환경 점검",
            "query": "환경 변수를 확인하라",
        }
        self.metadata: dict[str, Any] = {"agent_cli": "claude-code"}

    def get_user_input(self) -> str:
        return ""

    def get_context_data(self) -> dict:
        return {"extras": {"request_kind": "process"}}


class _Queue:
    def __init__(self) -> None:
        self.sent: list = []

    async def enqueue_event(self, event):
        self.sent.append(event)


def _status_texts(queue: _Queue) -> list[str]:
    from google.protobuf.json_format import MessageToDict

    return [json.dumps(MessageToDict(e), ensure_ascii=False) for e in queue.sent]


def test_a_work_item_with_a_key_leaves_it_off_the_timeline(tmp_path, monkeypatch, caplog):
    import executor as ex
    from core import workspace as workspace_module
    from core.settings import settings

    caplog.set_level(logging.DEBUG)
    _stub_rpc(monkeypatch, KEY)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", POD_TOKEN)
    monkeypatch.setattr(
        workspace_module, "settings", dataclasses.replace(settings, workspace_root=tmp_path)
    )
    started = _real_runner_with_fake_cli(monkeypatch)
    queue = _Queue()

    asyncio.run(ex.CliAgentExecutor().execute(_WorkItemContext(), queue))

    [run] = started
    assert run["env"]["ANTHROPIC_API_KEY"] == KEY
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in run["env"]

    rows = _status_texts(queue)
    # The tool call and its (cleaned) output are on the card; the run finished.
    assert any("tool_usage_finished" in r and credentials.REDACTED in r for r in rows)
    assert any("task_completed" in r for r in rows)
    # With a key there is no "using the pod's credentials" line.
    assert not any("기본 자격 증명" in r for r in rows)

    _no_key_anywhere(logs=caplog.records, events=rows, root=tmp_path)


def test_a_work_item_without_a_key_says_so_once_and_runs_on_the_pod(tmp_path, monkeypatch, caplog):
    import executor as ex
    from core import workspace as workspace_module
    from core.settings import settings

    caplog.set_level(logging.INFO)
    _stub_rpc(monkeypatch, None)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", POD_TOKEN)
    monkeypatch.setattr(
        workspace_module, "settings", dataclasses.replace(settings, workspace_root=tmp_path)
    )
    started = _real_runner_with_fake_cli(monkeypatch)
    queue = _Queue()

    asyncio.run(ex.CliAgentExecutor().execute(_WorkItemContext(), queue))

    [run] = started
    # Today's environment: the pod's token, no key put in its place.
    assert run["env"]["CLAUDE_CODE_OAUTH_TOKEN"] == POD_TOKEN
    assert "ANTHROPIC_API_KEY" not in run["env"]

    rows = [r for r in _status_texts(queue) if "기본 자격 증명" in r]
    assert len(rows) == 1 and "task_working" in rows[0]
    assert any("기본 자격 증명" in r.getMessage() for r in caplog.records)
