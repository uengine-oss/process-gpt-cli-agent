"""The HTTP surface, exercised through a real ASGI client.

Skipped wholesale when the `processgpt` extra is not installed — the library
suite must stay runnable in a checkout that never asked for a web framework.
"""

from __future__ import annotations

import dataclasses

import pytest

pytest.importorskip("starlette", reason="requires the processgpt extra")
pytest.importorskip("httpx", reason="requires the processgpt extra")

from starlette.applications import Starlette  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from api.availability import routes as availability_routes  # noqa: E402
from api.files import routes as file_routes  # noqa: E402
from api.replay import routes as replay_routes  # noqa: E402
from core import workspace as workspace_module  # noqa: E402
from core.journal import Journal  # noqa: E402
from core.settings import settings  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(
        workspace_module,
        "settings",
        dataclasses.replace(settings, workspace_root=tmp_path),
    )
    app = Starlette(routes=[*availability_routes, *file_routes, *replay_routes])
    return TestClient(app)


# --- availability ---------------------------------------------------------


def test_the_picker_learns_which_agents_exist_here(client):
    body = client.get("/agents").json()
    ids = {a["agent_id"] for a in body["agents"]}
    assert {"claude-code", "codex"} <= ids
    for agent in body["agents"]:
        assert "installed" in agent
        # An uninstalled agent must arrive with a way to fix that.
        assert agent["installed"] or agent["install_hint"]


def test_auth_state_is_unknown_rather_than_false_when_not_probed(client):
    body = client.get("/agents").json()
    assert all(a["authenticated"] is None for a in body["agents"])


# --- files ----------------------------------------------------------------


def test_a_run_lists_its_own_files_and_not_its_bookkeeping(client, tmp_path):
    ws = workspace_module.for_run("run-1", tenant_id="acme")
    (ws.path / "report.md").write_text("결과")
    Journal(ws.path).record_tool("office/create", {})

    body = client.get("/runs/run-1/files", params={"tenant_id": "acme"}).json()
    assert [f["path"] for f in body["files"]] == ["report.md"]


def test_a_download_returns_the_file(client):
    ws = workspace_module.for_run("run-2", tenant_id="acme")
    (ws.path / "report.md").write_text("결과입니다")

    response = client.get(
        "/runs/run-2/download", params={"tenant_id": "acme", "path": "report.md"}
    )
    assert response.status_code == 200
    assert response.text == "결과입니다"


def test_a_path_that_climbs_out_is_refused(client, tmp_path):
    workspace_module.for_run("run-3", tenant_id="acme")
    (tmp_path / "other-tenant-secret.txt").write_text("전략 문서")

    response = client.get(
        "/runs/run-3/download",
        params={"tenant_id": "acme", "path": "../../other-tenant-secret.txt"},
    )
    assert response.status_code == 404
    assert "전략 문서" not in response.text


def test_an_expired_run_says_so_rather_than_serving_an_empty_list(client, tmp_path):
    # A run whose workspace was swept: for_run would recreate the directory, so
    # the route must answer about the request, not about what it just made.
    response = client.get("/runs/never-existed/files", params={"tenant_id": "acme"})
    assert response.status_code == 404


# --- replay ---------------------------------------------------------------


def test_the_journal_reports_whether_a_run_can_be_fully_replayed(client):
    ws = workspace_module.for_run("run-4", tenant_id="acme")
    journal = Journal(ws.path)
    (ws.path / "a.txt").write_text("v1")
    journal.record_file(ws.path / "a.txt", "created", workspace_path=ws.path)

    body = client.get("/runs/run-4/journal", params={"tenant_id": "acme"}).json()
    assert body["fully_replayable"] is True

    journal.note_out_of_scope("워크스페이스 밖 경로 변경")
    body = client.get("/runs/run-4/journal", params={"tenant_id": "acme"}).json()
    assert body["fully_replayable"] is False
    assert body["limitations"]


def test_replay_reapplies_recorded_files_without_a_model(client):
    ws = workspace_module.for_run("run-5", tenant_id="acme")
    journal = Journal(ws.path)
    target = ws.path / "a.txt"
    target.write_text("agent output")
    journal.record_file(target, "created", workspace_path=ws.path)

    target.write_text("edited by someone else")
    body = client.post("/runs/run-5/replay", params={"tenant_id": "acme"}).json()

    assert body["applied"] == ["a.txt"]
    assert target.read_text() == "agent output"


def test_undo_names_what_it_could_not_take_back(client):
    ws = workspace_module.for_run("run-6", tenant_id="acme")
    journal = Journal(ws.path)
    journal.record_tool("office/create_document", {"title": "계약서"}, external=True)

    body = client.post("/runs/run-6/undo", params={"tenant_id": "acme"}).json()
    assert any("office/create_document" in item for item in body["irreversible"])
