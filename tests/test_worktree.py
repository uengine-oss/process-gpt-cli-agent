"""A work item on a repository: a worktree of a local path, a clone of a URL.

Process GPT Lite hands a shared folder over as a URL (its git smart HTTP
endpoint) because the folder's repository is on Lite's volume, not on this
pod's. These tests use a bare repository behind a ``file://`` URL for the
clone and push mechanics; the HTTP side (the token, what may be pushed) is
tested in Lite and in the cross test that runs this module against a Lite
server.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest
from cliagents import ExecEvent, ExecEventKind

from core import worktree
from core.worktree import Settings, WorktreeError, prepare, settings_from


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.test", *args],
        cwd=str(cwd), check=True, capture_output=True, text=True,
    ).stdout


def _origin(tmp_path: Path) -> tuple[Path, Path]:
    """A bare 'server' repository with main and one more branch, and a working copy of it."""
    bare = tmp_path / "server" / "folder.git"
    bare.parent.mkdir(parents=True)
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(bare))
    seed = tmp_path / "seed"
    git(tmp_path, "clone", "-q", str(bare), str(seed))
    (seed / "notice.md").write_text("# 안내문\n초안\n", encoding="utf-8")
    git(seed, "add", "-A")
    git(seed, "commit", "-q", "-m", "first")
    git(seed, "push", "-q", "origin", "HEAD:refs/heads/main")
    git(seed, "checkout", "-q", "-b", "inst/abc123")
    (seed / "line.md").write_text("instance line\n", encoding="utf-8")
    git(seed, "add", "-A")
    git(seed, "commit", "-q", "-m", "on the line")
    git(seed, "push", "-q", "origin", "HEAD:refs/heads/inst/abc123")
    return bare, seed


def _branches(bare: Path) -> list[str]:
    return git(bare, "for-each-ref", "--format=%(refname:short)", "refs/heads").split()


# --- settings ---------------------------------------------------------------------


def test_settings_read_the_token_under_either_spelling_and_keep_it_out_of_repr():
    s = settings_from({"project_repo": "http://lite:8787/api/git/f.git", "base_ref": "main", "project_repo_token": "tok-1"})
    assert (s.repo, s.base, s.token) == ("http://lite:8787/api/git/f.git", "main", "tok-1")
    assert "tok-1" not in repr(s)
    assert settings_from({"projectRepo": "/r", "projectRepoToken": "tok-2"}).token == "tok-2"
    assert settings_from({"project_repo": "/r"}).base == "origin/main"
    assert settings_from({"base_ref": "main"}) is None


def test_only_urls_are_cloned():
    assert worktree.is_remote("http://lite:8787/api/git/x.git")
    assert worktree.is_remote("HTTPS://host/x.git")
    assert worktree.is_remote("file:///srv/x.git")
    assert not worktree.is_remote("/data/folders/t/slug")
    assert not worktree.is_remote("~/repo")


# --- a URL: clone, branch, commit, push ---------------------------------------------


def test_a_url_is_cloned_into_the_workspace_and_the_branch_goes_back_on_commit(tmp_path):
    bare, _ = _origin(tmp_path)
    ws = tmp_path / "ws"
    tree = prepare(Settings(repo=bare.as_uri(), base="main", token="tok-secret"), ws, "run-0123456789abcdef")

    assert tree.remote and tree.path == ws / "folder"
    assert tree.branch == "pgpt-agent/run-01234567"
    assert tree.base == "origin/main"
    assert git(tree.path, "rev-parse", "--abbrev-ref", "HEAD").strip() == tree.branch
    assert (tree.path / "notice.md").read_text(encoding="utf-8").startswith("# 안내문")
    # the token travels as configuration of the command, never in the clone's config
    config = (tree.path / ".git" / "config").read_text(encoding="utf-8")
    assert "tok-secret" not in config and "extraheader" not in config.lower()
    assert "tok-secret" not in repr(tree)

    (tree.path / "notice.md").write_text("# 안내문\n고친 문장\n", encoding="utf-8")
    (tree.path / "회의록 11월.md").write_text("새 파일\n", encoding="utf-8")
    change = tree.commit("고침")

    assert change["pushed"] is True and change["commit"]
    # file names as they are, not git's quoted octal escapes
    assert change["files"] == ["notice.md", "회의록 11월.md"]
    assert change["remote"] == bare.as_uri()
    assert tree.branch in _branches(bare)
    assert git(bare, "show", f"{tree.branch}:notice.md") == "# 안내문\n고친 문장\n"
    # the base branch is untouched
    assert git(bare, "show", "main:notice.md") == "# 안내문\n초안\n"


@pytest.mark.parametrize("base", ["main", "origin/main"])
def test_the_base_may_be_named_with_or_without_origin(tmp_path, base):
    bare, _ = _origin(tmp_path)
    tree = prepare(Settings(repo=bare.as_uri(), base=base), tmp_path / "ws", "r1")
    assert tree.base == "origin/main"
    assert git(tree.path, "merge-base", "HEAD", "origin/main").strip() == git(bare, "rev-parse", "main").strip()


def test_a_step_on_an_instance_line_starts_from_that_branch(tmp_path):
    bare, _ = _origin(tmp_path)
    tree = prepare(Settings(repo=bare.as_uri(), base="inst/abc123"), tmp_path / "ws", "r2")
    assert tree.base == "origin/inst/abc123"
    assert (tree.path / "line.md").exists()


def test_a_base_the_repository_does_not_have_is_an_error(tmp_path):
    bare, _ = _origin(tmp_path)
    with pytest.raises(WorktreeError, match="no-such-branch"):
        prepare(Settings(repo=bare.as_uri(), base="no-such-branch"), tmp_path / "ws", "r3")


def test_nothing_changed_means_nothing_pushed(tmp_path):
    bare, _ = _origin(tmp_path)
    tree = prepare(Settings(repo=bare.as_uri(), base="main"), tmp_path / "ws", "r4")
    change = tree.commit("없음")
    assert change["commit"] is None and change["pushed"] is False and change["files"] == []
    assert tree.branch not in _branches(bare)


def test_a_paused_run_resumes_into_its_own_clone_without_cloning_again(tmp_path):
    bare, _ = _origin(tmp_path)
    ws = tmp_path / "ws"
    first = prepare(Settings(repo=bare.as_uri(), base="main"), ws, "r5")
    (first.path / "draft.md").write_text("반쯤 쓴 글", encoding="utf-8")  # uncommitted work
    # the server goes away: a resume must not need it
    os.rename(bare, bare.with_name("gone.git"))

    again = prepare(Settings(repo=bare.as_uri(), base="main"), ws, "r5")

    assert again.path == first.path and again.branch == first.branch and again.base == "origin/main"
    assert (again.path / "draft.md").read_text(encoding="utf-8") == "반쯤 쓴 글"


def test_a_branch_an_earlier_attempt_pushed_is_continued(tmp_path):
    bare, _ = _origin(tmp_path)
    one = prepare(Settings(repo=bare.as_uri(), base="main"), tmp_path / "ws1", "same-run")
    (one.path / "a.md").write_text("첫 시도", encoding="utf-8")
    one.commit("첫 시도")

    # the workspace was swept; the same run starts again elsewhere
    two = prepare(Settings(repo=bare.as_uri(), base="main"), tmp_path / "ws2", "same-run")
    assert (two.path / "a.md").read_text(encoding="utf-8") == "첫 시도"
    (two.path / "b.md").write_text("둘째 시도", encoding="utf-8")
    change = two.commit("둘째 시도")

    assert change["pushed"] is True
    log = git(bare, "log", "--format=%s", two.branch).split("\n")
    assert log[:2] == ["둘째 시도", "첫 시도"]


def test_a_push_the_server_refuses_is_an_error_that_says_why(tmp_path):
    bare, _ = _origin(tmp_path)
    tree = prepare(Settings(repo=bare.as_uri(), base="main"), tmp_path / "ws", "r6")
    hook = bare / "hooks" / "pre-receive"
    hook.write_text("#!/bin/sh\necho 'only refs/heads/pgpt-agent/* may be pushed' >&2\nexit 1\n")
    hook.chmod(0o755)
    (tree.path / "notice.md").write_text("거절될 변경", encoding="utf-8")

    with pytest.raises(WorktreeError) as caught:
        tree.commit("거절")

    message = str(caught.value)
    assert "could not push pgpt-agent/r6" in message and "rejected" in message
    assert tree.branch not in _branches(bare)
    # the commit is kept: a later attempt sends it
    hook.unlink()
    assert tree.push() is True
    assert tree.branch in _branches(bare)


def test_an_unreachable_server_is_a_clone_error_and_leaves_nothing_behind(tmp_path):
    ws = tmp_path / "ws"
    with pytest.raises(WorktreeError, match="could not clone"):
        prepare(Settings(repo=(tmp_path / "missing.git").as_uri(), base="main"), ws, "r7")
    assert not (ws / "missing").exists()


def test_a_password_in_the_url_is_used_as_the_token_and_not_kept_as_the_remote():
    url, token = worktree._split_credentials("http://x:tok%2F1@lite:8787/api/git/f.git", "")
    assert (url, token) == ("http://lite:8787/api/git/f.git", "tok/1")
    assert worktree._split_credentials("http://lite/api/git/f.git", "t2") == ("http://lite/api/git/f.git", "t2")
    env = worktree._auth_config("t3")
    pairs = {env[f"GIT_CONFIG_KEY_{i}"]: env[f"GIT_CONFIG_VALUE_{i}"] for i in range(int(env["GIT_CONFIG_COUNT"]))}
    assert pairs == {"credential.helper": "", "http.extraHeader": "Authorization: Bearer t3"}


def test_a_swept_workspace_takes_its_clone_with_it(tmp_path, monkeypatch):
    from core import workspace as workspace_module
    from core.settings import settings

    bare, _ = _origin(tmp_path)
    root = tmp_path / "root"
    monkeypatch.setattr(workspace_module, "settings", dataclasses.replace(settings, workspace_root=root))
    ws = workspace_module.for_run("old-run", tenant_id="acme")
    tree = prepare(Settings(repo=bare.as_uri(), base="main"), ws.path, "old-run")
    assert (tree.path / ".git" / "objects").is_dir()

    removed = workspace_module.sweep(now=time.time() + settings.retention_seconds + 60)

    assert ws.path in removed and not ws.path.exists()


# --- a path: today's worktree, unchanged ------------------------------------------------


def test_a_local_path_still_gets_a_worktree_and_nothing_is_pushed(tmp_path):
    _, seed = _origin(tmp_path)
    git(seed, "checkout", "-q", "main")
    ws = tmp_path / "ws"
    tree = prepare(Settings(repo=str(seed), base="main"), ws, "local-run")

    assert not tree.remote and tree.path == ws / "seed"
    assert (tree.path / ".git").is_file()  # a worktree, not a clone
    (tree.path / "notice.md").write_text("로컬 변경", encoding="utf-8")
    change = tree.commit("로컬")

    assert change["commit"] and "pushed" not in change
    assert git(seed, "show", f"{tree.branch}:notice.md") == "로컬 변경"


def test_a_path_that_is_not_a_repository_is_refused(tmp_path):
    with pytest.raises(WorktreeError, match="not a git repository"):
        prepare(Settings(repo=str(tmp_path), base="main"), tmp_path / "ws", "x")


# --- the executor: a work item that names a URL ------------------------------------------


class _WorkItem:
    task_id = "todo-remote-1"
    context_id = "proc-1"

    def __init__(self, agent_config: dict[str, Any]) -> None:
        self.row = {
            "id": "todo-remote-1",
            "tenant_id": "acme",
            "activity_name": "안내문 다듬기",
            "query": "안내문을 다듬어라",
            "agent_config": agent_config,
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


def _run_item(tmp_path, monkeypatch, agent_config, seen: dict[str, Any]):
    from cliagents import registry

    import executor as ex
    from core import workspace as workspace_module
    from core.settings import settings

    monkeypatch.setattr(workspace_module, "settings", dataclasses.replace(settings, workspace_root=tmp_path / "root"))
    monkeypatch.setattr(ex, "require_runnable", lambda chosen: registry.get(chosen.agent_id))

    async def fake_astream(provider, request, env=None, timeout=None):
        seen["workdir"] = request.workdir
        seen["extra_args"] = list(request.extra_args)
        Path(request.workdir, "notice.md").write_text("# 안내문\n에이전트가 다듬음\n", encoding="utf-8")
        yield ExecEvent(kind=ExecEventKind.RESULT, text="안내문을 다듬었어요.", session_id="S-1")

    monkeypatch.setattr(ex, "astream", fake_astream)
    queue = _Queue()
    asyncio.run(ex.CliAgentExecutor().execute(_WorkItem(agent_config), queue))
    return queue


def _texts(queue: _Queue) -> list[str]:
    from google.protobuf.json_format import MessageToDict

    return [json.dumps(MessageToDict(e), ensure_ascii=False) for e in queue.sent]


def test_a_work_item_on_a_url_runs_in_the_clone_and_stores_the_pushed_branch(tmp_path, monkeypatch):
    bare, _ = _origin(tmp_path)
    seen: dict[str, Any] = {}
    queue = _run_item(tmp_path, monkeypatch, {"project_repo": bare.as_uri(), "base_ref": "main", "project_repo_token": "tok-x"}, seen)

    # the agent ran in the clone, with the run's own folder (skills, instructions) added
    assert seen["workdir"].endswith("/todo-remote-1/folder")
    assert seen["extra_args"][-2:] == ["--add-dir", str(Path(seen["workdir"]).parent / ".claude")]
    branch = "pgpt-agent/todo-remote-"  # run id "todo-remote-1", first 12 characters
    assert branch in _branches(bare)
    assert git(bare, "show", f"{branch}:notice.md") == "# 안내문\n에이전트가 다듬음\n"
    texts = _texts(queue)
    final = [t for t in texts if "assistant_response" in t]
    assert final, texts
    body = json.loads(json.loads(final[-1])["artifact"]["parts"][0]["text"])
    assert body["text"] == "안내문을 다듬었어요."
    assert body["worktree"]["branch"] == branch and body["worktree"]["files"] == ["notice.md"]
    assert body["worktree"]["pushed"] is True
    assert any("커밋하고 올렸습니다" in t for t in texts)
    assert not any("tok-x" in t for t in texts)


def test_a_work_item_whose_repository_cannot_be_cloned_fails_and_does_not_run(tmp_path, monkeypatch):
    seen: dict[str, Any] = {}
    with pytest.raises(WorktreeError):
        _run_item(tmp_path, monkeypatch, {"project_repo": (tmp_path / "nope.git").as_uri(), "base_ref": "main"}, seen)
    assert seen == {}  # the CLI was never started


def test_a_push_that_fails_fails_the_work_item(tmp_path, monkeypatch):
    bare, _ = _origin(tmp_path)
    hook = bare / "hooks" / "pre-receive"
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)
    seen: dict[str, Any] = {}
    with pytest.raises(WorktreeError, match="could not push"):
        _run_item(tmp_path, monkeypatch, {"project_repo": bare.as_uri(), "base_ref": "main"}, seen)
    assert seen["workdir"]  # it ran; only the hand-back failed


def test_a_refused_token_is_said_in_words_a_reviewer_understands():
    said = worktree._explain(WorktreeError("git clone -q --no-checkout failed: fatal: could not read Username for 'http://lite'"))
    assert said.startswith("the server refused the token (HTTP 401")
    assert worktree._explain(WorktreeError("other")) == "other"
