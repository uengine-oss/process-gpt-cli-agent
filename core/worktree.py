"""Running a work item against a repository, on a branch of its own.

A work item can name a repository (``project_repo``) instead of an empty run
directory. The run then gets a checkout of that repository inside its
workspace, on a fresh branch ``pgpt-agent/<run>`` from ``base_ref``. The agent
edits there; the service — not the agent — commits what changed, so the branch
is a reviewable proposal and nothing reaches the base branch without a person
merging it.

``project_repo`` is one of two things:

- **a path on this pod's disk** — the run gets a ``git worktree`` of it. Only
  works when the repository and this service share a disk (local development).
- **a URL** (``http(s)://…``, or ``file://…`` in tests) — the run gets its own
  clone, and the branch goes back with ``git push`` once committed. This is how
  Process GPT Lite hands over a shared folder: the folder's repository lives on
  Lite's volume, which this pod cannot see, and Lite serves it over git smart
  HTTP. ``project_repo_token`` is the bearer token Lite minted for that one
  folder; it is sent as a request header on every fetch and push and never
  written into the clone's ``.git/config``.

The workspace sweep (``core.workspace.sweep``) removes a run directory with
everything in it, so a clone goes with its run. A worktree of a local
repository leaves a registration behind in that repository, which
``git worktree prune`` there clears.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

logger = logging.getLogger(__name__)

_REPO_KEYS = ("project_repo", "projectRepo")
_BASE_KEYS = ("base_ref", "baseRef")
_TOKEN_KEYS = ("project_repo_token", "projectRepoToken")
_GIT_TIMEOUT_SECONDS = 120
#: Cloning a folder with years of documents takes longer than a local command.
_NETWORK_TIMEOUT_SECONDS = 900
_REMOTE = re.compile(r"^(https?|file)://", re.IGNORECASE)
_COMMITTER = ("-c", "user.name=ProcessGPT cli-agent", "-c", "user.email=cli-agent@process-gpt.local")
#: Paths as they are (a shared folder is full of Korean file names), not C-quoted octal escapes.
_BASE = ("-c", "core.quotepath=false")


class WorktreeError(Exception):
    """The repository could not be prepared or the branch could not be handed
    back; the run says so instead of carrying on elsewhere."""


def is_remote(repo: str) -> bool:
    return bool(_REMOTE.match(str(repo or "").strip()))


@dataclass
class Settings:
    repo: str = ""
    base: str = "origin/main"
    #: Bearer token for a URL ``repo``. Kept out of ``repr`` so a logged
    #: settings object does not print it.
    token: str = field(default="", repr=False)


def settings_from(fields: dict) -> Settings | None:
    def first(keys: tuple[str, ...]) -> str:
        return next((str(fields[k]).strip() for k in keys if str(fields.get(k) or "").strip()), "")

    repo = first(_REPO_KEYS)
    if not repo:
        return None
    return Settings(repo=repo, base=first(_BASE_KEYS) or "origin/main", token=first(_TOKEN_KEYS))


@dataclass
class Worktree:
    #: The repository as named: a local path, or the URL without credentials.
    repo: str
    path: Path
    branch: str
    #: Where the branch started: ``origin/<base>`` for a clone, the given ref otherwise.
    base: str
    remote: bool = False
    #: git configuration for the clone's network commands (the token's header). Never logged.
    auth: dict[str, str] = field(default_factory=dict, repr=False)

    @property
    def name(self) -> str:
        return _repo_name(self.repo)

    def commit(self, message: str, *, exclude: list[str] | None = None) -> dict:
        """Commit everything the agent changed on the run's branch and describe it.

        For a clone the branch is then pushed back. A push that fails raises
        :class:`WorktreeError`: the commit exists only here, and a run that
        reported success would hand the reviewer a branch that is not there.
        """
        _git(self.path, "add", "-A")
        for rel in exclude or []:
            _git(self.path, "reset", "-q", "--", rel, check=False)
        staged = _git(self.path, "diff", "--cached", "--name-status").strip()
        sha = None
        if staged:
            _git(self.path, *_COMMITTER, "commit", "-q", "-m", message)
            sha = _git(self.path, "rev-parse", "--short", "HEAD").strip()
        stat = _git(self.path, "diff", "--stat", f"{self.base}...HEAD", check=False).strip().splitlines()
        out = {
            "branch": self.branch,
            "base": self.base,
            "commit": sha,
            "files": [line.split("\t", 1)[-1] for line in staged.splitlines()] if staged else [],
            "stat": stat[-1].strip() if stat else "",
            "worktree": str(self.path),
        }
        if self.remote:
            out["remote"] = self.repo
            out["pushed"] = self.push()
        return out

    def push(self) -> bool:
        """Send the branch back. False when it holds nothing beyond its base.

        Also sends commits an earlier attempt made but could not push (a
        resumed run, a push that failed): what counts is the branch, not this
        call's commit.
        """
        head = _git(self.path, "rev-parse", "HEAD").strip()
        start = _git(self.path, "rev-parse", f"{self.base}^{{commit}}", check=False).strip()
        on_remote = _ref(self.path, f"refs/remotes/origin/{self.branch}")
        if head == start and not on_remote:
            return False
        if head == on_remote:
            return True
        try:
            _git(
                self.path, "push", "--porcelain", "origin", f"HEAD:refs/heads/{self.branch}",
                env=self.auth, timeout=_NETWORK_TIMEOUT_SECONDS,
            )
        except WorktreeError as exc:
            raise WorktreeError(f"could not push {self.branch} to {self.repo}: {_explain(exc)}") from exc
        return True


def prepare(settings: Settings, workspace_path: Path, run_id: str) -> Worktree:
    """Create (or reuse, when a paused run resumes) the run's checkout."""
    branch = f"pgpt-agent/{run_id[:12]}"
    if is_remote(settings.repo):
        return _prepare_clone(settings, workspace_path, branch)
    return _prepare_worktree(settings, workspace_path, branch)


def _prepare_worktree(settings: Settings, workspace_path: Path, branch: str) -> Worktree:
    repo = Path(settings.repo).expanduser().resolve()
    if not (repo / ".git").exists():
        raise WorktreeError(f"not a git repository: {repo}")
    path = workspace_path / repo.name
    if path.is_dir() and (path / ".git").exists():
        return Worktree(repo=str(repo), path=path, branch=branch, base=settings.base)
    _git(repo, "fetch", "-q", "origin", check=False)
    try:
        _git(repo, "worktree", "add", "-q", "-b", branch, str(path), settings.base)
    except WorktreeError:
        # The branch survives a previous attempt whose workspace was swept.
        _git(repo, "worktree", "add", "-q", str(path), branch)
    logger.info("worktree %s on %s (from %s)", path, branch, settings.base)
    return Worktree(repo=str(repo), path=path, branch=branch, base=settings.base)


def _prepare_clone(settings: Settings, workspace_path: Path, branch: str) -> Worktree:
    url, token = _split_credentials(settings.repo, settings.token)
    auth = _auth_config(token)
    path = workspace_path / _repo_name(url)

    if (path / ".git").is_dir():
        # A paused run resumes into its own clone: the agent's uncommitted
        # work is there, and the token may be close to its end. No network.
        _git(path, "checkout", "-q", branch, check=False)
        base = _start(path, settings.base, allow_missing=True)
        return Worktree(repo=url, path=path, branch=branch, base=base, remote=True, auth=auth)

    if path.exists():
        # What an interrupted clone left; nothing in it is the agent's.
        shutil.rmtree(path)
    workspace_path.mkdir(parents=True, exist_ok=True)
    try:
        _git(workspace_path, "clone", "-q", "--no-checkout", url, str(path),
             env=auth, timeout=_NETWORK_TIMEOUT_SECONDS)
    except WorktreeError as exc:
        shutil.rmtree(path, ignore_errors=True)
        raise WorktreeError(f"could not clone {url}: {_explain(exc)}") from exc

    base = _start(path, settings.base)
    if _ref(path, f"refs/remotes/origin/{branch}"):
        # An earlier attempt of this run pushed the branch: continue it.
        _git(path, "checkout", "-q", "-B", branch, f"origin/{branch}")
    else:
        _git(path, "checkout", "-q", "-b", branch, base)
    logger.info("clone %s on %s (from %s)", path, branch, base)
    return Worktree(repo=url, path=path, branch=branch, base=base, remote=True, auth=auth)


def _start(path: Path, base: str, *, allow_missing: bool = False) -> str:
    """The ref the branch starts from: ``origin/<base>``, or ``base`` itself (a commit)."""
    base = (base or "main").strip()
    candidates = [base] if base.startswith("origin/") else [f"origin/{base}", base]
    for candidate in candidates:
        if _ref(path, f"{candidate}^{{commit}}"):
            return candidate
    if allow_missing:
        return candidates[0]
    raise WorktreeError(f"base '{base}' is not in the repository")


def _split_credentials(url: str, token: str) -> tuple[str, str]:
    """The URL without user:password (that would be stored as the remote) and the token.

    A password in the URL stands in for a missing token, so a hand-written
    ``http://x:<token>@host/…`` works the same as the two settings.
    """
    parts = urlsplit(url.strip())
    if parts.password and not token:
        token = unquote(parts.password)
    if parts.username or parts.password:
        netloc = parts.hostname or ""
        if parts.port:
            netloc = f"{netloc}:{parts.port}"
        url = urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    return url, token


def _auth_config(token: str) -> dict[str, str]:
    """git configuration given through the environment, like ``git -c``.

    Command-level configuration is never written to ``.git/config``. The
    environment rather than ``-c`` on the command line keeps the token out of
    the process list, which the agents in this pod can read. No credential
    helper is asked, and git never stops to prompt.
    """
    pairs = [("credential.helper", "")]
    if token:
        pairs.append(("http.extraHeader", f"Authorization: Bearer {token}"))
    env = {"GIT_CONFIG_COUNT": str(len(pairs))}
    for i, (key, value) in enumerate(pairs):
        env[f"GIT_CONFIG_KEY_{i}"] = key
        env[f"GIT_CONFIG_VALUE_{i}"] = value
    return env


def _repo_name(repo: str) -> str:
    if is_remote(repo):
        tail = urlsplit(repo).path.rstrip("/").rsplit("/", 1)[-1]
    else:
        tail = Path(repo).name
    name = re.sub(r"\.git$", "", tail)
    return re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("._") or "repo"


def _ref(cwd: Path, ref: str) -> str:
    return _git(cwd, "rev-parse", "--verify", "--quiet", ref, check=False).strip()


def _git(cwd: Path, *args: str, check: bool = True, env: dict[str, str] | None = None,
         timeout: int = _GIT_TIMEOUT_SECONDS) -> str:
    full_env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_CONFIG_")}
    full_env["GIT_TERMINAL_PROMPT"] = "0"
    full_env.update(env or {})
    try:
        done = subprocess.run(["git", *_BASE, *args], cwd=str(cwd), capture_output=True, text=True,
                              timeout=timeout, env=full_env)
    except subprocess.TimeoutExpired as exc:
        raise WorktreeError(f"git {' '.join(args[:3])} timed out after {timeout}s") from exc
    if check and done.returncode != 0:
        # a push says why a ref was refused on stdout (--porcelain), the rest on stderr
        detail = "\n".join(s.strip() for s in (done.stderr, done.stdout) if s and s.strip())
        raise WorktreeError(f"git {' '.join(args[:3])} failed: {_redact(detail, env)[:600]}")
    return done.stdout


#: What git says when the server answered 401 and it had nowhere to ask for credentials.
_AUTH_REFUSED = ("could not read Username", "Authentication failed", "The requested URL returned error: 401")


def _explain(exc: Exception) -> str:
    """git's words, with what they mean when the server refused the token."""
    text = str(exc)
    if any(marker in text for marker in _AUTH_REFUSED):
        return ("the server refused the token (HTTP 401: no token, an expired one, "
                f"or one for another repository) — {text}")
    return text


def _redact(text: str, env: dict[str, str] | None) -> str:
    """git does not print request headers, but an error is no place to find out it did."""
    for value in (env or {}).values():
        if value.startswith("Authorization: Bearer "):
            text = text.replace(value.split(" ", 2)[-1], "***")
    return text
