"""Environment-derived configuration, resolved once at import.

Everything an operator can turn is here, with a default that is safe rather
than convenient: a permission level that cannot reach outside the workspace, a
concurrency cap low enough that CLI processes do not eat the node, a retention
window long enough to answer "where did my file go?".
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from cliagents import Permission

#: The orchestration value this service polls. One value for every CLI: the
#: poller is keyed on it, so a per-CLI value would mean a deployment per CLI.
AGENT_TYPE = "cliagents"

_PERMISSION_BY_NAME = {p.value: p for p in Permission}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name) or default).expanduser()


@dataclass(frozen=True)
class Settings:
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("PORT", 8890))

    #: Root of all run workspaces. Must be on a persistent volume: a run that
    #: is waiting for a human keeps its context here, and a restart that loses
    #: it turns "answer the question" into "start over".
    workspace_root: Path = field(
        default_factory=lambda: _env_path("CLIAGENTS_WORKSPACE_ROOT", "/workspace")
    )
    #: How long a finished run's workspace survives. Answers "can I still
    #: download it?" and "can this still be resumed?" with one number.
    workspace_retention_hours: int = field(
        default_factory=lambda: _env_int("CLIAGENTS_WORKSPACE_RETENTION_HOURS", 72)
    )

    #: Directories holding tenant and bundled skills, in precedence order.
    skills_dirs: tuple[Path, ...] = field(
        default_factory=lambda: tuple(
            Path(p).expanduser()
            for p in (os.getenv("SKILLS_DIRS") or "./skills").split(",")
            if p.strip()
        )
    )
    system_skills_dir: Path = field(
        default_factory=lambda: _env_path("CLIAGENTS_SYSTEM_SKILLS_DIR", "/app/system-skills")
    )

    #: CLI processes are heavy — a process, a workspace, its own context
    #: window. Past a handful in parallel the node, not the model, is the
    #: bottleneck.
    max_concurrent_runs: int = field(
        default_factory=lambda: _env_int("CLIAGENTS_MAX_CONCURRENT_RUNS", 3)
    )
    #: Wall clock for one run. A CLI agent that has gone quiet is not going to
    #: come back, and the work item should say so rather than hang.
    run_timeout_seconds: int = field(
        default_factory=lambda: _env_int("CLIAGENTS_RUN_TIMEOUT_SECONDS", 1800)
    )

    default_agent_id: str = field(
        default_factory=lambda: os.getenv("CLIAGENTS_DEFAULT_CLI", "claude-code")
    )
    default_permission: Permission = field(
        default_factory=lambda: _PERMISSION_BY_NAME.get(
            os.getenv("CLIAGENTS_DEFAULT_PERMISSION", ""), Permission.WORKSPACE_WRITE
        )
    )

    #: Truncation point for file previews pushed to the browser. Whole files
    #: go through the download route instead.
    file_preview_max_bytes: int = field(
        default_factory=lambda: _env_int("CLIAGENTS_FILE_PREVIEW_MAX_BYTES", 400_000)
    )

    @property
    def retention_seconds(self) -> int:
        return self.workspace_retention_hours * 3600


settings = Settings()
