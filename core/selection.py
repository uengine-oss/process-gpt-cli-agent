"""Which CLI runs this work item, with which model and how much freedom.

The choice arrives from the UI on the work item, which means it arrives as
whatever the browser put there: a missing field, a legacy value, a CLI that was
uninstalled last week. This module turns that into a decision or a refusal —
never into a substitution. Running Codex because Claude Code was missing would
produce work nobody asked for, under a name they did not choose.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from cliagents import (
    AgentNotInstalledError,
    ExecRequest,
    Permission,
    Surface,
    UnknownAgentError,
    registry,
)

from .settings import settings

#: Where the UI stores the choice. Several spellings because the value reaches
#: us through the work item, the agent record and the chat request body, and
#: those have never agreed on casing.
_AGENT_KEYS = ("agent_cli", "agentCli", "cli_agent", "cliAgent")
_MODEL_KEYS = ("agent_model", "agentModel", "model")
_PERMISSION_KEYS = ("agent_permission", "agentPermission", "permission")

_PERMISSION_BY_NAME = {p.value: p for p in Permission}
#: Names the UI may send instead of ours. Kept explicit: an unrecognised value
#: falls back to the safe default rather than to the nearest-looking one.
_PERMISSION_ALIASES = {
    "read-only": Permission.READ_ONLY,
    "readonly": Permission.READ_ONLY,
    "workspace-write": Permission.WORKSPACE_WRITE,
    "write": Permission.WORKSPACE_WRITE,
    "command-exec": Permission.COMMAND_EXEC,
    "full": Permission.COMMAND_EXEC,
}


class CliSelectionError(Exception):
    """The chosen CLI cannot run here, and nothing else will be run instead.

    Carries user-facing guidance because the fix is always the operator's:
    install it, log in, or pick another one.
    """

    def __init__(self, message: str, *, agent_id: str, hint: str = "") -> None:
        self.agent_id = agent_id
        self.hint = hint
        super().__init__(f"{message} ({hint})" if hint else message)


@dataclass(frozen=True)
class Selection:
    agent_id: str
    model: str | None
    permission: Permission

    def to_request(
        self, prompt: str, workdir: str, *, resume_session: str | None = None
    ) -> ExecRequest:
        return ExecRequest(
            prompt=prompt,
            workdir=workdir,
            model=self.model,
            permission=self.permission,
            resume_session=resume_session,
        )


def _first(source: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _flattened(row: dict[str, Any]) -> dict[str, Any]:
    """Row fields plus anything nested in its JSON settings blob.

    The UI writes the CLI choice next to the orchestration on some paths and
    inside a settings object on others; reading both means neither path is the
    one that silently does nothing.
    """
    merged: dict[str, Any] = {k: v for k, v in row.items() if not isinstance(v, (dict, list))}
    for key in ("agent_config", "agentConfig", "settings", "metadata"):
        blob = row.get(key)
        if isinstance(blob, str) and blob.strip():
            try:
                blob = json.loads(blob)
            except json.JSONDecodeError:
                continue
        if isinstance(blob, dict):
            for k, v in blob.items():
                merged.setdefault(k, v)
    return merged


def resolve(row: dict[str, Any]) -> Selection:
    """Decide what to run, without checking whether it is installed.

    Separated from :func:`require_runnable` so a caller can report *what was
    asked for* even when it is unavailable — "codex is not installed" is a
    better error than "no agent".
    """
    fields = _flattened(row)

    agent_id = _first(fields, _AGENT_KEYS) or settings.default_agent_id
    model = _first(fields, _MODEL_KEYS) or None

    raw_permission = _first(fields, _PERMISSION_KEYS).lower()
    permission = (
        _PERMISSION_BY_NAME.get(raw_permission)
        or _PERMISSION_ALIASES.get(raw_permission)
        or settings.default_permission
    )

    return Selection(agent_id=agent_id, model=model, permission=permission)


def require_runnable(selection: Selection):
    """Return the provider, or explain why this run cannot happen.

    Every failure here is deliberate: the alternative is a job that quietly ran
    on a different agent than the one on the work item.
    """
    try:
        provider = registry.get(selection.agent_id)
    except UnknownAgentError as exc:
        raise CliSelectionError(
            f"'{selection.agent_id}' is not a known CLI agent",
            agent_id=selection.agent_id,
            hint=f"available: {', '.join(registry.ids())}",
        ) from exc

    if not provider.supports(Surface.EXEC):
        raise CliSelectionError(
            f"'{selection.agent_id}' cannot be given work non-interactively",
            agent_id=selection.agent_id,
        )

    try:
        provider.resolve_executable(refresh=True)
    except AgentNotInstalledError as exc:
        raise CliSelectionError(
            f"'{selection.agent_id}' is not installed in this environment",
            agent_id=selection.agent_id,
            hint=exc.install_hint,
        ) from exc

    return provider
