"""Giving the agent ProcessGPT's tools, without giving it someone else's.

Tools reach a CLI agent as MCP servers. Registering them is the library's job;
the part that belongs here is the consequence of *where* each agent keeps that
registration. Claude Code writes into the project, so one run's tools stay in
one run's workspace. Codex writes into ``$CODEX_HOME/config.toml`` — user-global,
shared by every run in the container.

Left alone, that means tenant B's Codex run inherits tenant A's servers and
whatever credentials were in their environment. So a run whose agent registers
globally gets its own ``CODEX_HOME``, and if that cannot be arranged the run
does not happen. Executing with the wrong tenant's tools is worse than not
executing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from cliagents import Artifact, ArtifactKind, ConfigScope, McpServer

logger = logging.getLogger(__name__)

#: Environment variable each globally-scoped agent uses to relocate its config.
_CONFIG_HOME_ENV = {"codex": "CODEX_HOME"}

#: A minimal command so `install_bridge` has something to install. The bridge
#: API pairs a command with the servers; ProcessGPT drives the agent by prompt,
#: so this exists to satisfy the contract and to give a human a way in.
_ENTRY_COMMAND = Artifact(
    kind=ArtifactKind.COMMAND,
    name="processgpt",
    content=(
        "---\n"
        "description: Run a ProcessGPT work item with the ProcessGPT tools available.\n"
        "---\n\n"
        "ProcessGPT 프로세스·폼·지식 도구가 MCP 로 연결되어 있습니다. "
        "업무 지시를 따르고, 결과는 지정된 형식으로 제출하세요.\n"
    ),
    description="Run a ProcessGPT work item",
)


class BridgeIsolationError(Exception):
    """The run could not be given a private tool configuration."""


@dataclass
class BridgeResult:
    servers: list[str] = field(default_factory=list)
    scope: str = ""
    config_path: str | None = None
    #: Servers that could not be registered, with the reason. The run
    #: continues without them, but says so.
    failed: dict[str, str] = field(default_factory=dict)
    #: Environment overrides the run must be started with for the registration
    #: to apply to it and to nothing else.
    env: dict[str, str] = field(default_factory=dict)

    @property
    def isolated(self) -> bool:
        return bool(self.env)


def install(provider, workdir: Path, servers: list[McpServer]) -> BridgeResult:
    """Register ``servers`` for one run of ``provider`` in ``workdir``.

    Idempotent by way of the library: re-running a work item rewrites the same
    entries and leaves anything the user added alone.
    """
    if not servers:
        return BridgeResult()

    env = _isolate_config_home(provider, workdir)

    try:
        result = provider.install_bridge(str(workdir), _ENTRY_COMMAND, servers)
    except Exception as exc:  # noqa: BLE001 - degraded, not fatal
        # Tools are how the agent reaches the process; losing them makes for a
        # worse answer, not a failed run. The caller surfaces this.
        logger.warning("MCP bridge install failed for %s: %s", provider.id, exc)
        return BridgeResult(
            failed={s.name: str(exc) for s in servers},
            env=env,
        )

    return BridgeResult(
        servers=list(result.servers),
        scope=result.scope.value if hasattr(result.scope, "value") else str(result.scope),
        config_path=result.mcp_config_path,
        env=env,
    )


def _isolate_config_home(provider, workdir: Path) -> dict[str, str]:
    """Give globally-scoped agents a config home of their own, or refuse.

    Returns the environment the run must inherit. Empty for agents that keep
    their registration in the project, which need no help.
    """
    if getattr(provider, "mcp_scope", None) is not ConfigScope.USER_GLOBAL:
        return {}

    variable = _CONFIG_HOME_ENV.get(provider.id)
    if not variable:
        raise BridgeIsolationError(
            f"'{provider.id}' registers MCP servers globally and this service does not "
            "know how to give it a private configuration — refusing to run rather than "
            "share tools between tenants"
        )

    home = workdir / ".agent-home" / provider.id
    try:
        home.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise BridgeIsolationError(f"could not create an isolated config home: {exc}") from exc

    return {variable: str(home)}


def processgpt_servers(
    *,
    tenant_mcp: dict | None = None,
    extra_env: dict[str, str] | None = None,
) -> list[McpServer]:
    """The MCP servers a run should see.

    Tenant-configured servers come from the ProcessGPT database in the shape
    MCP itself uses (``mcpServers: {name: {command, args, env}}``), so they are
    read as data rather than re-modelled.
    """
    servers: list[McpServer] = []
    config = (tenant_mcp or {}).get("mcpServers")
    if not isinstance(config, dict):
        return servers

    for name, spec in config.items():
        if not isinstance(spec, dict) or not spec.get("command"):
            continue
        env = dict(spec.get("env") or {})
        env.update(extra_env or {})
        servers.append(
            McpServer(
                name=str(name),
                command=str(spec["command"]),
                args=[str(a) for a in (spec.get("args") or [])],
                env=env,
            )
        )
    return servers
