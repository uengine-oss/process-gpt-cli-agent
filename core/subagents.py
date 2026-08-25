"""Translate DeepAgent-style activity agents into CLI-native subagents."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any

from cliagents import Artifact, ArtifactBundle, ArtifactKind, McpServer

from .runtime import RuntimeLease


@dataclass(frozen=True)
class RuntimeAgent:
    name: str
    description: str
    instructions: str
    skills: list[str] = field(default_factory=list)
    servers: list[McpServer] = field(default_factory=list)


def prepare(
    raw_agents: list[Any],
    *,
    tenant_servers: list[McpServer],
    fallback_skills: list[str],
    fallback_tools: list[str],
) -> list[RuntimeAgent]:
    available = {server.name: server for server in tenant_servers}
    agents: list[RuntimeAgent] = []
    used: set[str] = set()
    for index, raw in enumerate(raw_agents or []):
        if not isinstance(raw, dict):
            continue
        base = raw.get("alias") or raw.get("username") or raw.get("name") or f"worker-{index + 1}"
        name = _unique_slug(str(base), used)
        skill_names = _strings(raw.get("skills"))
        tool_names = _strings(raw.get("tools"))
        description = str(raw.get("description") or raw.get("role") or raw.get("goal") or name)
        instructions = "\n".join(
            part for part in (
                str(raw.get("role") or "").strip(),
                str(raw.get("goal") or "").strip(),
                str(raw.get("persona") or "").strip(),
                "배정된 업무만 수행하고 결과를 부모 에이전트에게 반환하세요.",
            ) if part
        )
        agents.append(
            RuntimeAgent(
                name=name,
                description=description,
                instructions=instructions,
                skills=skill_names,
                servers=[available[name] for name in tool_names if name in available],
            )
        )

    if not agents:
        names = fallback_tools or list(available)
        agents.append(
            RuntimeAgent(
                name="processgpt-worker",
                description="현재 ProcessGPT 액티비티를 수행하는 격리된 작업자",
                instructions="현재 액티비티를 끝까지 수행하고 검증된 결과만 부모 에이전트에게 반환하세요.",
                skills=list(dict.fromkeys(fallback_skills)),
                servers=[available[name] for name in names if name in available],
            )
        )
    return agents


def add_to_bundle(bundle: ArtifactBundle, agents: list[RuntimeAgent]) -> None:
    for agent in agents:
        bundle.add_role_agent(
            agent.name,
            agent.instructions,
            agent.description,
            metadata={"skills": list(agent.skills), "mcp_servers": list(agent.servers)},
        )


def provision_definitions(
    provider,
    bundle: ArtifactBundle,
    agents: list[RuntimeAgent],
    workdir,
    lease: RuntimeLease,
) -> list[str]:
    """Place native agents where the CLI can load them without global state.

    Claude's current trust rules can skip inline MCP declared by a newly
    created project agent. An isolated ``CLAUDE_CONFIG_DIR/agents`` is a
    user-scope source and therefore avoids both the trust prompt and the real
    machine's global configuration. Codex project agents do not have that
    trust rule and remain in ``.codex/agents``.
    """
    artifacts = [
        Artifact(
            kind=ArtifactKind.ROLE_AGENT,
            name=agent.name,
            content=agent.instructions,
            description=agent.description,
            metadata={"skills": list(agent.skills), "mcp_servers": list(agent.servers)},
        )
        for agent in agents
    ]
    if provider.id != "claude-code":
        for artifact in artifacts:
            bundle.artifacts.append(artifact)
        return []

    home = workdir / ".agent-home" / provider.id / "agents"
    targets = [home / f"{artifact.name}.md" for artifact in artifacts]
    lease.capture(targets)
    for artifact, target in zip(artifacts, targets, strict=True):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(provider.render(artifact), encoding="utf-8")
    return [str(target) for target in targets]


def delegation_instructions(agents: list[RuntimeAgent]) -> str:
    lines = ["이 업무는 다음 서브에이전트에게 위임해서 수행하세요. 부모는 직접 MCP를 사용하지 않습니다."]
    for agent in agents:
        lines.append(f"- {agent.name}: {agent.description}")
    return "\n".join(lines)


def selected_skill_names(agents: list[RuntimeAgent]) -> list[str]:
    return list(dict.fromkeys(name for agent in agents for name in agent.skills))


def preload_discovered_skills(
    agents: list[RuntimeAgent], discovered: list[str]
) -> list[RuntimeAgent]:
    """Make the synthesized worker preload the default-discovered skills."""
    if (
        len(agents) == 1
        and agents[0].name == "processgpt-worker"
        and not agents[0].skills
    ):
        return [replace(agents[0], skills=list(dict.fromkeys(discovered)))]
    return agents


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, list):
        return [str(part).strip() for part in value if str(part).strip()]
    return []


def _unique_slug(value: str, used: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "worker"
    candidate = base
    index = 2
    while candidate in used:
        candidate = f"{base}-{index}"
        index += 1
    used.add(candidate)
    return candidate
