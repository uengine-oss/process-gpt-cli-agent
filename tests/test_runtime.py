from pathlib import Path

from core.runtime import RuntimeLease


def test_runtime_lease_removes_new_files_and_restores_existing_files(tmp_path):
    existing = tmp_path / "AGENTS.md"
    created = tmp_path / ".codex" / "agents" / "worker.toml"
    existing.write_text("user content", encoding="utf-8")

    lease = RuntimeLease(tmp_path)
    lease.capture([existing, created])
    existing.write_text("runtime content", encoding="utf-8")
    created.parent.mkdir(parents=True)
    created.write_text("secret", encoding="utf-8")

    lease.restore()
    assert existing.read_text(encoding="utf-8") == "user content"
    assert not created.exists()


def test_runtime_lease_refuses_a_path_outside_the_workspace(tmp_path):
    import pytest

    lease = RuntimeLease(tmp_path / "run")
    lease.root.mkdir()
    with pytest.raises(PermissionError):
        lease.capture([tmp_path / "outside.txt"])


def test_claude_subagent_credentials_live_in_the_isolated_runtime_and_are_removed(tmp_path):
    from cliagents import ArtifactBundle, McpServer, registry

    from core import subagents

    lease = RuntimeLease(tmp_path)
    agents = subagents.prepare(
        [{"name": "Reviewer", "tools": "office", "skills": "policy"}],
        tenant_servers=[McpServer("office", "npx", ["office-mcp"], {"TOKEN": "secret"})],
        fallback_skills=[],
        fallback_tools=[],
    )
    targets = subagents.provision_definitions(
        registry.get("claude-code"), ArtifactBundle(), agents, tmp_path, lease
    )
    target = Path(targets[0])
    assert ".agent-home" in target.parts
    assert "TOKEN" in target.read_text(encoding="utf-8")

    lease.restore()
    assert not target.exists()
