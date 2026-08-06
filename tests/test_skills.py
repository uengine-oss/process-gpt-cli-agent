"""One skill, stored once, readable by whichever CLI runs the job.

This is the claim the whole artifact surface exists to make, so it is tested
against both providers rather than asserted in a docstring.
"""

from __future__ import annotations

import dataclasses
import subprocess

import pytest
from cliagents import registry

from core import skills
from core.settings import settings


@pytest.fixture
def skill_library(tmp_path, monkeypatch):
    """A tenant skills directory holding one skill with reference material."""
    root = tmp_path / "skills"
    folder = root / "expense-policy"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(
        "---\ndescription: 경비 규정을 판단한다\n---\n\n한도는 references/limits.md 참고.",
        encoding="utf-8",
    )
    (folder / "references").mkdir()
    (folder / "references" / "limits.md").write_text("식대 한도 2만원", encoding="utf-8")

    monkeypatch.setattr(
        skills,
        "settings",
        dataclasses.replace(settings, skills_dirs=(root,), system_skills_dir=tmp_path / "none"),
    )
    return root


@pytest.mark.parametrize("agent_id", ["claude-code", "codex"])
def test_the_same_skill_lands_where_each_cli_looks_for_it(agent_id, skill_library, tmp_path):
    provider = registry.get(agent_id)
    workdir = tmp_path / "run"
    workdir.mkdir()

    bundle, failures = skills.build_bundle(instructions="# 지시문", skill_names=[])
    result = skills.provision(provider, bundle, workdir, failures)

    written = {p for p in result.paths}
    # Each CLI reads its own instruction filename; neither is hard-coded here.
    assert any(p.endswith(".md") for p in written)
    assert (workdir / provider.layout.constitution_filename).is_file()

    skill_path = provider.layout.skill_path.format(name="expense-policy")
    assert (workdir / skill_path).is_file(), f"{agent_id} cannot find the skill"


@pytest.mark.parametrize("agent_id", ["claude-code", "codex"])
def test_reference_material_travels_with_the_skill(agent_id, skill_library, tmp_path):
    """A skill body that says 'see references/limits.md' is worse than useless
    if that file stayed behind."""
    provider = registry.get(agent_id)
    workdir = tmp_path / "run"
    workdir.mkdir()

    bundle, failures = skills.build_bundle(instructions="", skill_names=["expense-policy"])
    skills.provision(provider, bundle, workdir, failures)

    skill_path = workdir / provider.layout.skill_path.format(name="expense-policy")
    assert (skill_path.parent / "references" / "limits.md").is_file()


def test_codex_gets_the_frontmatter_it_requires(skill_library, tmp_path):
    """Codex refuses a SKILL.md without name and description."""
    provider = registry.get("codex")
    workdir = tmp_path / "run"
    workdir.mkdir()

    bundle, failures = skills.build_bundle(instructions="", skill_names=["expense-policy"])
    skills.provision(provider, bundle, workdir, failures)

    body = (workdir / ".agents/skills/expense-policy/SKILL.md").read_text(encoding="utf-8")
    assert body.startswith("---")
    assert "name: expense-policy" in body
    assert "description:" in body


def test_an_unreachable_git_skill_is_named_and_the_rest_still_run(skill_library, monkeypatch):
    """One dead repository must not cost the run its other skills."""

    def _failing_clone(*_args, **_kwargs):
        return subprocess.CompletedProcess(args=[], returncode=128, stderr="repository not found")

    monkeypatch.setattr(skills.subprocess, "run", _failing_clone)

    bundle, failures = skills.build_bundle(
        instructions="# 지시문",
        skill_names=[],
        git_skills={"remote-skill": "https://example.invalid/none.git"},
    )

    assert "remote-skill" in failures
    assert "git" in failures["remote-skill"]
    # The local skill and the instructions survived the failure.
    names = {a.name for a in bundle.artifacts}
    assert "expense-policy" in names


def test_a_git_skill_that_clones_is_provisioned(skill_library, tmp_path, monkeypatch):
    """The success path, without reaching the network: git is stubbed to
    populate the destination the way a real clone would."""

    def _fake_clone(argv, **_kwargs):
        destination = __import__("pathlib").Path(argv[-1])
        (destination / "references").mkdir(parents=True)
        (destination / "SKILL.md").write_text(
            "---\ndescription: 원격 스킬\n---\n\n본문", encoding="utf-8"
        )
        return subprocess.CompletedProcess(args=argv, returncode=0, stderr="")

    monkeypatch.setattr(skills.subprocess, "run", _fake_clone)

    bundle, failures = skills.build_bundle(
        instructions="",
        skill_names=["remote-skill"],
        git_skills={"remote-skill": "https://example.test/skill.git"},
    )

    assert failures == {}
    assert "remote-skill" in {a.name for a in bundle.artifacts}


def test_an_artifact_the_cli_cannot_place_is_reported_not_dropped(tmp_path, monkeypatch):
    """Silently losing a role definition looks exactly like the agent ignoring it."""
    from cliagents import ArtifactBundle

    monkeypatch.setattr(
        skills,
        "settings",
        dataclasses.replace(settings, skills_dirs=(), system_skills_dir=tmp_path / "none"),
    )

    class _NoRoles(type(registry.get("claude-code"))):
        layout = dataclasses.replace(
            registry.get("claude-code").layout, role_agent_path=None
        )

    workdir = tmp_path / "run"
    workdir.mkdir()
    bundle = ArtifactBundle().add_role_agent("reviewer", "리뷰 담당")

    result = skills.provision(_NoRoles(), bundle, workdir, {})
    assert result.skipped == ["reviewer"]
