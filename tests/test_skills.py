"""One skill, stored once, readable by whichever CLI runs the job.

This is the claim the whole artifact surface exists to make, so it is tested
against both providers rather than asserted in a docstring.
"""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path

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


def test_a_tenant_namespaced_skill_is_found(tmp_path, monkeypatch):
    """Uploads land under a tenant folder, and a top-level-only scan misses
    them entirely — which reads as "the agent ignored my skill"."""
    root = tmp_path / "skills"
    folder = root / "acme" / "expense-policy"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text("---\ndescription: 경비\n---\n본문", encoding="utf-8")

    monkeypatch.setattr(
        skills,
        "settings",
        dataclasses.replace(settings, skills_dirs=(root,), system_skills_dir=tmp_path / "none"),
    )

    assert skills._discover_local_skills() == {"expense-policy": folder}


def test_a_skills_own_subfolders_are_not_mistaken_for_skills(tmp_path, monkeypatch):
    """A skill whose entry file is a README must not have its references
    directory registered as a second skill."""
    root = tmp_path / "skills"
    folder = root / "guide"
    (folder / "references").mkdir(parents=True)
    (folder / "README.md").write_text("본문", encoding="utf-8")
    (folder / "references" / "README.md").write_text("참고", encoding="utf-8")

    monkeypatch.setattr(
        skills,
        "settings",
        dataclasses.replace(settings, skills_dirs=(root,), system_skills_dir=tmp_path / "none"),
    )

    assert sorted(skills._discover_local_skills()) == ["guide"]


def test_a_skills_scripts_travel_with_it(tmp_path, monkeypatch):
    """A skill that tells the agent to run scripts/validate.py needs the
    script in the workspace, not just the sentence naming it."""
    root = tmp_path / "skills"
    folder = root / "process-gen"
    (folder / "scripts" / "validation").mkdir(parents=True)
    (folder / "scripts" / "__pycache__").mkdir()
    (folder / "SKILL.md").write_text(
        "---\ndescription: 프로세스 생성\n---\n\nscripts/validate.py 를 실행한다.", encoding="utf-8"
    )
    (folder / "scripts" / "validate.py").write_text("print('ok')", encoding="utf-8")
    (folder / "scripts" / "validation" / "__init__.py").write_text("", encoding="utf-8")
    (folder / "scripts" / "__pycache__" / "validate.cpython-313.pyc").write_bytes(b"\x00stale")

    monkeypatch.setattr(
        skills,
        "settings",
        dataclasses.replace(settings, skills_dirs=(root,), system_skills_dir=tmp_path / "none"),
    )

    provider = registry.get("claude-code")
    workdir = tmp_path / "run"
    workdir.mkdir()
    bundle, failures = skills.build_bundle(instructions="", skill_names=["process-gen"])
    skills.provision(provider, bundle, workdir, failures)

    placed = workdir / provider.layout.skill_path.format(name="process-gen")
    assert (placed.parent / "scripts" / "validate.py").is_file()
    assert (placed.parent / "scripts" / "validation" / "__init__.py").is_file()
    # Compiled bytecode from another interpreter is noise at best.
    assert not (placed.parent / "scripts" / "__pycache__").exists()


def test_the_bundled_process_generation_skill_is_available_to_every_run(tmp_path, monkeypatch):
    """The claim this clone exists to make: cliagents offers the same process
    generation skill deepagents does, without anyone uploading it."""
    bundled = Path(__file__).resolve().parents[1] / "system-skills"
    if not (bundled / "bpmn-process-generation-skill" / "SKILL.md").is_file():
        pytest.skip("bundled system skills are not present in this checkout")

    monkeypatch.setattr(
        skills,
        "settings",
        dataclasses.replace(settings, skills_dirs=(tmp_path / "empty",), system_skills_dir=bundled),
    )

    provider = registry.get("claude-code")
    workdir = tmp_path / "run"
    workdir.mkdir()
    bundle, failures = skills.build_bundle(instructions="")
    skills.provision(provider, bundle, workdir, failures)

    placed = workdir / provider.layout.skill_path.format(name="bpmn-process-generation-skill")
    assert placed.is_file()
    # The body defers to these; arriving without them makes it unusable.
    assert (placed.parent / "references" / "12-deepagents-execution.md").is_file()
    assert (placed.parent / "assets" / "templates" / "process-definition.schema.json").is_file()


def test_the_artifact_path_section_names_this_runs_folder():
    """A skill documents its output file *names*; only the service knows the
    directory. Unsaid, the agent invents one and the definition lands where
    nothing looks for it."""
    from core import prompt

    section = prompt.artifact_paths("C:/ws/localhost/run-1", "run-1")
    folder = prompt.process_dir_name("run-1")

    assert "C:/ws/localhost/run-1/.bpmn/" in section
    assert folder in section
    # Derived, not drawn: a retry must write into the first attempt's folder.
    assert prompt.process_dir_name("run-1") == folder
    assert prompt.process_dir_name("run-2") != folder


def test_a_produced_file_is_reported_with_a_posix_path(tmp_path):
    """The path crosses to a browser that splits on `/` to build its folder
    tree; a Windows separator arrives as one long filename instead."""
    from core import workspace as workspace_module

    ws = workspace_module.Workspace(path=tmp_path, run_id="run-1")
    nested = tmp_path / ".bpmn" / "process-abc" / "forms"
    nested.mkdir(parents=True)
    target = nested / "apply.form"
    target.write_text("x", encoding="utf-8")

    assert ws.relative(target) == ".bpmn/process-abc/forms/apply.form"


def test_a_skill_chosen_in_the_designer_but_missing_here_is_named(skill_library):
    """The silent version of this is a worse answer and no explanation: the run
    proceeds, the agent simply never sees the skill."""
    bundle, failures = skills.build_bundle(
        instructions="", skill_names=["expense-policy", "hwpx-writer"]
    )

    assert "expense-policy" in {a.name for a in bundle.artifacts}
    assert "hwpx-writer" in failures
    assert "없습니다" in failures["hwpx-writer"]


def test_naming_no_skills_still_gets_the_bundled_ones(skill_library):
    """An activity that chose nothing should still get what the server ships."""
    bundle, failures = skills.build_bundle(instructions="", skill_names=[])

    assert failures == {}
    assert "expense-policy" in {a.name for a in bundle.artifacts}
