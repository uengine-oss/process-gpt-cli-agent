"""Getting instructions and skills in front of the agent, in its own idiom.

ProcessGPT stores a skill once. Claude Code wants it at
``.claude/skills/<name>/SKILL.md``; Codex wants ``.agents/skills/<name>/SKILL.md``
with mandatory frontmatter. Neither of those facts belongs here — they belong
to the providers, which is why this module only decides *what* travels and
leaves *where* to the library.

Sources, in the order they win: bundled system skills (always available, no
network), tenant skills (uploaded), and skills fetched from git. A skill that
fails to arrive is named in the result rather than dropped, because "the agent
ignored my skill" is otherwise indistinguishable from "the skill never got
there".
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from cliagents import ArtifactBundle, DirectorySink

from .settings import settings

logger = logging.getLogger(__name__)

#: Files inside a skill folder that are the skill itself rather than reference
#: material travelling with it.
_SKILL_ENTRY_NAMES = ("SKILL.md", "skill.md", "README.md")

_GIT_TIMEOUT_SECONDS = 60


@dataclass
class Provisioned:
    """What ended up in the workspace, and what did not."""

    paths: list[str] = field(default_factory=list)
    #: Artifacts the chosen CLI has no place for, by name.
    skipped: list[str] = field(default_factory=list)
    #: Reference files dropped because the layout gave them no folder.
    skipped_files: list[str] = field(default_factory=list)
    #: Skills that could not be collected at all, with the reason.
    failed: dict[str, str] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        bits = [f"{len(self.paths)} files"]
        if self.skipped:
            bits.append(f"{len(self.skipped)} unsupported")
        if self.failed:
            bits.append(f"{len(self.failed)} failed")
        return ", ".join(bits)


def build_bundle(
    *,
    instructions: str,
    skill_names: list[str] | None = None,
    git_skills: dict[str, str] | None = None,
) -> tuple[ArtifactBundle, dict[str, str]]:
    """Collect everything the run should be able to read.

    Returns the bundle and the failures, so a caller can report them without
    the bundle having to carry error state.
    """
    bundle = ArtifactBundle()
    failures: dict[str, str] = {}

    if instructions.strip():
        bundle.add_constitution(instructions)

    wanted = set(skill_names or [])
    for name, folder in _discover_local_skills().items():
        # An empty selection means "everything available" — a work item that
        # named no skills should still get the system ones.
        if wanted and name not in wanted:
            continue
        try:
            _add_skill_from_folder(bundle, name, folder)
        except OSError as exc:
            failures[name] = f"읽을 수 없음: {exc}"

    for name, repo in (git_skills or {}).items():
        try:
            _add_skill_from_git(bundle, name, repo)
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            # One unreachable repository must not cost the run its other
            # skills; the agent runs with less rather than not at all.
            failures[name] = f"git 가져오기 실패: {exc}"
            logger.warning("git skill %s could not be fetched: %s", name, exc)

    return bundle, failures


def provision(provider, bundle: ArtifactBundle, workdir: Path, failures: dict[str, str]) -> Provisioned:
    """Write the bundle into ``workdir`` the way ``provider`` expects to read it."""
    result = provider.emit(bundle, DirectorySink(str(workdir)))
    return Provisioned(
        paths=result.paths,
        skipped=[a.name for a in result.skipped],
        skipped_files=list(result.skipped_files),
        failed=dict(failures),
    )


def _discover_local_skills() -> dict[str, Path]:
    """Skill folders on disk, later directories overriding earlier names."""
    found: dict[str, Path] = {}
    roots = [settings.system_skills_dir, *settings.skills_dirs]
    for root in roots:
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if child.is_dir() and _skill_entry(child):
                found[child.name] = child
            elif child.is_file() and child.suffix.lower() == ".md":
                found[child.stem] = child
    return found


def _skill_entry(folder: Path) -> Path | None:
    for name in _SKILL_ENTRY_NAMES:
        candidate = folder / name
        if candidate.is_file():
            return candidate
    return None


def _add_skill_from_folder(bundle: ArtifactBundle, name: str, source: Path) -> None:
    if source.is_file():
        bundle.add_skill(name, source.read_text(encoding="utf-8"), description=_describe(source))
        return

    entry = _skill_entry(source)
    if entry is None:
        return

    # Reference material travels with the skill: a body that says "see
    # references/parsing.md" is worse than useless if that file stayed behind.
    companions: dict[str, str] = {}
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path == entry:
            continue
        if path.suffix.lower() not in (".md", ".txt", ".json", ".yaml", ".yml", ".csv"):
            continue
        try:
            companions[path.relative_to(source).as_posix()] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

    bundle.add_skill(
        name,
        entry.read_text(encoding="utf-8"),
        description=_describe(entry),
        files=companions,
    )


def _add_skill_from_git(bundle: ArtifactBundle, name: str, repo: str) -> None:
    """Shallow-clone ``repo`` into a temp dir and take the skill out of it."""
    with tempfile.TemporaryDirectory(prefix="cliagents-skill-") as tmp:
        target = Path(tmp) / name
        completed = subprocess.run(
            ["git", "clone", "--depth", "1", repo, str(target)],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr or "").strip().splitlines()[-1:] or "clone failed")
        _add_skill_from_folder(bundle, name, target)


def _describe(entry: Path) -> str:
    """Pull the description out of frontmatter, since Codex requires one."""
    try:
        head = entry.read_text(encoding="utf-8")[:2000]
    except (OSError, UnicodeDecodeError):
        return ""
    for line in head.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("description:"):
            return stripped.split(":", 1)[1].strip()
    return ""


def seed_system_skills(source: Path | None = None) -> int:
    """Copy bundled skills into the configured skills directory.

    Called at startup so an air-gapped deployment has the same skills as a
    connected one. Returns how many were seeded.
    """
    origin = source or settings.system_skills_dir
    if not origin.is_dir() or not settings.skills_dirs:
        return 0

    destination = settings.skills_dirs[0] / "process-gpt-system"
    destination.mkdir(parents=True, exist_ok=True)

    seeded = 0
    for child in sorted(origin.iterdir()):
        target = destination / child.name
        try:
            if child.is_dir():
                shutil.copytree(child, target, dirs_exist_ok=True)
            else:
                shutil.copy2(child, target)
            seeded += 1
        except OSError:
            logger.warning("could not seed system skill %s", child.name, exc_info=True)
    return seeded
