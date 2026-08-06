"""Run the E2E scenarios against real infrastructure.

Deliberately thin: it seeds a work item, lets the real executor handle it, and
asserts on what the database and the workspace hold afterwards. Everything
interesting is in the service, not here.

Refuses to run rather than pretending — no database, no CLI, no run. A suite
that "passes" because it skipped everything is worse than one that fails.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid

from cliagents import Surface, registry

from core.selection import CliSelectionError, require_runnable, resolve
from core.workspace import for_run
from e2e import scenarios


class SuiteError(Exception):
    """A precondition is missing. Reported, never worked around."""


def _require_environment() -> None:
    missing = [name for name in ("SUPABASE_URL", "SUPABASE_KEY") if not os.getenv(name)]
    if missing:
        raise SuiteError(
            f"missing environment: {', '.join(missing)}. "
            "Start the infrastructure with docker compose -f e2e/docker-compose.e2e.yml up -d "
            "and export the values from .env.e2e."
        )

    installed = [p.id for p in registry.for_surface(Surface.EXEC) if p.detect(refresh=True).installed]
    if not installed:
        raise SuiteError(
            "no CLI agent is installed here — these tests exercise real CLIs, so "
            "there is nothing to verify. Install Claude Code or Codex first."
        )
    print(f"CLI agents available: {', '.join(installed)}")


async def _run_scenario(scenario: scenarios.Scenario) -> bool:
    print(f"\n=== {scenario.id}: {scenario.description}")

    run_id = f"e2e-{scenario.id}-{uuid.uuid4().hex[:8]}"
    row = {**scenario.work_item, "id": run_id, "tenant_id": "e2e"}

    # The one assertion that needs no infrastructure and no credits: a work item
    # naming an agent this deployment cannot run must fail, and must not quietly
    # become a different agent.
    if scenario.id == "missing-cli":
        try:
            require_runnable(resolve(row))
        except CliSelectionError as exc:
            assert exc.agent_id == "gemini-cli", exc
            print(f"  ✓ refused with guidance: {exc}")
            return True
        print("  ✗ an unavailable CLI was accepted — something was substituted")
        return False

    workspace = for_run(run_id, tenant_id="e2e")
    print(f"  workspace: {workspace.path}")
    print("  ! not implemented: seed the work item and let the poller claim it")
    for expectation in scenario.expects:
        print(f"    · expected: {expectation}")
    return False


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="run one scenario by id")
    parser.add_argument(
        "--free-only",
        action="store_true",
        help="skip scenarios that spend model credits",
    )
    args = parser.parse_args()

    chosen = [scenarios.by_id(args.only)] if args.only else scenarios.ALL
    if args.free_only:
        chosen = [s for s in chosen if not s.spends_tokens]

    if not args.free_only:
        _require_environment()

    results = {s.id: await _run_scenario(s) for s in chosen}

    print("\n=== summary")
    for scenario_id, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {scenario_id}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except SuiteError as error:
        print(f"cannot run: {error}", file=sys.stderr)
        sys.exit(2)
