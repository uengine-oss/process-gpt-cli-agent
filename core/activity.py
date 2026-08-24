"""What the designer chose for this activity, read from the process definition.

The work item row does not carry it. `todolist` has no column for the skills an
activity was given or for the CLI it was configured to run on — those live in
`proc_def.definition.activities[]`, written by the designer panel. A service
that only reads the row therefore ignores every choice made on that screen and
silently falls back to its defaults, which looks exactly like the screen not
working.

So the definition is read here, the same way the deepagents orchestration reads
it, and every failure returns "nothing configured" rather than raising: a run
with default settings is worse than a configured one, but far better than no
run at all.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

#: Where the activity id sits, across the shapes the definition has had.
_ID_KEYS = ("id", "activity_id", "activityId", "key")

#: Where the CLI choice sits. The panel writes `agentConfig`; the snake_case
#: spelling is accepted because other writers have used it.
_CONFIG_KEYS = ("agentConfig", "agent_config")


@dataclass(frozen=True)
class Capabilities:
    """The activity's declared skills and CLI settings. Empty when unset."""

    skills: list[str] = field(default_factory=list)
    agent_config: dict[str, Any] = field(default_factory=dict)


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [s.strip() for s in value.split(",") if s.strip()]
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return []


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, str) and value.strip():
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def extract(definition: Any, activity_id: str) -> Capabilities:
    """Pull one activity's settings out of a definition. Pure, so it is testable
    without a database — which is where the shape mistakes actually happen."""
    target_id = (activity_id or "").strip()
    if not target_id or not isinstance(definition, dict):
        return Capabilities()

    # `activities` is what the engine stores; `elements` is what the process
    # generation skill writes before the engine normalises it. Read both, or a
    # freshly generated definition silently has no skills.
    for key in ("activities", "elements"):
        items = definition.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            if any(str(item.get(k) or "").strip() == target_id for k in _ID_KEYS):
                config: dict[str, Any] = {}
                for config_key in _CONFIG_KEYS:
                    config = _as_dict(item.get(config_key))
                    if config:
                        break
                return Capabilities(
                    skills=list(dict.fromkeys(_as_str_list(item.get("skills")))),
                    agent_config=config,
                )
    return Capabilities()


def for_work_item(row: dict[str, Any]) -> Capabilities:
    """Look up the definition this work item came from."""
    tenant_id = str(row.get("tenant_id") or "").strip()
    proc_def_id = str(row.get("proc_def_id") or "").strip()
    activity_id = str(row.get("activity_id") or "").strip()
    if not (tenant_id and proc_def_id and activity_id):
        return Capabilities()

    try:
        from processgpt_agent_sdk.database import get_db_client

        response = (
            get_db_client()
            .table("proc_def")
            .select("definition")
            .eq("tenant_id", tenant_id)
            .eq("id", proc_def_id)
            .limit(1)
            .execute()
        )
    except Exception:  # noqa: BLE001 - a lookup failure must not fail the run
        logger.exception(
            "activity settings lookup failed | tenant=%s proc_def=%s activity=%s",
            tenant_id,
            proc_def_id,
            activity_id,
        )
        return Capabilities()

    rows = getattr(response, "data", None)
    if not isinstance(rows, list) or not rows:
        return Capabilities()

    definition = rows[0].get("definition") if isinstance(rows[0], dict) else None
    if isinstance(definition, str):
        try:
            definition = json.loads(definition)
        except json.JSONDecodeError:
            return Capabilities()

    return extract(definition, activity_id)
