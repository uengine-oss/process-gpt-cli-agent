"""Reading the agent's last message as a business result.

The contract asked for a JSON object when the process defined a form. This
module decides whether it got one — and refuses to pretend otherwise. A form
field filled with a shrug is indistinguishable, downstream, from a field a
human filled in, and the process will act on it.

So: parse honestly, report the mismatch, and keep the raw text either way so a
reviewer can see what the agent actually said.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .prompt import field_keys

#: ```json … ``` — the fence the model reaches for even when told not to.
_FENCE = re.compile(r"```(?:json)?\s*(?P<body>\{.*?\})\s*```", re.DOTALL)


@dataclass
class Outcome:
    """What a finished run produced, and whether it honoured the contract."""

    raw_text: str
    #: Parsed form outputs, empty when the process defined no form.
    outputs: dict[str, Any] = field(default_factory=dict)
    #: True when a contract existed and was met.
    contract_met: bool = True
    #: Why not, in words a reviewer can act on.
    mismatch_reason: str = ""
    missing_fields: list[str] = field(default_factory=list)

    @property
    def payload(self) -> dict[str, Any]:
        """What gets stored on the work item.

        The raw text rides along even on success: the form holds the answer,
        the text holds the reasoning, and reviewers want both.
        """
        body: dict[str, Any] = {"text": self.raw_text}
        if self.outputs:
            body.update(self.outputs)
        return body


def interpret(final_text: str, form_fields: Any) -> Outcome:
    """Read ``final_text`` against the form the process defined."""
    text = (final_text or "").strip()
    keys = field_keys(form_fields)

    if not keys:
        # No form, no contract to break. Prose is the deliverable.
        return Outcome(raw_text=text)

    if not text:
        return Outcome(
            raw_text=text,
            contract_met=False,
            mismatch_reason="에이전트가 결과를 반환하지 않았습니다.",
            missing_fields=keys,
        )

    parsed = _extract_object(text)
    if parsed is None:
        return Outcome(
            raw_text=text,
            contract_met=False,
            mismatch_reason=(
                "결과가 JSON 객체가 아닙니다. 폼 필드에 채울 값을 판별할 수 없습니다."
            ),
            missing_fields=keys,
        )

    missing = [k for k in keys if k not in parsed or _is_blank(parsed.get(k))]
    outputs = {k: parsed[k] for k in keys if k in parsed}

    if missing:
        return Outcome(
            raw_text=text,
            outputs=outputs,
            contract_met=False,
            mismatch_reason=f"필수 폼 필드가 비어 있습니다: {', '.join(missing)}",
            missing_fields=missing,
        )

    return Outcome(raw_text=text, outputs=outputs)


def _extract_object(text: str) -> dict[str, Any] | None:
    """Find the result object, tolerating the wrappers models add.

    Tried in order of confidence: the whole message, a fenced block, then the
    last balanced ``{…}`` in the text — last, because when a model explains
    itself and *then* answers, the answer is at the end.
    """
    direct = _load(text)
    if direct is not None:
        return direct

    fenced = _FENCE.findall(text)
    for candidate in reversed(fenced):
        parsed = _load(candidate)
        if parsed is not None:
            return parsed

    for candidate in reversed(_balanced_objects(text)):
        parsed = _load(candidate)
        if parsed is not None:
            return parsed
    return None


def _load(candidate: str) -> dict[str, Any] | None:
    try:
        value = json.loads(candidate.strip())
    except (json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _balanced_objects(text: str) -> list[str]:
    """Top-level ``{…}`` spans, brace-counted rather than regexed."""
    found: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False

    for i, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = i
            depth += 1
        elif char == "}":
            if depth:
                depth -= 1
                if depth == 0 and start >= 0:
                    found.append(text[start : i + 1])
    return found


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return False
