"""Turning a work item into something worth handing an agent.

A CLI agent gets one prompt and whatever is on disk. Everything the LangGraph
orchestration would have injected as tool context — who asked, what the form
wants back, what the reviewer objected to last time — has to be *said*, or it
does not exist.

The output contract gets the most attention because it is the part that fails
silently: an agent that writes a beautiful essay when the form wanted three
fields has done the job wrong in a way no exception reports.
"""

from __future__ import annotations

import json
from typing import Any

#: Enough of a large input to be useful, little enough to leave room for the
#: work. Inputs past this are written to the workspace instead and referenced.
_INLINE_INPUT_LIMIT = 8_000


def build(row: dict[str, Any], extras: dict[str, Any], *, workdir: str) -> str:
    """The prompt for one work item."""
    sections: list[str] = []

    activity = (row.get("activity_name") or "").strip()
    query = (row.get("query") or "").strip()

    sections.append(
        "당신은 ProcessGPT 업무 프로세스의 담당 에이전트입니다. "
        "아래 업무를 끝까지 수행하고, 마지막에 결과를 제시하세요."
    )
    if activity:
        sections.append(f"## 업무\n{activity}")
    if query:
        sections.append(f"## 지시사항\n{query}")

    participants = _participants(extras)
    if participants:
        sections.append(f"## 참여자\n{participants}")

    inputs = _inputs(row, extras)
    if inputs:
        sections.append(f"## 입력 데이터\n{inputs}")

    feedback = (extras.get("summarized_feedback") or "").strip()
    if feedback:
        # Feedback means a previous attempt was rejected. Saying so is the
        # difference between a revision and the same answer again.
        sections.append(
            "## 이전 결과에 대한 피드백\n"
            f"{feedback}\n\n"
            "이 피드백을 반영해 결과를 수정하세요. 같은 결과를 반복하지 마세요."
        )

    sources = _sources(extras)
    if sources:
        sections.append(f"## 참고 자료\n{sources}")

    sections.append(
        "## 작업 공간\n"
        f"작업 디렉터리는 `{workdir}` 입니다. 산출 파일은 이 디렉터리 안에 만드세요. "
        "이 디렉터리 밖에는 쓰지 마세요."
    )

    sections.append(output_contract(extras.get("form_fields")))
    return "\n\n".join(s for s in sections if s.strip())


def output_contract(form_fields: Any) -> str:
    """How the agent is told to close.

    With a form, the ask is a strict JSON object — parsing prose into fields
    guesses, and a guess stored as a business record is worse than a failure.
    Without one, the ask is plain text, because inventing a schema the process
    never defined would fail every run for not matching it.
    """
    fields = _field_list(form_fields)
    if not fields:
        return (
            "## 결과 제출 형식\n"
            "작업을 마치면 마지막 메시지에 최종 결과 본문을 그대로 작성하세요."
        )

    described = "\n".join(
        f"- `{f['key']}`: {f.get('label') or f['key']}"
        + (f" (형식: {f['type']})" if f.get("type") else "")
        for f in fields
    )
    skeleton = json.dumps({f["key"]: "" for f in fields}, ensure_ascii=False, indent=2)
    return (
        "## 결과 제출 형식\n"
        "작업을 마치면 **마지막 메시지에 아래 JSON 객체 하나만** 출력하세요. "
        "설명 문장이나 코드펜스 밖 텍스트를 함께 쓰지 마세요.\n\n"
        f"필드:\n{described}\n\n"
        f"형태:\n```json\n{skeleton}\n```"
    )


def _field_list(form_fields: Any) -> list[dict[str, Any]]:
    """Normalise the form definition, which arrives in more than one shape."""
    if isinstance(form_fields, str):
        try:
            form_fields = json.loads(form_fields)
        except json.JSONDecodeError:
            return []
    if isinstance(form_fields, dict):
        form_fields = form_fields.get("fields") or form_fields.get("properties") or []
    if not isinstance(form_fields, list):
        return []

    out: list[dict[str, Any]] = []
    for field in form_fields:
        if not isinstance(field, dict):
            continue
        key = field.get("key") or field.get("name") or field.get("id")
        if not key:
            continue
        out.append(
            {
                "key": str(key),
                "label": field.get("label") or field.get("title") or "",
                "type": field.get("type") or "",
            }
        )
    return out


def field_keys(form_fields: Any) -> list[str]:
    return [f["key"] for f in _field_list(form_fields)]


def _participants(extras: dict[str, Any]) -> str:
    lines: list[str] = []
    for user in extras.get("users") or []:
        if isinstance(user, dict):
            name = user.get("name") or user.get("username") or ""
            if name:
                lines.append(f"- 담당자: {name}")
    for agent in extras.get("agents") or []:
        if isinstance(agent, dict):
            name = agent.get("name") or agent.get("username") or ""
            role = agent.get("role") or agent.get("goal") or ""
            if name:
                lines.append(f"- 에이전트: {name}{f' ({role})' if role else ''}")
    return "\n".join(lines)


def _inputs(row: dict[str, Any], extras: dict[str, Any]) -> str:
    payload = row.get("output") or row.get("draft") or extras.get("sensitive_data")
    if not payload:
        return ""
    if not isinstance(payload, str):
        payload = json.dumps(payload, ensure_ascii=False, indent=2)
    if len(payload) > _INLINE_INPUT_LIMIT:
        return (
            payload[:_INLINE_INPUT_LIMIT]
            + f"\n… (이하 {len(payload) - _INLINE_INPUT_LIMIT}자 생략)"
        )
    return payload


def _sources(extras: dict[str, Any]) -> str:
    lines: list[str] = []
    for source in extras.get("sources") or []:
        if not isinstance(source, dict):
            continue
        name = source.get("file_name") or ""
        path = source.get("file_path") or ""
        if name or path:
            lines.append(f"- {name} {f'({path})' if path else ''}".strip())
    return "\n".join(lines)
