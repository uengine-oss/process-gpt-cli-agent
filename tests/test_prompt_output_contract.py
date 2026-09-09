"""출력 계약이 폼의 라벨과 허용값을 실제로 실어 보내는지.

회귀: form_def.fields_json 은 라벨을 `text` 로, 선택지를 `items` 로 적는데
프롬프트 빌더는 `label`/`title` 만 읽고 `items` 는 통째로 버렸다. 그래서
에이전트는 `- review_required: review_required (형식: select)` 만 보고 값을
지어냈고('예', 'Y', ...), 게이트웨이의 결정론 분기 조건과 어긋나 매번 LLM
폴백으로 넘어갔다.
"""

from __future__ import annotations

import json

import pytest

from core.prompt import _choices, field_keys, output_contract

SELECT_FIELD = {
    "key": "review_required",
    "text": "프리세일즈 검토 필요 여부",
    "type": "select",
    "items": [{"required": "검토 필요"}, {"not_required": "검토 불필요"}],
}
TEXT_FIELD = {"key": "proposal_title", "text": "제안서 제목", "type": "text"}


def test_no_fields_asks_for_plain_text():
    assert "최종 결과 본문" in output_contract(None)


def test_label_comes_from_text_key():
    contract = output_contract([TEXT_FIELD])
    assert "제안서 제목" in contract


def test_select_allowed_values_are_listed():
    contract = output_contract([SELECT_FIELD])
    assert "`required`" in contract
    assert "`not_required`" in contract
    assert "검토 필요" in contract
    assert "허용값" in contract


def test_select_gets_do_not_paraphrase_instruction():
    contract = output_contract([SELECT_FIELD])
    assert "그대로" in contract
    assert "'Y'" in contract or "Y" in contract


def test_no_choice_hint_when_no_select_field():
    assert "허용값" not in output_contract([TEXT_FIELD])


def test_skeleton_still_lists_every_key():
    contract = output_contract([TEXT_FIELD, SELECT_FIELD])
    skeleton = contract.split("```json")[1].split("```")[0]
    assert json.loads(skeleton) == {"proposal_title": "", "review_required": ""}


def test_field_keys_unchanged():
    assert field_keys([TEXT_FIELD, SELECT_FIELD]) == ["proposal_title", "review_required"]


@pytest.mark.parametrize(
    "raw,expected",
    [
        ([{"required": "검토 필요"}], [("required", "검토 필요")]),
        ([{"value": "a", "label": "에이"}], [("a", "에이")]),
        ([{"key": "b", "text": "비"}], [("b", "비")]),
        (["x", "y"], [("x", "x"), ("y", "y")]),
        ("[{'예':'검토 필요'}]", [("예", "검토 필요")]),
        (None, []),
        ("not json", []),
        ({"a": 1}, []),
    ],
)
def test_choices_shapes(raw, expected):
    assert _choices(raw) == expected


def test_legacy_label_key_still_works():
    contract = output_contract([{"key": "k", "label": "옛 라벨", "type": "text"}])
    assert "옛 라벨" in contract
