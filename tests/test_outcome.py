"""Reading a result honestly — the failure mode here is a *stored* wrong answer."""

from __future__ import annotations

from core.outcome import interpret

FORM = [
    {"key": "summary", "label": "요약", "type": "text"},
    {"key": "amount", "label": "금액", "type": "number"},
]


def test_no_form_means_the_prose_is_the_deliverable():
    outcome = interpret("검토를 마쳤습니다.", None)
    assert outcome.contract_met
    assert outcome.payload["text"] == "검토를 마쳤습니다."
    assert outcome.outputs == {}


def test_a_clean_json_answer_fills_the_form():
    outcome = interpret('{"summary": "완료", "amount": 1000}', FORM)
    assert outcome.contract_met
    assert outcome.outputs == {"summary": "완료", "amount": 1000}
    # The prose is kept even on success — reviewers want the reasoning too.
    assert outcome.payload["text"]


def test_a_fenced_answer_still_counts():
    text = '설명입니다.\n\n```json\n{"summary": "완료", "amount": 5}\n```'
    outcome = interpret(text, FORM)
    assert outcome.contract_met
    assert outcome.outputs["amount"] == 5


def test_the_last_object_wins_when_the_model_thinks_out_loud():
    text = '먼저 {"summary": "초안"} 을 고려했으나 최종은 {"summary": "완료", "amount": 7} 입니다.'
    outcome = interpret(text, FORM)
    assert outcome.outputs["summary"] == "완료"


def test_prose_where_a_form_was_required_is_a_mismatch_not_an_empty_form():
    outcome = interpret("아주 훌륭한 보고서입니다.", FORM)
    assert not outcome.contract_met
    assert outcome.outputs == {}
    assert outcome.missing_fields == ["summary", "amount"]
    assert "JSON" in outcome.mismatch_reason


def test_a_half_filled_form_is_reported_with_the_field_names():
    outcome = interpret('{"summary": "완료", "amount": ""}', FORM)
    assert not outcome.contract_met
    assert outcome.missing_fields == ["amount"]
    # What did arrive is kept, so the reviewer starts from partial work.
    assert outcome.outputs["summary"] == "완료"


def test_an_empty_answer_is_a_mismatch_not_a_success():
    outcome = interpret("", FORM)
    assert not outcome.contract_met
    assert outcome.missing_fields == ["summary", "amount"]


def test_braces_inside_strings_do_not_break_extraction():
    outcome = interpret('{"summary": "a } b {", "amount": 1}', FORM)
    assert outcome.contract_met
    assert outcome.outputs["summary"] == "a } b {"
