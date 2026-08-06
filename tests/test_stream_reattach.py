"""Rejoining a run in progress.

The failure this guards against is quiet: the run finishes fine, and the user
watches a spinner that will never resolve because their connection dropped and
nothing let them back in.
"""

from __future__ import annotations

import asyncio

import pytest

from core.stream_registry import StreamRegistry


async def _collect(stream, limit: int) -> list[dict]:
    out: list[dict] = []
    async for event in stream:
        out.append(event)
        if len(out) >= limit:
            break
    return out


async def test_reattaching_replays_what_was_already_sent():
    registry = StreamRegistry()
    registry.start("run-1")
    registry.publish("run-1", {"type": "text", "content": "작업을 시작합니다"})
    registry.publish("run-1", {"type": "tool_start", "tool": "Write"})

    replayed = await _collect(registry.attach("run-1"), 2)

    assert [e["type"] for e in replayed] == ["text", "tool_start"]


async def test_a_reattached_client_then_follows_live():
    registry = StreamRegistry()
    registry.start("run-2")
    registry.publish("run-2", {"type": "text", "content": "이전 내용"})

    stream = registry.attach("run-2")
    first = await anext(stream)
    assert first["content"] == "이전 내용"

    async def _later():
        await asyncio.sleep(0.05)
        registry.publish("run-2", {"type": "text", "content": "새 내용"})

    asyncio.create_task(_later())
    live = await asyncio.wait_for(anext(stream), timeout=2)
    assert live["content"] == "새 내용"


async def test_a_finished_run_closes_the_stream_instead_of_hanging():
    registry = StreamRegistry()
    registry.start("run-3")
    registry.publish("run-3", {"type": "text", "content": "끝났습니다"})
    registry.finish("run-3")

    events = await asyncio.wait_for(_collect(registry.attach("run-3"), 2), timeout=2)
    assert events[-1]["type"] == "done"


async def test_attaching_to_an_unknown_run_says_so_rather_than_waiting():
    """A client left hanging on a stream that will never produce cannot tell it
    apart from a working one, and never falls back."""
    registry = StreamRegistry()
    with pytest.raises(KeyError):
        await anext(registry.attach("never-existed"))


async def test_two_listeners_both_receive_live_events():
    registry = StreamRegistry()
    registry.start("run-4")

    first = registry.attach("run-4")
    second = registry.attach("run-4")
    # Attach both before publishing: the generator body does not run until
    # it is first advanced, which is when the listener actually registers.
    task_a = asyncio.create_task(anext(first))
    task_b = asyncio.create_task(anext(second))
    await asyncio.sleep(0.05)

    registry.publish("run-4", {"type": "text", "content": "공유 이벤트"})
    got_a, got_b = await asyncio.wait_for(asyncio.gather(task_a, task_b), timeout=2)

    assert got_a["content"] == got_b["content"] == "공유 이벤트"


async def test_a_long_run_truncates_its_backlog_and_admits_it():
    registry = StreamRegistry()
    registry.start("run-5")
    for i in range(600):
        registry.publish("run-5", {"type": "text", "content": f"청크 {i}"})

    events = await _collect(registry.attach("run-5"), 3)

    assert events[0]["type"] == "notice"
    assert "생략" in events[0]["content"]
    # What survives is the recent end, which is what a reattaching user wants.
    assert events[1]["content"] != "청크 0"


async def test_a_restarted_run_supersedes_the_previous_stream():
    registry = StreamRegistry()
    registry.start("run-6")
    registry.publish("run-6", {"type": "text", "content": "첫 시도"})

    registry.start("run-6")
    registry.publish("run-6", {"type": "text", "content": "재시도"})

    events = await _collect(registry.attach("run-6"), 1)
    assert events[0]["content"] == "재시도"
