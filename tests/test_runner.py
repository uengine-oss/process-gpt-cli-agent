"""Running a CLI from an event loop: cancellation, timeout, capacity.

The failure these guard against is a CLI process that outlives the work item it
was started for — still writing files, still spending money, attached to
nothing.
"""

from __future__ import annotations

import asyncio
import io
import json

import pytest
from cliagents import ExecEventKind, ExecRequest, registry

from core.runner import ConcurrencyLimit, RunTimeoutError, astream

pytestmark = pytest.mark.asyncio


class _SlowProc:
    """A child that emits a line, then stalls — the shape of a hung agent."""

    def __init__(self, lines: list[str], *, stall: bool = False) -> None:
        self._lines = lines
        self._stall = stall
        self.stdout = self
        self.stderr = io.StringIO("")
        self.returncode = 0
        self.terminated = False
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self._lines):
            line = self._lines[self._index]
            self._index += 1
            return line
        if self._stall and not self.terminated:
            # Block the pump thread the way a live pipe would.
            import time

            while not self.terminated:
                time.sleep(0.01)
        raise StopIteration

    def close(self):
        pass

    def poll(self):
        return None if (self._stall and not self.terminated) else self.returncode

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def kill(self):  # pragma: no cover - only if terminate is ignored
        self.terminated = True


def _lines(*payloads: dict) -> list[str]:
    return [json.dumps(p) + "\n" for p in payloads]


async def test_events_reach_the_loop_in_order():
    provider = registry.get("claude-code")
    proc = _SlowProc(
        _lines(
            {"type": "system", "subtype": "init", "session_id": "s1", "model": "m"},
            {"type": "result", "subtype": "success", "result": "done", "session_id": "s1"},
        )
    )
    kinds = [
        event.kind
        async for event in astream(provider, ExecRequest(prompt="hi"), **popen_for_test(proc))
    ]
    assert kinds == [ExecEventKind.RUN_START, ExecEventKind.RESULT]


async def test_a_hung_run_is_stopped_by_its_timeout():
    provider = registry.get("claude-code")
    proc = _SlowProc(
        _lines({"type": "system", "subtype": "init", "session_id": "s1", "model": "m"}),
        stall=True,
    )

    with pytest.raises(RunTimeoutError):
        async for _event in astream(
            provider, ExecRequest(prompt="hi"), timeout=0.3, **popen_for_test(proc)
        ):
            pass

    assert proc.terminated, "a timed-out run must not leave the CLI process running"


async def test_cancelling_the_consumer_terminates_the_child():
    provider = registry.get("claude-code")
    proc = _SlowProc(
        _lines({"type": "system", "subtype": "init", "session_id": "s1", "model": "m"}),
        stall=True,
    )

    async def _consume():
        async for _event in astream(provider, ExecRequest(prompt="hi"), **popen_for_test(proc)):
            pass

    task = asyncio.create_task(_consume())
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert proc.terminated, "a cancelled work item must not leave the CLI process running"


async def test_capacity_is_reported_before_a_slot_is_taken():
    """The poller asks *before* claiming: claiming and then queueing parks a
    work item in 'running' behind a queue nobody can see."""
    limit = ConcurrencyLimit(1)
    assert not limit.at_capacity()

    async with limit:
        assert limit.at_capacity()
        assert limit.in_flight == 1

    assert not limit.at_capacity()


async def test_the_cap_actually_serialises_runs():
    limit = ConcurrencyLimit(1)
    order: list[str] = []

    async def _run(name: str):
        async with limit:
            order.append(f"{name}:start")
            await asyncio.sleep(0.05)
            order.append(f"{name}:end")

    await asyncio.gather(_run("a"), _run("b"))
    assert order in (
        ["a:start", "a:end", "b:start", "b:end"],
        ["b:start", "b:end", "a:start", "a:end"],
    )


def popen_for_test(proc):
    """Keyword bundle that hands ``astream`` a canned child process."""

    def _popen(_argv, **_kwargs):
        return proc

    return {"env": None, "popen": _popen}
