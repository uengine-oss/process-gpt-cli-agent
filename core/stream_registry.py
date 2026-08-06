"""Letting a browser rejoin a run that is still going.

A CLI agent run takes minutes. Laptops sleep, tabs get reloaded, wifi drops —
and without somewhere to rejoin, the user sees a dead spinner while the work
they are waiting for finishes invisibly in a container.

So a run in progress is registered here, keeping what it has already emitted and
a fan-out to whoever is listening now. Reattaching replays the backlog, then
follows live.

**This registry is process-local, and that is a constraint, not an oversight.**
A second uvicorn worker would answer "no such run" for a stream the first one is
still serving. The failure is soft — the client falls back to stored messages —
but the live catch-up is gone, silently. Run one worker, or move this to shared
storage before adding a second.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

#: Keeps a reattach useful without letting one chatty run pin unbounded memory.
#: A run that emitted more than this replays its most recent events only.
_MAX_BACKLOG_EVENTS = 500

#: How long a finished run stays reattachable. Long enough for the "I reloaded
#: right as it finished" case, short enough not to accumulate.
_LINGER_SECONDS = 60


@dataclass
class _Run:
    run_id: str
    backlog: list[dict] = field(default_factory=list)
    listeners: set[asyncio.Queue] = field(default_factory=set)
    done: bool = False
    finished_at: float | None = None
    truncated: bool = False

    def add(self, event: dict) -> None:
        self.backlog.append(event)
        if len(self.backlog) > _MAX_BACKLOG_EVENTS:
            # Drop the oldest: a reattaching client cares most about now, and
            # the full history is in the stored conversation anyway.
            self.backlog.pop(0)
            self.truncated = True


class StreamRegistry:
    """Runs currently streaming in this process."""

    def __init__(self) -> None:
        self._runs: dict[str, _Run] = {}

    # -- producer side ---------------------------------------------------

    def start(self, run_id: str) -> None:
        """Begin (or restart) a run's stream. Any previous one is superseded."""
        self._sweep()
        self._runs[run_id] = _Run(run_id=run_id)

    def publish(self, run_id: str, event: dict) -> None:
        run = self._runs.get(run_id)
        if run is None:
            return
        run.add(event)
        for queue in list(run.listeners):
            # put_nowait, never await: a stalled listener must not be able to
            # slow down the run it is watching.
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:  # pragma: no cover - unbounded queues
                logger.warning("dropping event for a listener that fell behind")

    def finish(self, run_id: str) -> None:
        run = self._runs.get(run_id)
        if run is None:
            return
        run.done = True
        run.finished_at = time.time()
        for queue in list(run.listeners):
            queue.put_nowait({"type": "done"})

    # -- consumer side ---------------------------------------------------

    def is_active(self, run_id: str) -> bool:
        run = self._runs.get(run_id)
        return run is not None and not run.done

    async def attach(self, run_id: str):
        """Yield the backlog, then follow live until the run ends.

        Raises :class:`KeyError` when there is nothing to attach to, so the
        route can say "no active stream" instead of hanging on a run that
        finished an hour ago.
        """
        run = self._runs.get(run_id)
        if run is None:
            raise KeyError(run_id)

        queue: asyncio.Queue = asyncio.Queue()
        run.listeners.add(queue)
        try:
            if run.truncated:
                # Say so rather than presenting a partial history as complete.
                yield {
                    "type": "notice",
                    "content": "이어보기: 이전 진행 내용 일부는 생략되었습니다.",
                }
            for event in list(run.backlog):
                yield event
            if run.done:
                yield {"type": "done"}
                return

            while True:
                event = await queue.get()
                yield event
                if event.get("type") == "done":
                    return
        finally:
            run.listeners.discard(queue)

    # -- housekeeping ----------------------------------------------------

    def _sweep(self) -> None:
        cutoff = time.time() - _LINGER_SECONDS
        for run_id, run in list(self._runs.items()):
            if run.done and not run.listeners and (run.finished_at or 0) < cutoff:
                del self._runs[run_id]


#: Process-wide, for the same reason the concurrency cap is: it describes this
#: container's state, not one request's.
registry = StreamRegistry()


def as_sse(event: dict[str, Any]) -> str:
    """Render one event in the SSE framing the existing chat client parses."""
    import json

    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
