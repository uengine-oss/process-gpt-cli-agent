"""Bridging the library's synchronous stream into the service's event loop.

:func:`cliagents.stream_exec` is a blocking generator by design — the library
refuses to pick an async framework for its hosts. This service *is* an async
host, so the adaptation happens here, once, instead of at every call site.

What this adds beyond a thread: a wall-clock limit, and cancellation that
actually kills the child process. A CLI agent that keeps running after its work
item was cancelled is not a cosmetic problem — it keeps writing files and
spending money on a job nobody is waiting for.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import threading
from collections.abc import AsyncIterator

from cliagents import ExecEvent, ExecRequest, stream_exec

logger = logging.getLogger(__name__)

#: Sentinels on the queue. A tuple so they cannot collide with an event.
_DONE = ("done",)


class RunTimeoutError(Exception):
    """The run exceeded its wall-clock budget and was stopped."""


async def astream(
    provider,
    request: ExecRequest,
    *,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
    popen=subprocess.Popen,
) -> AsyncIterator[ExecEvent]:
    """Yield events from a headless run, cancellable and time-boxed.

    On cancellation or timeout the child is terminated and the library's own
    teardown reaps it, so there is exactly one place that knows how to close a
    run down and exactly one that knows how to stop one.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    stop = threading.Event()
    failure: list[BaseException] = []
    #: Set by the library as soon as the child exists. The only way to unblock
    #: a pump thread that is waiting on a silent agent's stdout.
    child: list = [None]

    def _remember(process) -> None:
        child[0] = process

    def _pump() -> None:
        generator = stream_exec(provider, request, env=env, popen=popen, on_start=_remember)
        try:
            for event in generator:
                loop.call_soon_threadsafe(queue.put_nowait, event)
                if stop.is_set():
                    break
        except BaseException as exc:  # noqa: BLE001 - re-raised on the loop side
            failure.append(exc)
        finally:
            # Closing runs the library's teardown, which reaps the child.
            generator.close()
            loop.call_soon_threadsafe(queue.put_nowait, _DONE)

    worker = threading.Thread(target=_pump, name="cliagents-exec", daemon=True)
    worker.start()

    deadline = (loop.time() + timeout) if timeout else None
    try:
        while True:
            remaining = (deadline - loop.time()) if deadline else None
            if remaining is not None and remaining <= 0:
                raise RunTimeoutError(f"run exceeded {timeout:.0f}s")
            try:
                item = await asyncio.wait_for(queue.get(), timeout=remaining)
            except asyncio.TimeoutError as exc:
                raise RunTimeoutError(f"run exceeded {timeout:.0f}s") from exc

            if item is _DONE:
                break
            yield item

        if failure:
            raise failure[0]
    finally:
        stop.set()
        # `stop` is only noticed between events, and the case that matters —
        # timeout, cancellation — is a run producing no events at all. So the
        # child is terminated directly: that closes the pipe, the pump's read
        # returns, and the library's teardown reaps it.
        process = child[0]
        if process is not None and process.poll() is None:
            process.terminate()
        await asyncio.to_thread(worker.join, 10)
        if worker.is_alive() and process is not None:
            process.kill()
            await asyncio.to_thread(worker.join, 5)
        if worker.is_alive():
            logger.warning("exec pump did not stop; leaving it detached")


class ConcurrencyLimit:
    """How many CLI processes may run at once.

    A semaphore that can also be *asked* whether it would block, because the
    polling loop needs to decide not to claim a work item at all. Claiming one
    and then waiting would park the item in "running" behind a queue nobody can
    see.
    """

    def __init__(self, limit: int) -> None:
        self._limit = max(1, limit)
        self._semaphore = asyncio.Semaphore(self._limit)
        self._in_flight = 0
        self._lock = asyncio.Lock()

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def limit(self) -> int:
        return self._limit

    def at_capacity(self) -> bool:
        return self._in_flight >= self._limit

    async def __aenter__(self) -> "ConcurrencyLimit":
        await self._semaphore.acquire()
        async with self._lock:
            self._in_flight += 1
        return self

    async def __aexit__(self, *_exc) -> None:
        async with self._lock:
            self._in_flight -= 1
        self._semaphore.release()
