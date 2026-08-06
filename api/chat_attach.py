"""Rejoining a run that is still streaming.

The chat endpoint itself is mounted by the ProcessGPT SDK. This adds the other
half: a client whose connection dropped can come back to the same run instead of
staring at a spinner while the work finishes without it.

The answer to "is there anything to rejoin?" is deliberately explicit. A client
that is told *no* falls back to loading saved messages; one left hanging on a
stream that will never produce anything looks identical to a working stream and
never recovers.
"""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from core.stream_registry import as_sse, registry


async def status(request: Request) -> JSONResponse:
    run_id = request.path_params["run_id"]
    return JSONResponse({"run_id": run_id, "active": registry.is_active(run_id)})


async def attach(request: Request):
    """GET /runs/{run_id}/stream — backlog, then live, until the run ends."""
    run_id = request.path_params["run_id"]

    try:
        stream = registry.attach(run_id)
    except KeyError:
        return JSONResponse(
            {
                "error": "no active stream",
                "run_id": run_id,
                # Told plainly so the client knows to fall back rather than wait.
                "hint": "이 실행의 진행 스트림이 남아 있지 않습니다. 저장된 대화를 불러오세요.",
            },
            status_code=404,
        )

    async def _body():
        async for event in stream:
            yield as_sse(event)

    return StreamingResponse(
        _body(),
        media_type="text/event-stream",
        headers={
            # Proxies that buffer turn a live stream into one lump at the end.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


routes = [
    Route("/runs/{run_id}/stream", attach, methods=["GET"]),
    Route("/runs/{run_id}/stream/status", status, methods=["GET"]),
]
