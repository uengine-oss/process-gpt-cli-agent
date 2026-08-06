"""Replaying or undoing what a run changed.

Both operate on the journal, never on a model: a replay re-applies recorded
effects, and an undo walks them back. Anything the journal could not observe is
reported rather than silently skipped, because a caller who thinks a run was
fully undone will act on that belief.
"""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from core.journal import Journal
from core.workspace import locate


def _load(request: Request):
    workspace = locate(
        request.path_params["run_id"], tenant_id=request.query_params.get("tenant_id", "")
    )
    return workspace, Journal(workspace.path)


async def describe(request: Request) -> JSONResponse:
    workspace, journal = _load(request)
    if not workspace.exists:
        return JSONResponse({"error": "run not found or expired"}, status_code=404)
    return JSONResponse(
        {
            "run_id": workspace.run_id,
            "entries": len(journal.entries()),
            "fully_replayable": journal.replayable,
            "limitations": journal.limitations(),
        }
    )


async def replay(request: Request) -> JSONResponse:
    workspace, journal = _load(request)
    if not workspace.exists:
        return JSONResponse({"error": "run not found or expired"}, status_code=404)
    try:
        applied = journal.replay_files(workspace.path)
    except OSError as exc:
        # Say how far it got: a half-applied replay the caller knows about is
        # recoverable, one they don't is not.
        return JSONResponse({"error": str(exc), "partial": True}, status_code=500)
    return JSONResponse(
        {
            "applied": applied,
            "fully_replayable": journal.replayable,
            "limitations": journal.limitations(),
        }
    )


async def undo(request: Request) -> JSONResponse:
    workspace, journal = _load(request)
    if not workspace.exists:
        return JSONResponse({"error": "run not found or expired"}, status_code=404)
    restored, irreversible = journal.undo_files(workspace.path)
    return JSONResponse({"restored": restored, "irreversible": irreversible})


routes = [
    Route("/runs/{run_id}/journal", describe, methods=["GET"]),
    Route("/runs/{run_id}/replay", replay, methods=["POST"]),
    Route("/runs/{run_id}/undo", undo, methods=["POST"]),
]
