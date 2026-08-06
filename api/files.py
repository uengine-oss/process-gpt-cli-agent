"""Downloading what a run produced.

Every path is resolved inside the run's own workspace before anything is read.
The run id comes from the client, so "which run" is exactly as trustworthy as
the caller — and the containment check is what stops one tenant's request from
reading another's draft.
"""

from __future__ import annotations

import mimetypes

from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from core.workspace import locate


async def list_files(request: Request) -> JSONResponse:
    workspace = locate(
        request.path_params["run_id"], tenant_id=request.query_params.get("tenant_id", "")
    )
    if not workspace.exists:
        return JSONResponse({"error": "run not found or expired"}, status_code=404)

    files = [
        {"path": workspace.relative(p), "bytes": p.stat().st_size}
        for p in workspace.files()
        # Bookkeeping the run wrote for us, not work product the user asked for.
        if not workspace.relative(p).startswith(".processgpt")
    ]
    return JSONResponse({"run_id": workspace.run_id, "files": files})


async def download(request: Request):
    workspace = locate(
        request.path_params["run_id"], tenant_id=request.query_params.get("tenant_id", "")
    )
    relative = request.query_params.get("path", "")
    if not relative:
        return JSONResponse({"error": "path is required"}, status_code=400)

    try:
        target = workspace.resolve_within(relative)
    except PermissionError:
        # Deliberately the same answer as "not there": a caller probing for
        # other runs' files learns nothing from the difference.
        return JSONResponse({"error": "not found"}, status_code=404)

    if not target.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)

    media_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    return FileResponse(target, media_type=media_type, filename=target.name)


routes = [
    Route("/runs/{run_id}/files", list_files, methods=["GET"]),
    Route("/runs/{run_id}/download", download, methods=["GET"]),
]
