"""Which CLI agents this deployment can actually run.

The picker in the browser calls this before offering a choice. Without it the
user picks Codex, waits for the queue, and finds out at execution time that the
binary was never installed in the image.
"""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from core.availability import snapshot


async def list_agents(request: Request) -> JSONResponse:
    """GET /agents — installed state per CLI, auth state on request.

    Auth probing spawns a subprocess per agent, so it is opt-in: a picker that
    only needs to grey out uninstalled agents should not pay for it.
    """
    check_auth = request.query_params.get("check_auth") in ("1", "true", "yes")
    agents = snapshot(refresh=True, check_auth=check_auth)
    return JSONResponse({"agents": [a.as_dict() for a in agents]})


routes = [Route("/agents", list_agents, methods=["GET"])]
