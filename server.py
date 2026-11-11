"""Process entry point: the polling worker plus the HTTP surface.

Two jobs in one process, as with every other ProcessGPT agent type. The worker
claims work items whose orchestration is ``cliagents``; the HTTP side answers
the questions the browser has to ask before and after a run — which CLIs exist
here, what skills are installed, what files did the run produce — and serves
chat.

Chat is the SDK's standard set of routes: ``/chat/stream``, ``/chat/stream/
attach``, ``/chat/stop`` and ``/chat/steer``. In production they are served by
one runner Pod per conversation, which the session router creates and reclaims;
that Pod runs with ``CLIAGENTS_PROCESS_POLLING=false`` and tells the router
through ``/health``'s ``busy`` whether reclaiming it now would cut a run short.

Single worker on purpose: the chat reattach registry and the concurrency cap
are process-local, and a second uvicorn worker would answer "no such run" for
a stream the other one is still serving.
"""

from __future__ import annotations

import asyncio
import logging

import uvicorn
from dotenv import load_dotenv
from processgpt_agent_sdk import ProcessGPTAgentServer, get_inflight_registry
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse

from api.availability import routes as availability_routes
from api.chat_attach import routes as chat_attach_routes
from api.files import routes as file_routes
from api.replay import routes as replay_routes
from api.skills import routes as skill_routes
from core import skills as skill_store
from core import workspace
from core.settings import AGENT_TYPE, settings
from executor import CliAgentExecutor, active_runs, limiter

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#: How often finished workspaces are swept. Retention is measured in hours, so
#: checking hourly is precise enough and costs nothing.
_SWEEP_INTERVAL_SECONDS = 3600


async def _health(_request: Request) -> JSONResponse:
    """Liveness, plus whether reclaiming this process now would cut a run short.

    ``busy`` is what the session router asks before reclaiming an idle runner
    Pod. A turn can run far longer than the idle TTL, so wall-clock idleness
    alone cannot answer it. Two sources, either one enough: the SDK's chat
    turns (from request to the last event) and this service's own runs (chat
    turns and work items, including the session upload after a turn).
    """
    turns = get_inflight_registry().active_conversation_ids()
    runs = active_runs()
    return JSONResponse(
        {
            "status": "ok",
            "agent_type": AGENT_TYPE,
            "busy": bool(turns) or runs > 0,
            "active_turns": len(turns),
            "active_runs": runs,
            "runs_in_flight": limiter.in_flight,
            "max_concurrent_runs": limiter.limit,
            "process_polling": settings.process_polling,
        }
    )


async def _sweep_workspaces() -> None:
    while True:
        try:
            removed = workspace.sweep()
            if removed:
                logger.info("swept %d expired workspaces", len(removed))
        except Exception:  # noqa: BLE001 - a sweep failure must not stop the service
            logger.exception("workspace sweep failed")
        await asyncio.sleep(_SWEEP_INTERVAL_SECONDS)


def build_app(server: ProcessGPTAgentServer | None = None) -> Starlette:
    """The HTTP surface. With ``server``, the SDK chat routes are mounted too."""
    app = Starlette(
        routes=[
            *availability_routes,
            *chat_attach_routes,
            *skill_routes,
            *file_routes,
            *replay_routes,
        ]
    )
    app.add_route("/health", _health, methods=["GET"])
    if server is not None:
        # The SDK owns chat end to end: request parsing, SSE framing and
        # heartbeat, reattach, stop, steer — and, with tenant_auth on, the
        # guard on the stream path. The paths are the ones the chat screen
        # already calls for deepagents and codex. This executor has no
        # `steer()`, so /chat/steer answers 501 `unsupported`, as the SDK
        # specifies, rather than pretending to redirect a CLI mid-run.
        #
        # /health stays this service's own (health_path=None): its `busy`
        # also counts work items and the post-turn upload.
        server.mount_chat_routes(app, stream_paths=("/chat/stream",), health_path=None)
    return app


async def main() -> None:
    try:
        seeded = await asyncio.to_thread(skill_store.seed_system_skills)
        logger.info("seeded %d bundled system skills", seeded)
    except Exception:  # noqa: BLE001 - bundled skills are a nicety, not a gate
        logger.exception("could not seed bundled system skills; continuing without them")

    server = ProcessGPTAgentServer(
        agent_executor=CliAgentExecutor(),
        agent_type=AGENT_TYPE,
        # The request's tenant_id becomes the run's workspace path, its skill
        # directory and its MCP servers. Unverified, changing that one value
        # would reach another tenant's files. Checked against the caller's
        # Supabase JWT, as deepagents does.
        tenant_auth=True,
    )
    app = build_app(server)

    http = uvicorn.Server(
        uvicorn.Config(app, host=settings.host, port=settings.port, log_level="info")
    )
    logger.info(
        "cliagents agent type up: agent_orch=%s http=http://%s:%d todolist polling=%s",
        AGENT_TYPE,
        settings.host,
        settings.port,
        "on" if settings.process_polling else "off",
    )

    tasks = [
        asyncio.create_task(http.serve(), name="http"),
        asyncio.create_task(_sweep_workspaces(), name="sweeper"),
    ]
    if settings.process_polling:
        # Off in a per-conversation runner Pod: a Pod that claimed a work item
        # would take it down when the router reclaims the Pod for being idle.
        tasks.append(asyncio.create_task(server.run(), name="poller"))

    # Whichever exits first (a signal reaches the poller, or the HTTP server
    # when there is no poller) takes the rest down.
    _done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    logger.info("cliagents agent type stopped")


if __name__ == "__main__":
    asyncio.run(main())
