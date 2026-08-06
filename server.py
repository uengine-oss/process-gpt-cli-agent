"""Process entry point: the polling worker plus the HTTP surface.

Two jobs in one process, as with every other ProcessGPT agent type. The worker
claims work items whose orchestration is ``cliagents``; the HTTP side answers
the questions the browser has to ask before and after a run — which CLIs exist
here, what skills are installed, what files did the run produce.

Single worker on purpose: the chat reattach registry and the concurrency cap
are process-local, and a second uvicorn worker would answer "no such run" for
a stream the other one is still serving.
"""

from __future__ import annotations

import asyncio
import logging

import uvicorn
from dotenv import load_dotenv
from processgpt_agent_sdk import ProcessGPTAgentServer
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
from executor import CliAgentExecutor, limiter

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#: How often finished workspaces are swept. Retention is measured in hours, so
#: checking hourly is precise enough and costs nothing.
_SWEEP_INTERVAL_SECONDS = 3600


async def _health(_request: Request) -> JSONResponse:
    """Liveness plus the two things an operator actually asks about."""
    return JSONResponse(
        {
            "status": "ok",
            "agent_type": AGENT_TYPE,
            "runs_in_flight": limiter.in_flight,
            "max_concurrent_runs": limiter.limit,
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


def build_app() -> Starlette:
    return Starlette(
        routes=[
            *availability_routes,
            *chat_attach_routes,
            *skill_routes,
            *file_routes,
            *replay_routes,
        ]
    )


async def main() -> None:
    try:
        seeded = await asyncio.to_thread(skill_store.seed_system_skills)
        logger.info("seeded %d bundled system skills", seeded)
    except Exception:  # noqa: BLE001 - bundled skills are a nicety, not a gate
        logger.exception("could not seed bundled system skills; continuing without them")

    server = ProcessGPTAgentServer(agent_executor=CliAgentExecutor(), agent_type=AGENT_TYPE)

    app = build_app()
    app.add_route("/health", _health)
    server.mount_chat_sse(app, path="/chat/stream")

    http = uvicorn.Server(
        uvicorn.Config(app, host=settings.host, port=settings.port, log_level="info")
    )
    logger.info(
        "cliagents agent type up: agent_orch=%s http=http://%s:%d",
        AGENT_TYPE,
        settings.host,
        settings.port,
    )

    tasks = [
        asyncio.create_task(server.run(), name="poller"),
        asyncio.create_task(http.serve(), name="http"),
        asyncio.create_task(_sweep_workspaces(), name="sweeper"),
    ]

    # Whichever exits first (a signal reaches the poller) takes the rest down.
    _done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    logger.info("cliagents agent type stopped")


if __name__ == "__main__":
    asyncio.run(main())
