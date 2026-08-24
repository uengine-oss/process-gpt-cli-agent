# process-gpt-cli-agent

The ProcessGPT orchestration (`agent_orch = cliagents`) that delegates a work
item to a **CLI coding agent** — Claude Code or Codex — instead of to an
in-process LLM graph.

Which CLI runs, on which model, with how much permission is per work item. All
of the per-agent knowledge (how to invoke it headlessly, how to read its event
stream, where it looks for skills, where it keeps MCP registrations) lives in
[uengine-oss/cliagents](https://github.com/uengine-oss/cliagents), so adding a
third CLI is a provider there, not a change here.

```bash
pip install -r requirements.txt
python server.py
```

Runs as one process: a worker claiming work items whose orchestration is
`cliagents`, and an HTTP surface for the questions the browser asks before and
after a run.

## Shape

| Module | Responsibility |
| --- | --- |
| `server.py` | Process entry point: work-item polling loop + HTTP surface |
| `executor.py` | One work item / chat turn, start to finish |
| `core/stream_registry.py` | Runs in progress, so a dropped browser can rejoin |
| `core/settings.py` | Environment-derived configuration, read once |
| `core/selection.py` | Which CLI, which model, how much permission |
| `core/availability.py` | Is that CLI installed and authenticated here? |
| `core/workspace.py` | Per-run directory, path containment, retention |
| `core/prompt.py` | Work item → prompt |
| `core/skills.py` | Skills and instructions → CLI-native project artifacts |
| `core/bridge.py` | ProcessGPT tools → MCP registration, isolated per run |
| `core/events.py` | Normalised exec events → ProcessGPT UI events |
| `core/outcome.py` | Final text → form outputs, honestly or not at all |
| `core/hitl.py` | Permission requests → human questions → resume |
| `core/journal.py` | What changed, so it can be replayed or undone |

## Why the agent type is one value, not two

`agent_orch` is `cliagents`; which CLI runs is a separate field. The polling
server is keyed on `agent_orch`, so a per-CLI orchestration value would mean a
second deployment of the same service to poll a second queue, and a third when
someone adds Gemini. The library already treats "which agent" as data.

## Deploying

Build from the service directory:

```bash
IMAGE_TAG=$(git rev-parse HEAD)
IMAGE_NAME=ghcr.io/uengine-oss/process-gpt-cli-agent:${IMAGE_TAG}
docker build -t ${IMAGE_NAME} .
docker push ${IMAGE_NAME}
```

### Credentials

Each CLI authenticates itself; this service never sees a token. Two supported
shapes, in order of preference:

1. **API keys** (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) — injectable, rotatable,
   and they need no interactive login inside a container.
2. **Subscription login** — mount the CLI's credential directory read-only.
   For Codex this is `CODEX_HOME`, which is *also* where it keeps MCP
   registrations; a run is given its own isolated copy for that reason, so the
   mounted path must not be reused as a run's config home.

Check what a deployment actually has with `GET /agents?check_auth=1` — that is
the same probe the picker uses, and it distinguishes "not installed" from "not
logged in" from "unknown".

### Gateway

The browser reaches this service at `/process-gpt-cli-agent/`. A ready-made
nginx block is in [`deploy/nginx.conf`](deploy/nginx.conf); these are the paths
it covers:

| Path | Used by |
| --- | --- |
| `GET /agents` | the CLI picker in the work-item panel |
| `GET/POST/DELETE /skills…` | tenant skill management |
| `GET /runs/{id}/files`, `/download` | produced-file browsing and download |
| `POST /chat/stream` | chat with a CLI agent (SSE) |
| `GET /runs/{id}/stream` | rejoining a run whose connection dropped |

Both streaming routes need `proxy_buffering off` and a read timeout longer than
a run, or progress arrives in one lump at the end — or not at all.

### Storage

`/workspace` must be a persistent volume. A run paused for a human resumes into
its own directory, and downloads read from it after the process has exited.
`CLIAGENTS_WORKSPACE_RETENTION_HOURS` decides how long both stay possible; the
service sweeps hourly.

### Rolling back

Nothing else depends on this service, so a rollback is:

1. Hide the `cliagents` option in the orchestration picker (frontend).
2. Scale the service to zero.

Work items already marked `cliagents` are then **not polled by anyone** and stay
in their queue — they do not fail and they do not run. An operator can re-point
them at another orchestration by changing `agent_orch` on the work item. Other
orchestrations are untouched: they poll their own value and never saw this one.

## Scaling

**One worker.** `core/stream_registry.py` keeps runs-in-progress in process
memory, so a second uvicorn worker would answer "no active stream" for a run the
first one is still serving. That degrades softly — the client falls back to
stored messages — but the live catch-up disappears silently. Move the registry
to shared storage before scaling out.

The concurrency cap (`CLIAGENTS_MAX_CONCURRENT_RUNS`) is per process for the
same reason: it describes what this container can carry.

## Tests

```bash
pytest                                  # unit + HTTP, no infrastructure needed
python -m e2e.run_suite --free-only     # the E2E checks that spend no credits
```

`e2e/` holds the scenarios that need a real database and a real CLI; see
[e2e/README.md](e2e/README.md).
