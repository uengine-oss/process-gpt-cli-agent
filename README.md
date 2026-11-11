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
after a run, plus chat. `CLIAGENTS_PROCESS_POLLING=false` drops the worker and
leaves only HTTP — that is how a per-conversation chat runner Pod runs (see
[Chat on runner Pods](#chat-on-runner-pods)).

## Shape

| Module | Responsibility |
| --- | --- |
| `server.py` | Process entry point: work-item polling loop + HTTP surface |
| `executor.py` | One work item / chat turn, start to finish |
| `core/stream_registry.py` | Runs in progress, for the legacy `/runs/{id}/stream` rejoin |
| `core/chat_session.py` | What a chat keeps beyond its Pod: session archive, output links |
| `core/settings.py` | Environment-derived configuration, read once |
| `core/selection.py` | Which CLI, which model, how much permission |
| `core/availability.py` | Is that CLI installed and authenticated here? |
| `core/workspace.py` | Per-run directory, path containment, retention |
| `core/prompt.py` | Work item → prompt |
| `core/skills.py` | Skills and instructions → CLI-native project artifacts |
| `core/bridge.py` | ProcessGPT tools → MCP registration, isolated per run |
| `core/credentials.py` | The tenant's own API key → that run's CLI environment, and nowhere else |
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
IMAGE_TAG=$(git rev-parse --short=7 HEAD)
IMAGE_NAME=ghcr.io/uengine-oss/process-gpt-cli-agent:${IMAGE_TAG}
docker build -t ${IMAGE_NAME} .
docker push ${IMAGE_NAME}
```

### Credentials

A run uses its **tenant's own API key** when the tenant's administrator has set
one (see [Tenant API keys](#tenant-api-keys)). Otherwise each CLI authenticates
with what the pod gives it, and this service never sees that token. Two
supported shapes for the pod, in order of preference:

1. **API keys** (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) — injectable, rotatable,
   and they need no interactive login inside a container.
2. **Subscription login** — mount the CLI's credential directory read-only.
   For Codex this is `CODEX_HOME`, which is *also* where it keeps MCP
   registrations; a run is given its own isolated copy for that reason, so the
   mounted path must not be reused as a run's config home.

Check what a deployment actually has with `GET /agents?check_auth=1` — that is
the same probe the picker uses, and it distinguishes "not installed" from "not
logged in" from "unknown".

A run relocates Claude Code's config directory into its workspace
(`CLAUDE_CONFIG_DIR`). On macOS the subscription login lives in the Keychain
under a name derived from that directory, so a local run does not see it —
use `CLAUDE_CODE_OAUTH_TOKEN` or an API key there too.

### Tenant API keys

**Why.** The pod's Claude credential is typically one person's subscription
token (`claude setup-token`). Anthropic's terms do not allow a subscription
credential to be shared by many people or used to serve other people's
requests; a company's own API key, used by that company's authorised users, is
allowed ([legal and compliance](https://code.claude.com/docs/en/legal-and-compliance)).
A deployment that runs many companies' work on one subscription is exactly the
first case. So each company's administrator stores the company's own Anthropic
and OpenAI keys in the admin screen, and runs for that company use them.

**What is read.** Before each run — work item or chat turn — the service calls
`cli_agent_credential_get(p_tenant, p_provider)` with its service-role key
(`SUPABASE_KEY`, the same client the rest of the service uses):
`anthropic` for Claude Code, `openai` for Codex. The key itself lives in
Supabase Vault; `cli_agent_credentials` holds only who set it, when, and the
last four characters. The function is executable by `service_role` only.

**Precedence, per run:**

| Tenant key | Work item | Chat turn |
| --- | --- | --- |
| set | runs on the key (below) | runs on the key (below) |
| not set — or no function in the database yet | the pod's environment, unchanged, plus a timeline line | **refused**: SSE `error`, `code: "company_key_required"` — nothing starts, nothing is archived |
| lookup failed (transient) | the pod's environment, unchanged, plus a timeline line | **refused**: SSE `error`, `code: "company_key_check_failed"` («잠시 뒤 다시») |
| not set, with `CLIAGENTS_CHAT_ALLOW_POD_CREDENTIALS=true` | as above | the pod's environment, unchanged — the old fallback |

Running on the key means:

| CLI | Started with |
| --- | --- |
| Claude Code | `ANTHROPIC_API_KEY=<key>`; the pod's `CLAUDE_CODE_OAUTH_TOKEN` and `ANTHROPIC_AUTH_TOKEN` are **removed** from that run's environment |
| Codex | `CODEX_API_KEY` and `OPENAI_API_KEY` = `<key>` |

Why chat and work items differ: the screen offers CLI chat because the company
set a key, so a chat that quietly ran on the pod's credential — usually one
person's subscription — would be the sharing above, unnoticed. A work item
keeps falling back, because that is how every run worked before tenant keys
and a stopped queue is worse. The chat refusal tells the user that an
administrator can add the key under 설정 › CLI 에이전트 API 키; a failed lookup
asks them to send again instead of falling back "just this once". Set
`CLIAGENTS_CHAT_ALLOW_POD_CREDENTIALS=true` only where the pod's credential is
itself the company's own.

The pod's token is removed rather than left beside the key: with both present,
which one is billed depends on the CLI's precedence rules (`ANTHROPIC_AUTH_TOKEN`
outranks the key outright), and billing a company's work to the shared
subscription is what this exists to stop. Nothing changes in `os.environ`: the
key goes into the environment built for that one CLI process, so a concurrent
run for another tenant cannot inherit it. The library's `exec_env` can only
add to `os.environ`, never remove from it, so the run's provider is wrapped
with an `exec_env` that drops those two variables after the merge.

Without a key, the work item's timeline gets one line — 「회사 Anthropic API 키가
설정되지 않아 이 서버의 기본 자격 증명으로 실행합니다」, or 「…확인하지 못해…」 when the
lookup failed. A database without the function yet (production before the SQL
is applied) counts as "not set" and is logged once per process — which also
means chat is refused there until the SQL is applied and a key is set.

**Caching.** Answers, including "no key", are reused for
`CLIAGENTS_CREDENTIAL_CACHE_SECONDS` (default 60) per tenant and provider, so a
key saved or removed in the admin screen applies within a minute. A chat turn
does not trust a cached "no key" — it asks again before refusing, so a key
just added works on the next message. A failed lookup is not cached; the next
run asks again.

**Never written down.** The key is not logged, not put in an event, the
timeline, the journal or the session archive. The agent could still print it
(an `env` in a shell): every CLI event is cleaned of it before anything reads
the event, and after the CLI exits the run's workspace — the CLI's session
transcript included — is scrubbed before it is archived, published or
downloaded. `tests/test_credentials.py` runs a fake CLI that does exactly that
and checks logs, chat events, timeline rows, workspace files and the archive.

### Platform

| Variable | Needed for |
| --- | --- |
| `SUPABASE_URL` | everything below; also the JWKS address for tenant auth |
| `SUPABASE_KEY` | the service-role key: polling, chat message storage, the room-tenant check on attach/stop/steer, the `users` lookup behind tenant auth, the session archive |
| `SUPABASE_JWT_SECRET` | optional — verifies HS256 user tokens locally. Without it, asymmetric tokens are checked against JWKS and HS256 ones by asking GoTrue (`/auth/v1/user`) |
| `MEMENTO_BASE_URL` | download links for chat outputs (Memento's private `artifacts` bucket) |
| `CLIAGENTS_PROCESS_POLLING` | `false` in runner Pods (default `true`) |
| `CLIAGENTS_CREDENTIAL_CACHE_SECONDS` | how long a tenant's API key (or its absence) is reused, default `60` |
| `CLIAGENTS_CHAT_ALLOW_POD_CREDENTIALS` | `true` lets a chat for a tenant without its own key run on the pod's credentials; default `false` refuses it |

### Gateway

The browser reaches this service at `/process-gpt-cli-agent/`. A ready-made
nginx block is in [`deploy/nginx.conf`](deploy/nginx.conf); these are the paths
it covers:

| Path | Used by |
| --- | --- |
| `GET /agents` | the CLI picker in the work-item panel (and a chat picker) |
| `GET/POST/DELETE /skills…` | tenant skill management |
| `GET /runs/{id}/files`, `/download` | a work item's produced files, on this Deployment's volume |
| `POST /chat/stream` | chat with a CLI agent (SSE) |
| `POST /chat/stream/attach` | rejoining a chat turn whose connection dropped |
| `POST /chat/stop` | stopping a chat turn on the server, not only in the browser |
| `POST /chat/steer` | redirecting a turn — answers `501 unsupported` (no `steer()` here) |
| `GET /runs/{id}/stream` | legacy rejoin, by run id; no current caller — chat uses `/chat/stream/attach` |
| `GET /health` | probes, and the session router's `busy` check |

The four `/chat/*` routes are the SDK's standard set, the same paths the chat
screen calls for deepagents and codex. The streaming routes need
`proxy_buffering off` and a read timeout longer than a run, or progress arrives
in one lump at the end — or not at all.

### Chat on runner Pods

In production, chat is served by **one runner Pod per conversation**, created
by `process-gpt-session-router` from a PodTemplate. The router sends
`/chat/stream`, `/chat/stream/attach`, `/chat/stop` and `/chat/steer` to the
conversation's Pod and everything else to the shared Deployment, and reclaims
a Pod idle for 30 minutes — unless its `GET /health` says `"busy": true`.

What that requires of this service, and how it is met:

| Requirement | How |
| --- | --- |
| A runner never claims a work item | `CLIAGENTS_PROCESS_POLLING=false`. A reclaimed Pod would take the item down with it. |
| The router can tell a running turn from an idle Pod | `/health` → `busy`: a chat turn in flight (SDK registry) or any run of this process, including the session upload after a turn |
| The caller owns the tenant it names | tenant auth on (`tenant_auth=True`): the stream is guarded by the caller's Supabase JWT; attach/stop/steer also check the room's tenant |
| The conversation continues on a fresh Pod | SDK session archive: the workspace and each CLI's session transcript are restored before a turn and saved after it |
| Produced files stay downloadable | SDK artifact contract: files the turn put in `outputs/` are stored privately through Memento and linked on the final event |

**Choosing the CLI.** A chat request names its CLI in `metadata` — the SDK
passes `metadata` through and drops unknown top-level body fields:

| Field (aliases) | Values |
| --- | --- |
| `metadata.agent_cli` (`agentCli`, `cli_agent`, `cliAgent`) | `claude-code`, `codex` — must be installed here (`GET /agents`) |
| `metadata.agent_model` (`agentModel`, `model`) | optional; the CLI's own default otherwise |
| `metadata.agent_permission` (`agentPermission`, `permission`) | `read_only`, `workspace_write` (default), `command_exec`, or the UI aliases |

The same keys inside `metadata.agent_config` work too. **A chat that names no
CLI is refused** with an SSE `error` whose `code` is `cli_not_selected` and
whose `available` lists the installed CLIs; `CLIAGENTS_DEFAULT_CLI` applies to
work items only. A named CLI that is not installed answers `cli_unavailable`.
A tenant without its own key for that CLI answers `company_key_required` (or
`company_key_check_failed` when the key could not be read) — see
[Tenant API keys](#tenant-api-keys). None of these runs anything in its place.

**Outputs.** Only files in the turn's `outputs/` directory are deliverables
(the agent is told so). They are collected after the turn, stored in Memento's
private artifact bucket, and returned as signed links on `done.files` — also
stored on the chat message (`pdfFiles`), so they survive a reload and the Pod.
`/runs/{id}/download` is not used for chat: the router sends it to the shared
Deployment, which never had the runner's disk. Without `MEMENTO_BASE_URL` the
turn still completes and says in its answer which files got no link.

**Session archive.** `agent_sessions` table and the private `agent-sessions`
bucket (`process-gpt-agent-sdk/database_schema.sql`), shared by every agent
type. Credentials and per-run configuration in a CLI's home (`auth.json`,
`.credentials.json`, Claude Code's `.claude.json`, `config.toml`, `agents/`,
`skills/`) are never archived; resuming needs only the session transcripts.
Without the table or bucket a turn still runs; the next turn on a fresh Pod
starts without the earlier files and CLI session.

### Storage

On the shared Deployment, `/workspace` must be a persistent volume. A work item
paused for a human resumes into its own directory, and downloads read from it
after the process has exited. `CLIAGENTS_WORKSPACE_RETENTION_HOURS` decides how
long both stay possible; the service sweeps hourly.

A runner Pod's `/workspace` can be an `emptyDir`: for chat the session archive
is the original and the disk a cache.

### Rolling back

Nothing else depends on this service, so a rollback is:

1. Hide the `cliagents` option in the orchestration picker (frontend).
2. Scale the service to zero.

Work items already marked `cliagents` are then **not polled by anyone** and stay
in their queue — they do not fail and they do not run. An operator can re-point
them at another orchestration by changing `agent_orch` on the work item. Other
orchestrations are untouched: they poll their own value and never saw this one.

## Scaling

**One worker.** The SDK's chat registries (reattach, stop) and
`core/stream_registry.py` keep runs-in-progress in process memory, so a second
uvicorn worker would answer "no active stream" for a run the first one is still
serving. That degrades softly — the client falls back to stored messages — but
the live catch-up disappears silently.

Chat scales by Pods instead: the session router sends every request of one
conversation to that conversation's runner, which is exactly the condition
under which process-local registries are correct.

The concurrency cap (`CLIAGENTS_MAX_CONCURRENT_RUNS`) is per process for the
same reason: it describes what this container can carry.

## Tests

```bash
pytest                                  # unit + HTTP, no infrastructure needed
python -m e2e.run_suite --free-only     # the E2E checks that spend no credits
```

`e2e/` holds the scenarios that need a real database and a real CLI; see
[e2e/README.md](e2e/README.md).
