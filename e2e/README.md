# E2E: delegating a work item to a CLI agent

Verifies the whole path a work item takes — claimed from the queue, turned into
a prompt, run on a real CLI, streamed, and stored — against real infrastructure
rather than doubles.

Follows the repository convention: **Dockerised infrastructure, source-run
application services.** Postgres/Supabase runs in a container; this service runs
from the working tree, so what is verified is the code you just edited.

## What it needs

| | |
| --- | --- |
| Postgres with the ProcessGPT schema | `docker compose -f docker-compose.e2e.yml up -d` |
| At least one CLI installed and authenticated | `claude auth status` / `codex login status` |
| Real model credits | these tests spend tokens — they are not free |

## Running

```bash
docker compose -f e2e/docker-compose.e2e.yml up -d
export $(grep -v '^#' .env.e2e | xargs)
python -m e2e.run_suite            # all scenarios
python -m e2e.run_suite --only hitl
```

Each scenario seeds a work item, starts the service against it, and asserts on
what the database and the workspace hold afterwards.

## Scenarios

| id | Covers | Asserts |
| --- | --- | --- |
| `basic` | work item → run → stored result | the item completes, the result is stored, produced files exist in the workspace |
| `both-clis` | the same work item on Claude Code and on Codex | both produce the same event kinds, the same file, and a stored result — the parity claim |
| `hitl` | a run that needs permission | the item goes to input-required, a pending request is recorded, and answering resumes the *same* session rather than restarting |
| `missing-cli` | a CLI that is not installed | the item fails with install guidance, and **no other agent ran** |
| `draft-vs-complete` | agent_mode | draft parks for review, complete closes the item |

`missing-cli` is the one worth running even without credits: it needs no model
call and it guards the rule that matters most operationally — that a missing CLI
never silently becomes a different one.

## Status

The scenario definitions and the runner are here; they have **not** been
executed against live infrastructure yet — this checkout has no ProcessGPT
database to point them at. Treat a green run as the actual acceptance gate.
