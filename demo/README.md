# Demo

A recording of the agent type doing its job, in two steps that stay separate on
purpose.

```bash
python -m demo.run_demo        # runs it for real → demo/transcript.json
node demo/record.mjs           # renders the transcript → docs/demo/cli-agent-demo.mp4
```

**`run_demo.py` runs the real thing.** A real work item shape, the real
selection and prompt code, a real `claude` process, the real outcome parsing.
Everything in `transcript.json` happened.

**`record.mjs` only decides how it is shown.** Splitting them means the video
can be re-cut — pacing, wording, layout — without spending tokens on another
run, and it means a presentation choice can never change what is being claimed.

Recording uses Playwright's own `recordVideo` rather than a screen capture, so
the output is deterministic on any machine: no window manager, no cursor, no
notification sliding in over the demo.

## Scenes

| | |
| --- | --- |
| 1. 어떤 CLI 를 쓸 수 있나 | `GET /agents` — installed, with install hints for what isn't |
| 2. 업무 위임 | a work item → prompt → real run → events → produced file → form contract |
| 3. 없는 CLI 는 대체하지 않는다 | an unavailable CLI fails with guidance; nothing else runs |
| 4. 되돌리기 | the journal puts the workspace back, without calling a model |

Scene 3 costs nothing to run — no model is ever reached — so it also works as a
smoke test: `python -m demo.run_demo --scene refusal`.

## Re-recording with Codex

```bash
python -m demo.run_demo --cli codex && node demo/record.mjs
```

Same transcript shape, same renderer, different agent — which is the parity
claim the library exists to make.
