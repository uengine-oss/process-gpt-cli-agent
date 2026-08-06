// Render demo/transcript.json as a video.
//
// The transcript is the record of a real run (see demo/run_demo.py); this only
// decides how it is shown. Keeping the two apart means the video can be re-cut
// — pacing, wording, layout — without spending tokens on another run, and it
// keeps a presentation choice from ever being able to change what is claimed.
//
// Recording is Playwright's own `recordVideo` rather than a screen capture, so
// the output is deterministic and identical on any machine: no window manager,
// no cursor, no notification sliding in over the demo.
//
//   node demo/record.mjs [--out ../../docs/demo/cli-agent-demo.mp4]
//
import { chromium } from '../../frontend/node_modules/playwright/index.mjs';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const outArg = args.indexOf('--out');
const OUT = path.resolve(
    HERE,
    outArg >= 0 ? args[outArg + 1] : '../../../docs/demo/cli-agent-demo.mp4',
);

const transcript = JSON.parse(await fs.readFile(path.join(HERE, 'transcript.json'), 'utf8'));

// Pacing. Reading speed is the constraint, not rendering speed: a step that
// appears and vanishes before it can be read is worse than no step at all.
const DWELL = {
    narration: 2600,
    workitem: 3400,
    prompt: 4200,
    command: 1800,
    files: 2000,
    filebody: 4000,
    outcome: 3800,
    failure: 3600,
    json: 3000,
    note: 1500,
    event: 420,
    default: 1600,
};

const page_html = String.raw`
<!doctype html>
<meta charset="utf-8">
<style>
  @keyframes fadein { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; background: #0b1020; color: #e6edf6;
       font: 20px/1.55 "SF Mono", "JetBrains Mono", Menlo, monospace; }
  #app { height: 100vh; display: grid; grid-template-rows: 78px 1fr 132px; }
  header { display: flex; align-items: center; gap: 18px; padding: 0 40px;
       background: #121a33; border-bottom: 1px solid #222f52; }
  .brand { font-weight: 700; letter-spacing: .4px; color: #7fb2ff; font-size: 22px; }
  .scene { color: #cdd8ea; font-size: 24px; font-weight: 600; }
  .spacer { flex: 1; }
  .badge { font-size: 15px; color: #8ea3c4; border: 1px solid #2c3a5e; border-radius: 999px;
       padding: 5px 14px; }
  main { padding: 26px 40px; overflow: hidden; display: flex; flex-direction: column; gap: 12px; }
  .row { animation: fadein .22s ease-out both; }
  .cmd { color: #9ef0b4; }
  .cmd::before { content: "$ "; color: #4f7a5e; }
  .note { color: #8ea3c4; font-size: 18px; }
  .note::before { content: "· "; }
  .label { color: #7fb2ff; font-size: 17px; letter-spacing: .3px; text-transform: none; }
  pre { margin: 0; padding: 16px 20px; background: #0f1730; border: 1px solid #222f52;
       border-radius: 10px; white-space: pre-wrap; word-break: break-word;
       max-height: 420px; overflow: hidden; font-size: 18px; line-height: 1.5; }
  pre.small { font-size: 16px; max-height: 300px; }
  .files { display: flex; flex-wrap: wrap; gap: 10px; }
  .file { background: #17223f; border: 1px solid #2c3a5e; border-radius: 8px;
       padding: 7px 14px; font-size: 18px; color: #d7e3f5; }
  .file.new { border-color: #2f6f45; background: #12261b; color: #9ef0b4; }
  .file.gone { border-color: #6f2f36; background: #261215; color: #f09ea6;
       text-decoration: line-through; }
  .evt { display: flex; align-items: baseline; gap: 12px; font-size: 18px; }
  .evt .k { color: #f0c86e; min-width: 132px; }
  .evt .v { color: #cdd8ea; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .evt .v.stream { white-space: pre-wrap; text-overflow: clip; }
  .stream { color: #cdd8ea; font-size: 18px; white-space: pre-wrap; }
  .fail { border-color: #6f2f36; background: #1d1013; color: #f5b8bf; }
  .ok { border-color: #2f6f45; background: #10201a; }
  footer { padding: 20px 40px; background: #121a33; border-top: 1px solid #222f52; }
  .cap-title { font-size: 26px; font-weight: 700; color: #ffffff; }
  .cap-detail { font-size: 19px; color: #9fb3d1; margin-top: 6px; }
  kbd { background: #22304f; border-radius: 5px; padding: 2px 7px; font-size: 17px; }
</style>
<div id="app">
  <header>
    <span class="brand">ProcessGPT × CLI Agent</span>
    <span class="scene" id="scene"></span>
    <span class="spacer"></span>
    <span class="badge" id="badge"></span>
  </header>
  <main id="stream"></main>
  <footer>
    <div class="cap-title" id="capTitle"></div>
    <div class="cap-detail" id="capDetail"></div>
  </footer>
</div>
<script>
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? '').replace(/[&<>]/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

  window.setScene = (title, badge) => {
    $('scene').textContent = title;
    $('badge').textContent = badge || '';
    $('stream').innerHTML = '';
  };
  window.caption = (title, detail) => {
    $('capTitle').textContent = title || '';
    $('capDetail').textContent = detail || '';
  };
  window.push = (html) => {
    const el = document.createElement('div');
    el.className = 'row';
    el.innerHTML = html;
    const s = $('stream');
    s.appendChild(el);
    // Oldest rows fall off the top rather than the panel scrolling: a video
    // frame should read as a still, and a scrollbar mid-motion does not.
    while (s.scrollHeight > s.clientHeight && s.children.length > 1) s.removeChild(s.firstChild);
  };
  window.pushEvent = (kind, text) => {
    window.push('<div class="evt"><span class="k">' + esc(kind) + '</span><span class="v">' + esc(text) + '</span></div>');
  };
  // Streamed text arrives as deltas. A row per delta shows the transport, not
  // the answer — so consecutive chunks grow one block instead.
  window.streamText = (chunk) => {
    const s = $('stream');
    let last = s.lastElementChild;
    if (!last || !last.dataset.stream) {
      window.push('<div class="evt"><span class="k">assistant</span><span class="v stream"></span></div>');
      last = s.lastElementChild;
      last.dataset.stream = '1';
    }
    const target = last.querySelector('.stream');
    target.textContent = (target.textContent + chunk).slice(-620);
  };
  window.pushFiles = (label, files, mark) => {
    const chips = files.map((f) => '<span class="file ' + (mark?.[f] || '') + '">' + esc(f) + '</span>').join('');
    window.push('<div class="label">' + esc(label) + '</div><div class="files">' + chips + '</div>');
  };
  window.pushPre = (label, body, cls) => {
    window.push((label ? '<div class="label">' + esc(label) + '</div>' : '') +
      '<pre class="' + (cls || '') + '">' + esc(body) + '</pre>');
  };
</script>
`;

function truncate(text, lines, width = 118) {
    const out = String(text ?? '')
        .split('\n')
        .slice(0, lines)
        .map((l) => (l.length > width ? l.slice(0, width) + '…' : l));
    const total = String(text ?? '').split('\n').length;
    if (total > lines) out.push(`… (${total - lines}줄 더)`);
    return out.join('\n');
}

async function renderStep(page, step, state) {
    let dwell = DWELL[step.kind] ?? DWELL.default;
    if (step.kind === 'event' && step.event_kind === 'assistant_text') dwell = 190;

    switch (step.kind) {
        case 'narration':
            await page.evaluate(([t, d]) => window.caption(t, d), [step.title, step.detail || '']);
            break;

        case 'command': {
            // The prompt is one argv element and hundreds of characters long.
            // Printing it inline buries the flags that matter, and it already
            // gets a panel of its own two steps later.
            // Replace the prompt argument itself rather than slicing by
            // length, which cut mid-JSON and read as garbled output.
            const shown = step.text.replace(
                /(--print|exec)\s[\s\S]*?\s(--output-format|--json)/,
                '$1 "…업무 프롬프트…" $2',
            );
            await page.evaluate((t) => window.push('<div class="cmd">' + t.replace(/[&<>]/g, '') + '</div>'), shown);
            break;
        }

        case 'note':
            await page.evaluate((t) => window.push('<div class="note">' + t.replace(/[&<>]/g, '') + '</div>'), step.text);
            break;

        case 'workitem':
            await page.evaluate(
                ([label, body]) => window.pushPre(label, body),
                ['워크아이템', JSON.stringify(step.payload, null, 2)],
            );
            break;

        case 'prompt':
            await page.evaluate(
                ([label, body]) => window.pushPre(label, body, 'small'),
                [step.title, truncate(step.text, 14)],
            );
            break;

        case 'json':
            await page.evaluate(
                ([label, body]) => window.pushPre(label, body, 'small'),
                [step.title || '', truncate(JSON.stringify(step.payload, null, 2), 16)],
            );
            break;

        case 'files': {
            // Colour by what changed since the previous file listing, so undo
            // reads as an action rather than as two similar lists.
            const previous = state.lastFiles || [];
            const mark = {};
            for (const f of step.payload) if (!previous.includes(f)) mark[f] = 'new';
            const gone = previous.filter((f) => !step.payload.includes(f));
            await page.evaluate(
                ([label, files, m]) => window.pushFiles(label, files, m),
                [step.title, step.payload, mark],
            );
            if (gone.length) {
                await page.evaluate(
                    ([label, files, m]) => window.pushFiles(label, files, m),
                    ['제거됨', gone, Object.fromEntries(gone.map((f) => [f, 'gone']))],
                );
            }
            state.lastFiles = step.payload;
            break;
        }

        case 'filebody':
            await page.evaluate(
                ([label, body]) => window.pushPre(label, body, 'small'),
                [step.title, truncate(step.text, 13)],
            );
            break;

        case 'event': {
            if (step.event_kind === 'assistant_text') {
                await page.evaluate((t) => window.streamText(t), step.text);
            } else {
                const detail = [step.tool, step.path, step.change].filter(Boolean).join('  ');
                await page.evaluate(([k, v]) => window.pushEvent(k, v), [step.event_kind, detail]);
            }
            break;
        }

        case 'outcome': {
            const ok = step.payload.contract_met;
            await page.evaluate(
                ([label, body, cls]) => window.pushPre(label, body, cls),
                [
                    step.title + (ok ? '  ✓ 통과' : '  ✗ 불일치'),
                    JSON.stringify(step.payload, null, 2),
                    ok ? 'ok' : 'fail',
                ],
            );
            break;
        }

        case 'failure':
            await page.evaluate(
                ([label, body]) => window.pushPre(label, body, 'fail'),
                [step.title, step.text],
            );
            break;

        default:
            break;
    }

    await page.waitForTimeout(dwell);
}

const raw = path.join(HERE, '.recording');
await fs.rm(raw, { recursive: true, force: true });
await fs.mkdir(raw, { recursive: true });

const htmlPath = path.join(raw, 'stage.html');
await fs.writeFile(htmlPath, page_html, 'utf8');

const browser = await chromium.launch();
const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 },
    recordVideo: { dir: raw, size: { width: 1920, height: 1080 } },
    deviceScaleFactor: 1,
});
const page = await context.newPage();
await page.goto('file://' + htmlPath);

// Title card, so the first frames say what is being watched.
await page.evaluate(() => {
    window.setScene('CLI 코딩 에이전트에게 업무 위임하기', 'agent_orch = cliagents');
    window.caption(
        '이 영상의 모든 출력은 실제 실행 결과입니다',
        'Claude Code 를 실제로 실행해 얻은 기록(demo/transcript.json)을 그대로 재생합니다',
    );
});
await page.waitForTimeout(4200);

for (const scene of transcript.scenes) {
    const state = {};
    await page.evaluate(
        ([title, badge]) => window.setScene(title, badge),
        [scene.title, transcript.cli],
    );
    await page.waitForTimeout(900);
    for (const step of scene.steps) await renderStep(page, step, state);
    await page.waitForTimeout(1200);
}

await page.evaluate(() => {
    window.setScene('요약', '');
    window.caption(
        'CLI 별 분기는 라이브러리 안에만 있습니다',
        'services/cliagents = 어떤 CLI 인가 · services/cli-agent = ProcessGPT 업무 실행',
    );
    window.pushFiles('검증된 것', [
        '가용성 조회',
        '업무 위임 → 산출물 → 폼 계약',
        '대체 실행 금지',
        '되돌리기',
    ], {});
});
await page.waitForTimeout(4500);

const video = page.video();
await context.close();
await browser.close();

const webm = await video.path();
await fs.mkdir(path.dirname(OUT), { recursive: true });
// H.264 + faststart so it plays in a browser and in Slack rather than only in
// a desktop player that happens to know VP8.
execFileSync('ffmpeg', [
    '-y', '-i', webm,
    '-c:v', 'libx264', '-preset', 'medium', '-crf', '22',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    OUT,
], { stdio: 'inherit' });

const { size } = await fs.stat(OUT);
console.log(`\n${OUT}  (${(size / 1e6).toFixed(1)} MB)`);
await fs.rm(raw, { recursive: true, force: true });
