// Drive the real Wikiracing page in headless Chrome and compare the DOM with server state.
// Usage: deno run --allow-net scripts/browser_check.ts http://127.0.0.1:8765
const APP = Deno.args[0] ?? "http://127.0.0.1:8765";
const CDP = "http://127.0.0.1:9333";

const targets = await (await fetch(`${CDP}/json/list`)).json();
const page = targets.find((t: { type: string }) => t.type === "page");
const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0;
const pending = new Map<number, (v: any) => void>();
await new Promise((r) => (ws.onopen = r as unknown as () => void));
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  if (message.id && pending.has(message.id)) {
    pending.get(message.id)!(message);
    pending.delete(message.id);
  }
};
const send = (method: string, params: Record<string, unknown> = {}) =>
  new Promise<any>((resolve) => {
    const callId = ++id;
    pending.set(callId, resolve);
    ws.send(JSON.stringify({ id: callId, method, params }));
  });
const evaluate = async (expression: string) => {
  const out = await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
  if (out.result?.exceptionDetails) throw new Error(JSON.stringify(out.result.exceptionDetails));
  return out.result.result.value;
};
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

await send("Page.enable");
await send("Runtime.enable");
await send("Page.navigate", { url: APP });
await sleep(2500);

const server = await (await fetch(`${APP}/api/races`)).json();
const bench = await evaluate(`({
  wins: document.getElementById('bench-wins').textContent,
  accuracy: document.getElementById('bench-accuracy').textContent,
  latency: document.getElementById('bench-latency').textContent,
  visible: !document.getElementById('benchmark').hidden,
  options: document.getElementById('race-select').options.length,
  note: document.getElementById('setup-note').textContent
})`);
const expectedWins = `${server.report.base.races.wins} → ${server.report.tuned.races.wins}`;
const expectedAccuracy = `${(server.report.base.decisions.top1_accuracy * 100).toFixed(1)}% → ${(server.report.tuned.decisions.top1_accuracy * 100).toFixed(1)}%`;
const expectedLatency = `${server.report.base.races.latency_ms_p50.toFixed(1)} → ${server.report.tuned.races.latency_ms_p50.toFixed(1)} ms`;
console.log("benchmark panel", bench);
if (!bench.visible || bench.wins !== expectedWins || bench.accuracy !== expectedAccuracy || bench.latency !== expectedLatency) {
  throw new Error(`benchmark panel mismatch: expected ${expectedWins} / ${expectedAccuracy} / ${expectedLatency}`);
}
if (bench.options !== server.races.length) throw new Error(`race count mismatch: ${bench.options} vs ${server.races.length}`);

for (const mode of ["benchmark", "explore"]) {
  await evaluate(`(() => { const input = document.querySelector('input[name="race-mode"][value="${mode}"]');
    input.checked = true; input.dispatchEvent(new Event('change', {bubbles: true}));
    document.getElementById('race-select').selectedIndex = 0;
    document.getElementById('start-button').click(); return true; })()`);
  let lanes: any;
  for (let i = 0; i < 120; i++) {
    await sleep(500);
    lanes = await evaluate(`({
      done: document.getElementById('start-button').disabled === false,
      note: document.getElementById('setup-note').textContent,
      base: {state: document.getElementById('base-state').textContent,
             hops: document.getElementById('base-hops').textContent,
             steps: document.querySelectorAll('#base-trail .step').length,
             end: document.querySelector('#base-trail .end-message')?.textContent ?? '',
             p50: document.getElementById('base-p50').textContent},
      tuned: {state: document.getElementById('tuned-state').textContent,
              hops: document.getElementById('tuned-hops').textContent,
              steps: document.querySelectorAll('#tuned-trail .step').length,
              end: document.querySelector('#tuned-trail .end-message')?.textContent ?? '',
              p50: document.getElementById('tuned-p50').textContent}
    })`);
    if (lanes.done) break;
  }
  if (!lanes.done) throw new Error(`${mode}: race did not finish`);
  const limit = mode === "benchmark" ? 8 : 16;
  console.log(mode, JSON.stringify(lanes, null, 1));
  for (const name of ["base", "tuned"]) {
    const lane = (lanes as any)[name];
    if (Number(lane.hops) > limit) throw new Error(`${mode}/${name}: ${lane.hops} hops exceeds ${limit}`);
    // The trail renders a START card plus one card per hop.
    if (lane.steps !== Number(lane.hops) + 1) throw new Error(`${mode}/${name}: ${lane.steps} steps but ${lane.hops} hops shown`);
  }
}
ws.close();
console.log("browser check OK");
