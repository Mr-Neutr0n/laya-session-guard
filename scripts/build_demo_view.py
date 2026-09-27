"""Build a self-contained static HTML view of router demo predictions.

The JSON is inlined into the file, so the result opens by double click with no
server, no network and no build step. No CDN either: the page is one file.
"""
import argparse
import html
import json
from pathlib import Path

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Laya arXiv router demo</title>
<style>
  :root { color-scheme: light dark; --ink:#16181d; --muted:#6b7280; --line:#d8dbe0;
           --bg:#fbfbfc; --card:#ffffff; --good:#0f7b46; --warn:#a4600a; --bad:#a32020; }
  @media (prefers-color-scheme: dark) { :root { --ink:#e8eaed; --muted:#9aa0a6; --line:#333;
           --bg:#14161a; --card:#1c1f24; --good:#5dd39e; --warn:#e0a458; --bad:#ff8080; } }
  * { box-sizing: border-box; }
  body { margin:0; padding:2rem 1.25rem 4rem; background:var(--bg); color:var(--ink);
          font:15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
  main { max-width: 60rem; margin: 0 auto; }
  h1 { font-size:1.5rem; margin:0 0 .35rem; letter-spacing:-.01em; }
  h2 { font-size:1.05rem; margin:0 0 .2rem; }
  p.sub { color:var(--muted); margin:.15rem 0 1.5rem; }
  code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:.86em; }
  .meta { border:1px solid var(--line); background:var(--card); border-radius:10px;
           padding:.85rem 1rem; margin-bottom:1.5rem; font-size:.86rem; }
  .meta div { display:flex; gap:.5rem; padding:.12rem 0; }
  .meta span:first-child { color:var(--muted); min-width:9.5rem; }
  .card { border:1px solid var(--line); background:var(--card); border-radius:10px;
           padding:1rem 1.1rem; margin-bottom:1rem; }
  .card.real { border-left:3px solid var(--good); }
  .card.edge { border-left:3px solid var(--warn); }
  .top { display:flex; align-items:baseline; gap:.6rem; flex-wrap:wrap; }
  .top .label { font-weight:650; font-size:1.05rem; }
  .top .prob { color:var(--muted); font-size:.85rem; }
  .badge { font-size:.72rem; text-transform:uppercase; letter-spacing:.06em;
            border:1px solid var(--line); border-radius:999px; padding:.1rem .5rem; color:var(--muted); }
  .bars { margin-top:.7rem; }
  .bar { display:grid; grid-template-columns:11rem 1fr 3.2rem; align-items:center;
          gap:.6rem; margin:.18rem 0; font-size:.82rem; }
  .bar .name { color:var(--muted); }
  .bar .track { display:block; background:var(--line); border-radius:4px; height:.62rem; overflow:hidden; }
  .bar .fill { display:block; height:100%; background:var(--muted); border-radius:4px; }
  .bar.top1 .fill { background:var(--ink); }
  .bar .val { text-align:right; font-variant-numeric:tabular-nums; color:var(--muted); }
  .abstract { color:var(--muted); font-size:.85rem; margin:.6rem 0 0; }
  .foot { color:var(--muted); font-size:.78rem; margin-top:.4rem;
           display:flex; gap:1rem; flex-wrap:wrap; }
  .warn { color:var(--bad); }
  .ok { color:var(--good); }
  footer { color:var(--muted); font-size:.8rem; margin-top:2.5rem; border-top:1px solid var(--line);
            padding-top:1rem; }
</style>
</head>
<body>
<main>
  <h1>Laya arXiv topic router, live on the published checkpoint</h1>
  <p class="sub">Eight subject areas, title plus the first 100 abstract words, CPU only.
     Distribution is softmax over the eight option logits, uncalibrated.</p>
  <div class="meta">__META__</div>
  <div id="cards"></div>
  <footer>__FOOTER__</footer>
</main>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const data = JSON.parse(document.getElementById('payload').textContent);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const pct = v => (v * 100).toFixed(1) + '%';
const known = new Set(data.labels);

document.getElementById('cards').innerHTML = data.cases.map(c => {
  const ranked = Object.entries(c.distribution).sort((a, b) => b[1] - a[1]);
  const gold = c.gold_primary_category || null;
  const goldKnown = gold && known.has(gold);
  const verdict = gold
    ? (goldKnown
        ? `<span class="${c.top_label === gold ? 'ok' : 'warn'}">gold {esc(gold)} is in the option set; top choice ${c.top_label === gold ? 'matches' : 'differs'}</span>`
        : `<span class="warn">gold {esc(gold)} is not one of the eight options, so no answer here can be right</span>`)
    : '<span class="mono">contrived case, no gold label</span>';
  const bars = ranked.map(([label, value], i) => `
    <div class="bar ${i === 0 ? 'top1' : ''}">
      <span class="name mono">${esc(label)}</span>
      <span class="track"><span class="fill" style="width:${(value * 100).toFixed(1)}%"></span></span>
      <span class="val">${pct(value)}</span>
    </div>`).join('');
  return `<section class="card ${c.kind}">
    <div class="top">
      <span class="badge">${esc(c.kind)}</span>
      <h2>${esc(c.title)}</h2>
    </div>
    <div class="top" style="margin-top:.35rem">
      <span class="label mono">${esc(c.top_label)}</span>
      <span class="prob">${pct(c.top_probability)} &middot; margin over runner-up ${pct(c.margin_over_runner_up)}</span>
    </div>
    <div class="bars">${bars}</div>
    <p class="abstract">${esc(c.abstract)}</p>
    <div class="foot">
      <span>${verdict}</span>
      <span>${c.tokens} tokens</span>
      <span>${Math.round(c.latency_ms_per_example)} ms/example</span>
      <span class="mono">${esc(c.id)}</span>
    </div>
  </section>`;
}).join('');
</script>
</body>
</html>
"""


def build(predictions_path, cases_path, output):
    predictions = json.loads(Path(predictions_path).read_text())
    cases = {json.loads(line)["id"]: json.loads(line)
             for line in Path(cases_path).read_text().splitlines() if line.strip()}
    merged = []
    for item in predictions["predictions"]:
        case = cases.get(item["id"], {})
        merged.append({**item, "kind": case.get("kind", "unknown"),
                       "abstract": case.get("abstract", ""),
                       "gold_primary_category": case.get("gold_primary_category"),
                       "note": case.get("note"),
                       "first_version_date": case.get("first_version_date")})
    meta = "".join(
        f"<div><span>{html.escape(label)}</span><span class='mono'>{html.escape(value)}</span></div>"
        for label, value in (
            ("model", f"{predictions['model']}@{predictions['revision'][:12]}"),
            ("device", predictions["device"]),
            ("options", ", ".join(predictions["labels"])),
            ("abstract words kept", str(predictions.get("max_abstract_words", 100))),
            ("context budget", f"{predictions.get('max_len')} tokens, head {predictions.get('head_max_len')}"),
        ))
    footer = ("Real papers come from arXiv metadata submitted in 2025, far outside the 2007-2009 "
              "training window, and most of their true categories are not among the eight options. "
              "That is the point: the router has no abstain option, so a confident wrong answer is "
              "its default failure. Confidence here is uncalibrated and is not an "
              "out-of-distribution signal.")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Inside a <script> block the payload is raw text, so HTML-escaping it would
    # break JSON.parse. Only "</" needs escaping, and "<\/" is a valid JSON escape.
    payload = json.dumps({"labels": predictions["labels"], "cases": merged}).replace("</", "<\\/")
    page = (TEMPLATE.replace("__META__", meta).replace("__FOOTER__", footer)
            .replace("__PAYLOAD__", payload))
    target.write_text(page)
    return target, len(merged)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default="demo/predictions.json")
    parser.add_argument("--cases", default="demo/verification_cases.jsonl")
    parser.add_argument("--output", default="demo/index.html")
    args = parser.parse_args()
    path, count = build(args.predictions, args.cases, args.output)
    print(f"Wrote {path} with {count} cases, {path.stat().st_size // 1024} kB, no external requests")
