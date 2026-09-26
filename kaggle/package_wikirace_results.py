"""Package the public Wikiracing results notebook from the measured artifacts.

The notebook recomputes every table from files published with the Hugging Face checkpoint.
It needs no GPU, no API key, and never retrains anything.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "kaggle/wikiracing-results"

REPO = "hxrikp/laya-wikirace-wikispeedia"
REVISION = "81d6d2143fd81859f62520ba09dd61ba9b7bc319"
FILES = ["report.json", "baseline.json", "decision_predictions.json", "race_traces.json",
         "test.jsonl", "eval_races.json", "graph.json", "data_provenance.json", "WIKIRACE_RESULTS.md",
         "eval_races_240.json", "eval_240_races.json"]


def cell(kind, source, identifier):
    return {"cell_type": kind, "id": identifier, "metadata": {}, "source": source.splitlines(keepends=True),
            **({"outputs": [], "execution_count": None} if kind == "code" else {})}


header = f"""# Laya Wikiracing: base versus fine-tuned on Wikispeedia

I fine-tuned Laya to pick the next Wikipedia link on its way to a target article. On 600 held-out link choices it went from **37.0% to 47.2%**. On the 60 held-out races measured on Kaggle it went from **13 wins to 23**, and on a 4x larger held-out race set it went from **52 wins to 90**.

This notebook recomputes those numbers from the per-decision logits and race traces published with the checkpoint. **Run All needs no GPU and no API key.** A cell near the end can rerun the model itself, and it is off by default.

Extended:
- [Hugging Face: fine-tuned checkpoint and evaluation artifacts](https://huggingface.co/{REPO})
- [Private Kaggle training notebook, owner access required](https://www.kaggle.com/code/uranium53/laya-wikiracing-base-vs-fine-tuned)
- [Training code, data builder, evaluation, and app: the `codex/laya-wikirace` branch, not yet pushed](https://github.com/Mr-Neutr0n/laya-session-guard)
- [Data source: Stanford SNAP Wikispeedia paths-and-graph](https://snap.stanford.edu/data/wikispeedia.html)

**Experimental only.** The graph is a historical 4,604-article Wikipedia subset, and a perfect agent using the same shortlist still loses 5 of the 60 Kaggle routes and 18 of the 240. A 37.5% win rate is a weak navigator.
"""

SETUP = f"""import hashlib, json, os, random, re, subprocess, sys, urllib.request
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

REPO, REVISION = '{REPO}', '{REVISION}'
ROOT = Path(os.environ.get('WIKIRACE_EVAL_DIR', Path('/kaggle/working/wikirace-eval')))
ROOT.mkdir(parents=True, exist_ok=True)
FILES = {FILES!r}
for name in FILES:
    path = ROOT / name
    if not path.exists():
        url = f'https://huggingface.co/{{REPO}}/resolve/{{REVISION}}/evaluation/{{name}}'
        with urllib.request.urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
print('Pinned artifacts:', REPO, '@', REVISION)
print('Downloaded', len(FILES), 'small files. No inference, no training, no paid calls yet.')

read = lambda name: json.loads((ROOT / name).read_text())
lines = lambda name: [json.loads(line) for line in (ROOT / name).read_text().splitlines()]
report = read('report.json')
decisions = read('decision_predictions.json')
traces = read('race_traces.json')
test_rows = {{row['id']: row for row in lines('test.jsonl')}}
races = read('eval_races.json')
graph = read('graph.json')
provenance = read('data_provenance.json')

assert report['status'] == 'completed'
assert len(decisions['base']) == len(decisions['tuned']) == 600
assert len(traces['base']) == len(traces['tuned']) == 60
assert [r['id'] for r in decisions['base']] == [r['id'] for r in decisions['tuned']]
print('Report base revision:', report['base_revision'], '| GPU:', report['gpu'], '| seed:', report['seed'])
print('Split sizes:', report['split_sizes'])
print('Data provenance:', provenance['articles'], 'articles,', provenance['links'], 'links,',
      provenance['valid_human_paths'], 'verified human paths')
"""

HELPERS = """def choice(row):
    return int(np.argmax(row['logits']))

def correct(row):
    return choice(row) == row['label']

def soft_prediction(row, temperature):
    z = np.asarray(row['logits'], dtype=np.float64) / temperature
    z -= z.max()
    p = np.exp(z)
    return p / p.sum()

def brier_and_ece(rows, temperature):
    # Menus vary from 2 to 12 links, so each decision is scored on its own.
    total_brier = 0.0
    buckets = {}
    for row in rows:
        distribution = soft_prediction(row, temperature)
        truth = np.zeros(len(distribution)); truth[row['label']] = 1.0
        total_brier += float(((distribution - truth) ** 2).sum())
        confidence, prediction = float(distribution.max()), int(distribution.argmax())
        low = min(int(confidence * 10), 9)
        buckets.setdefault(low, []).append((confidence, prediction == row['label']))
    ece = sum(len(items) / len(rows) *
              abs(np.mean([c for c, _ in items]) - np.mean([g for _, g in items]))
              for items in buckets.values())
    return total_brier / len(rows), ece

def paired_interval(differences, seed=42, draws=5000):
    rng = random.Random(seed)
    n = len(differences)
    samples = [sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws)]
    return tuple(float(np.percentile(samples, p)) for p in (2.5, 97.5))
"""

TRAINING = """history = report['history']
training = pd.DataFrame([
    {'Epoch': row['epoch'], 'Mean training loss': row['mean_loss']} for row in history
]).set_index('Epoch')
display(Markdown('### Training, one Tesla T4'))
display(training.style.format({'Mean training loss': '{:.3f}'}))
print(f"GPU: {report['gpu']} | recorded training and final evaluation: {report['training_seconds']:.1f} s")
print('Base Laya revision:', report['base_revision'])
print('Fitted choice temperatures, base then tuned:',
      report['base']['decisions']['temperature'], report['tuned']['decisions']['temperature'])
print('Optimizer steps this run: 1125, epochs 3, batch 4, accumulation 4, seed 42')
"""

DECISIONS = """rows = []
for name, key in [('Base Laya', 'base'), ('Fine-tuned Laya', 'tuned')]:
    raw = decisions[key]
    temperature = report[key]['decisions']['temperature']
    brier, ece = brier_and_ece(raw, temperature)
    rows.append({
        'Model': name,
        'Top-1 accuracy': np.mean([correct(row) for row in raw]),
        'Brier': brier,
        'ECE (10 bins)': ece,
    })
decision_table = pd.DataFrame(rows).set_index('Model')
display(Markdown('### Held-out choices: 600 decisions, same questions for both checkpoints'))
display(decision_table.style.format({'Top-1 accuracy': '{:.1%}', 'Brier': '{:.3f}', 'ECE (10 bins)': '{:.3f}'}))

# Recompute independently of the training script and fail loudly on any drift.
for name, key in [('Base Laya', 'base'), ('Fine-tuned Laya', 'tuned')]:
    recorded = report[key]['decisions']
    mine = decision_table.loc[name]
    assert abs(mine['Top-1 accuracy'] - recorded['top1_accuracy']) < 1e-9
    assert abs(mine['Brier'] - recorded['brier']) < 1e-9
    assert abs(mine['ECE (10 bins)'] - recorded['ece_10_bin']) < 1e-9
print('Recomputed accuracy, Brier, and ECE match the training report for both checkpoints.')

indirect = [i for i, row in enumerate(decisions['base'])
            if test_rows[row['id']]['target'] not in test_rows[row['id']]['candidates']]
indirect_table = pd.DataFrame([
    {'Model': name,
     'Accuracy without a direct target link': np.mean([correct(decisions[key][i]) for i in indirect])}
    for name, key in [('Base Laya', 'base'), ('Fine-tuned Laya', 'tuned')]]).set_index('Model')
print(f'\\n{len(indirect)} of 600 held-out decisions do not offer the target as a choice:')
display(indirect_table.style.format({'Accuracy without a direct target link': '{:.1%}'}))

delta = [int(correct(t)) - int(correct(b)) for b, t in zip(decisions['base'], decisions['tuned'])]
lo, hi = paired_interval(delta)
print(f'Paired accuracy difference (tuned minus base): {np.mean(delta):+.1%}; '
      f'95% paired bootstrap interval [{lo:.1%}, {hi:.1%}]')

fig, ax = plt.subplots(figsize=(7.5, 4.2))
bars = ax.bar([row['Model'] for row in rows], [row['Top-1 accuracy'] for row in rows],
              color=['#7B8794', '#237A9B'])
ax.set_ylim(0, 0.6)
ax.set_ylabel('Top-1 choice accuracy')
ax.set_yticks([0, .2, .4, .6], ['0%', '20%', '40%', '60%'])
ax.bar_label(bars, labels=[f"{row['Top-1 accuracy']:.1%}" for row in rows], padding=3)
ax.spines[['top', 'right']].set_visible(False)
ax.set_title('600 held-out link choices, identical menus and identical label rule')
fig.tight_layout()
plt.show()
"""

CEILING = """def reachable_within(start, target, hops):
    \"\"\"Best case for this shortlist: hop only through links the model is actually offered.\"\"\"
    seen, queue = {start}, deque([(start, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth >= hops:
            continue
        for candidate in shortlist(graph, node, target):
            if candidate == target:
                return True
            if candidate not in seen:
                seen.add(candidate)
                queue.append((candidate, depth + 1))
    return False

def shortlist(g, current, target, limit=12, exclude=()):
    \"\"\"Identical to wikirace/graph.py: title overlap and log indegree, never distance to target.\"\"\"
    titles, links = g['titles'], g['links']
    available = [v for v in links[current] if v not in set(exclude)]
    if target in available:
        selected = [target] + [v for v in available if v != target][:limit - 1]
        return sorted(selected, key=lambda v: hashlib.sha256(f'{current}:{target}:{v}'.encode()).digest())
    target_terms = terms(titles[target])
    indegree = g['indegree']

    def score(v):
        t = terms(titles[v])
        overlap = len(t & target_terms) / max(1, len(t | target_terms))
        return (4 * overlap + 0.10 * np.log1p(indegree[v]), -v)

    selected = sorted(available, key=score, reverse=True)[:limit]
    return sorted(selected, key=lambda v: hashlib.sha256(f'{current}:{target}:{v}'.encode()).digest())

def terms(title):
    return set(re.findall(r'[a-z0-9]+', title.lower())) - {'the', 'of', 'and', 'in', 'a'}

ceiling = [reachable_within(r['start'], r['target'], 8) for r in races]
print(f"Shortlist permits a route within 8 moves for {sum(ceiling)} of {len(ceiling)} races.")
reachable_within_list = lambda rs: [reachable_within(r['start'], r['target'], 8) for r in rs]
print('That is the maximum possible win count for a perfect agent with this shortlist.')
print('Raw graph distance is not the binding limit; the offered menu is.')
"""

RACES = """race_table = pd.DataFrame([
    {'Model': name,
     'Wins within 8 moves': sum(1 for row in traces[key] if row['success']),
     'Win rate': np.mean([row['success'] for row in traces[key]]),
     'Mean hops on wins': np.mean([row['hops'] for row in traces[key] if row['success']]),
     'Mean excess hops on wins': np.mean([row['hops'] - row['shortest_hops']
                                          for row in traces[key] if row['success']])}
    for name, key in [('Base Laya', 'base'), ('Fine-tuned Laya', 'tuned')]]).set_index('Model')
display(Markdown('### Held-out races: 60 routes, 8-move limit, same routes for both checkpoints'))
display(race_table.style.format({'Win rate': '{:.1%}', 'Mean hops on wins': '{:.2f}',
                                'Mean excess hops on wins': '{:.2f}'}))

won_delta = [int(t['success']) - int(b['success']) for b, t in zip(traces['base'], traces['tuned'])]
lo, hi = paired_interval(won_delta)
new_wins = sum(not b['success'] and t['success'] for b, t in zip(traces['base'], traces['tuned']))
lost_wins = sum(b['success'] and not t['success'] for b, t in zip(traces['base'], traces['tuned']))
print(f'Paired win difference (tuned minus base): {np.mean(won_delta):+.1%}; '
      f'95% paired bootstrap interval [{lo:.1%}, {hi:.1%}]')
print(f'Tuned won {new_wins} races that base lost and lost {lost_wins} that base won.')
print('Wins achievable by a perfect agent with this shortlist:', sum(ceiling), 'of', len(ceiling))
for name, key in [('Base Laya', 'base'), ('Fine-tuned Laya', 'tuned')]:
    assert int(race_table.loc[name, 'Wins within 8 moves']) == report[key]['races']['wins']
print('Recomputed win counts match the training report for both checkpoints.')

fig, ax = plt.subplots(figsize=(7.5, 4.2))
labels = ['Base Laya', 'Fine-tuned Laya', 'Perfect agent,\\nsame shortlist']
values = [race_table.loc['Base Laya', 'Win rate'],
          race_table.loc['Fine-tuned Laya', 'Win rate'], float(np.mean(ceiling))]
bars = ax.bar(labels, values, color=['#7B8794', '#237A9B', '#C06C32'])
ax.set_ylim(0, 1.05)
ax.set_ylabel('Races won within 8 moves')
ax.set_yticks([0, .25, .5, .75, 1], ['0%', '25%', '50%', '75%', '100%'])
ax.bar_label(bars, labels=[f'{round(v*60)}/60\\n{v:.1%}' for v in values], padding=3)
ax.spines[['top', 'right']].set_visible(False)
ax.set_title('60 held-out races against the achievable ceiling')
fig.tight_layout()
plt.show()
"""

FAILURE = """# Look at a race the fine-tune lost and base won, rather than only the aggregate.
lost = [i for i, (b, t) in enumerate(zip(traces['base'], traces['tuned']))
        if b['success'] and not t['success']]
print(f'{len(lost)} races won by base and lost by the fine-tune:', lost)
for i in lost:
    route = races[i]
    print(f"\\nRoute: {graph['titles'][route['start']]} -> {graph['titles'][route['target']]} "
          f"(shortest {route['shortest_hops']} hops)")
    for name, key in [('Base Laya', 'base'), ('Fine-tuned Laya', 'tuned')]:
        row = traces[key][i]
        print(f"  {name:15s} success={row['success']} hops={row['hops']}")
        print('    ' + ' -> '.join(graph['titles'][v] for v in row['path']))

worst = [i for i, (b, t) in enumerate(zip(traces['base'], traces['tuned']))
         if b['success'] and t['success'] and t['hops'] > b['hops']]
print(f"\\nBoth won but the fine-tune took longer on {len(worst)} route"
      f"{'' if len(worst) == 1 else 's'}.")
for i in worst[:3]:
    route = races[i]
    print(f"  {graph['titles'][route['start']]} -> {graph['titles'][route['target']]}: "
          f"base {traces['base'][i]['hops']} hops, tuned {traces['tuned'][i]['hops']} hops, "
          f"shortest {route['shortest_hops']}")
"""

RERUN = """RERUN_LAYA = False

if RERUN_LAYA:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                    'laya==0.3.5', 'transformers==4.57.6', 'safetensors==0.6.2'], check=True)
    os.environ.update(USE_TF='0', USE_FLAX='0')
    from huggingface_hub import snapshot_download
    checkpoints = {
        'base': ('convaiinnovations/laya', report['base_revision']),
        'tuned': (REPO, REVISION),
    }
    for label, (repo, revision) in checkpoints.items():
        path = snapshot_download(repo, revision=revision,
                                 allow_patterns=['model.safetensors', 'rl_agent_config.json',
                                                 'encoder/config.json', 'tokenizer/*'])
        print(label, 'downloaded to', path)
    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    print('Rerun device:', device, '| a full rerun needs about 25 minutes on one T4')
else:
    print('Model rerun disabled. The published logits, traces, and metrics were used above.')
"""

EXTENDED_MD = """## 6. A 4x larger held-out race set

Sixty routes is a small sample, and those sixty were the routes the release was built around. Because the route ordering is a deterministic hash of the seed and the article pair, taking the first 240 entries gives a strict superset: the original 60 are unchanged, and 180 routes are genuinely new.

Both published checkpoints were replayed over all 240 on CPU through the project's own app server, which is a different serving path and different hardware from the Kaggle measurement. The faithfulness check is in the data: on the shared 60 routes the replay reproduces the Kaggle traces on all 60 outcomes and all 60 hop counts for both checkpoints. The latency numbers below are CPU numbers and are not comparable to the T4 figures earlier in this notebook.
"""

EXTENDED = """extended_races = read('eval_races_240.json')
extended = read('eval_240_races.json')
assert len(extended_races) == 240
assert extended_races[:60] == races, 'the published 60 must be a prefix of the larger set'
big_reachable = reachable_within_list(extended_races)
assert sum(big_reachable) == extended['ceiling']

def wins(paths):
    return [path[-1] == route['target'] for path, route in zip(paths, extended_races)]

big_base, big_tuned = wins(extended['paths']['base']), wins(extended['paths']['tuned'])
print(f\"Shortlist permits a route within 8 moves for {sum(big_reachable)} of 240 routes.\")
big_table = pd.DataFrame([
    {'Model': 'Base Laya', 'Wins within 8 moves': sum(big_base), 'Win rate': np.mean(big_base),
     'Share of ceiling': sum(big_base) / sum(big_reachable)},
    {'Model': 'Fine-tuned Laya', 'Wins within 8 moves': sum(big_tuned), 'Win rate': np.mean(big_tuned),
     'Share of ceiling': sum(big_tuned) / sum(big_reachable)},
]).set_index('Model')
display(Markdown('### 240 held-out routes, replayed locally on CPU'))
display(big_table.style.format({'Win rate': '{:.1%}', 'Share of ceiling': '{:.1%}'}))

big_delta = [int(t) - int(b) for b, t in zip(big_base, big_tuned)]
blo, bhi = paired_interval(big_delta)
print(f'Paired win difference over 240 routes: {np.mean(big_delta):+.1%}; '
      f'95% paired bootstrap interval [{blo:.1%}, {bhi:.1%}]')
new_only = big_delta[60:]
nlo, nhi = paired_interval(new_only)
print(f'On the 180 routes that are new here: {np.mean(new_only):+.1%}; '
      f'95% paired bootstrap interval [{nlo:.1%}, {nhi:.1%}]')
print('Tuned won', sum(1 for b, t in zip(big_base, big_tuned) if not b and t),
      'races base lost and lost', sum(1 for b, t in zip(big_base, big_tuned) if b and not t),
      'that base won.')
print('Original 60 inside the larger set: base', sum(big_base[:60]), 'tuned', sum(big_tuned[:60]))
print('New 180: base', sum(big_base[60:]), 'tuned', sum(big_tuned[60:]))

fig, ax = plt.subplots(figsize=(7.5, 4.2))
labels = ['Base Laya', 'Fine-tuned Laya', 'Perfect agent,\\nsame shortlist']
values = [np.mean(big_base), np.mean(big_tuned), float(np.mean(big_reachable))]
bars = ax.bar(labels, values, color=['#7B8794', '#237A9B', '#C06C32'])
ax.set_ylim(0, 1.05)
ax.set_ylabel('Races won within 8 moves')
ax.set_yticks([0, .25, .5, .75, 1], ['0%', '25%', '50%', '75%', '100%'])
ax.bar_label(bars, labels=[f'{round(v*240)}/240\\n{v:.1%}' for v in values], padding=3)
ax.spines[['top', 'right']].set_visible(False)
ax.set_title('240 held-out routes, 180 of them never used before this replay')
fig.tight_layout()
plt.show()
"""

REVISION_BASE = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"

LIMITS_EXTRA = """- **The larger race set is a local CPU replay, not a Kaggle run.** It shares 60 routes with the Kaggle measurement and reproduces all of them, but the added 180 routes have no second-host replication.
- **More routes are not more seeds.** The 240 routes share one training run, one checkpoint, and one shortlist, so they sharpen the route estimate without adding any evidence about training variance.
"""

LIMITS = f"""## 7. What this establishes, and what it does not

The fine-tune is a real, reproducible improvement over the base checkpoint on the same held-out questions and the same routes, with paired intervals that exclude zero. It holds when the race set is quadrupled and restricted to routes that are new in that larger set. It is still a weak navigator: 90 wins against a ceiling of 222 on the 240 routes, on a historical subset of Wikipedia, with labels defined by graph search rather than human preference.

Important limits:

- **Historical graph.** 4,604 articles from the archived Wikispeedia dataset, not live Wikipedia. Live edits do not affect a race.
- **Achievable ceiling.** 5 of the 60 Kaggle routes and 18 of the 240 have no route within eight moves through the offered shortlist, so a perfect 240/240 is not a valid target.
- **Shortlist-bound task.** Candidates are ranked by title overlap with the target and log indegree, never by distance. A path-aware planner would raise the ceiling and change every number here.
- **Graph-search labels.** The correct choice is the candidate that reduces reverse distance fastest, with a deterministic hash tie-break. This is a proxy for racing, not a human notion of the best link.
- **One seed, one run.** No seed variance and no repeated evaluation. The bootstrap intervals resample these same 600 decisions and 240 routes only.
{LIMITS_EXTRA}
- **Calibration did not travel far.** ECE 0.046 with a temperature fitted on the same fixture family as training.
- **Latency is hardware-specific.** The 30.8 ms p50 is warm T4 inference including batch preparation and device transfer, excluding tokenization, checkpoint load, graph lookup, and rendering. On CPU the same checkpoint takes roughly 100 to 300 ms per move.
- **No live-page, multilingual, or human evaluation was performed.**
- **No private data.** Everything is derived from a public research archive. No user sessions or transcripts are involved.

### Sources and artifacts

- [Fine-tuned checkpoint, evaluation artifacts, and this model card](https://huggingface.co/{REPO})
- [Training notebook, owner access required](https://www.kaggle.com/code/uranium53/laya-wikiracing-base-vs-fine-tuned)
- **Code and app**: the `codex/laya-wikirace` branch of [Mr-Neutr0n/laya-session-guard](https://github.com/Mr-Neutr0n/laya-session-guard). It is not pushed yet, and `main` does not contain these files. Everything needed to reproduce the numbers here is in the `evaluation/` folder of the Hugging Face repository.
- [Wikispeedia paths-and-graph, Stanford SNAP](https://snap.stanford.edu/data/wikispeedia.html)
- Base model: [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya), revision `{REVISION_BASE}`. The native act head was frozen and is not used for link choice.
"""

TRAIN_MD = """## 2. What was trained

Laya's English 421M-parameter checkpoint was fully fine-tuned with supervised cross-entropy for three epochs on one Kaggle Tesla T4, starting from the pinned base revision below. The encoder and the typed option scorer were updated; the native act head was frozen and never used for link choice.

The data is 6,000 training decisions, 600 calibration decisions, and 600 held-out decisions over the archived Wikispeedia graph. Human finished paths supply current-page and target pairs, and reverse graph search labels the candidate that reaches the target in the fewest remaining hops. Targets are disjoint across the three splits by construction, so no test target was trained or calibrated on.

The shortlist is deterministic and shuffled by a hash of the article pair, so there is no position cue. It ranks only legal outgoing links, by title overlap with the target and by log indegree, and it never consults distance to the target. That is what makes the race a test of the model rather than of the search.
"""

DECISION_MD = """## 3. Held-out link choices

Both checkpoints answered the same 600 questions with the same candidate menus. Accuracy is the fraction of questions where the highest-scoring link is the labelled one. Brier score and expected calibration error use each checkpoint's own temperature, fitted only on the 600 calibration decisions.

The second table removes the 111 easy questions that offer the target article directly. That is the more honest number: without a one-click shortcut, both checkpoints are far below chance-plus-a-little, and the gap is the part of the task that actually requires planning.
"""

RACE_MD = """## 4. Held-out races

The 60 evaluation routes were played out hop by hop, with the model choosing every link and the loop stopping at 8 moves. Before showing the result, it is worth asking what a perfect agent could do at all, because the shortlist decides that.

A breadth-first search that may only hop through links the model is actually offered reaches the target within 8 moves for fewer than all 60 races. The ceiling is the honest denominator, so the chart puts the fine-tune and the base checkpoint next to it.
"""

FAILURE_MD = """## 5. A race the fine-tune lost

Aggregates hide the interesting cases. This section prints the routes where the base checkpoint won and the fine-tune lost, and the routes where both won but the fine-tune wandered longer. The point is not that the fine-tune is uniformly better. It is that the improvement is uneven, and the published traces let a reader check that claim instead of taking it on trust.
"""

RERUN_MD = """## 8. Rerun the model, optionally

Everything above is recomputed from published logits and traces. That keeps Run All free and fast, and it keeps the notebook from quietly replacing measured results with new ones. If you want fresh inferences, enable a GPU in Kaggle and set `RERUN_LAYA = True`; CPU works but is much slower. The cell only downloads the two checkpoints. Reruns are your own and are not written back here.
"""

REVISION_BASE = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"

CELLS = [
    cell("markdown", header, "intro"),
    cell("markdown", "## 1. The question\n\nWikispeedia gives you a current article and a target article. You may only click a link that the current article actually contains. The model sees the current article, the target, and a shortlist of at most 12 offered outgoing links, then picks one. Races are scored by whether the target is reached within a fixed number of moves.\n\nThis is a navigation task with a hard, checkable answer, which is why it is worth measuring: the label rule is a graph search, not an agent's opinion, and the same 600 questions and 60 routes are asked of every checkpoint.", "question"),
    cell("code", SETUP, "setup"),
    cell("code", HELPERS, "helpers"),
    cell("markdown", TRAIN_MD, "trained-md"),
    cell("code", TRAINING, "trained"),
    cell("markdown", DECISION_MD, "decisions-md"),
    cell("code", DECISIONS, "decisions"),
    cell("code", CEILING, "ceiling"),
    cell("markdown", RACE_MD, "races-md"),
    cell("code", RACES, "races"),
    cell("markdown", FAILURE_MD, "failure-md"),
    cell("code", FAILURE, "failure"),
    cell("markdown", EXTENDED_MD, "extended-md"),
    cell("code", EXTENDED, "extended"),
    cell("markdown", RERUN_MD, "rerun-md"),
    cell("code", RERUN, "rerun"),
    cell("markdown", LIMITS, "limits"),
]


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                "cells": CELLS}
    (DEST / "wikiracing-results.ipynb").write_text(json.dumps(notebook, indent=1))
    metadata = {"id": "uranium53/laya-wikiracing-base-vs-fine-tuned-results",
                "title": "Laya Wikiracing Base vs Fine Tuned Results", "code_file": "wikiracing-results.ipynb",
                "language": "python", "kernel_type": "notebook", "is_private": False,
                "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
                "keywords": [],
                "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (DEST / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {DEST}/wikiracing-results.ipynb with {len(CELLS)} cells")


if __name__ == "__main__":
    main()
