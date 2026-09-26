# Wikiracing integration readiness

Audit of `codex/laya-wikirace` for merge into `main`. No merge, push, or publication has been
performed. This file is the reviewer's map of the branch.

- Branch tip at audit time: `47e6700`.
- Published base: `origin/main` = `963269d`, an ancestor of the branch tip. The full branch diff
  against that base is 39 files and 11,729 added lines, all additions except seven lines in
  `.gitignore` and `README.md`.
- Published already, and unaffected by the merge decision:
  - Checkpoint and evaluation artifacts: <https://huggingface.co/hxrikp/laya-wikirace-wikispeedia>
  - Public results notebook: <https://www.kaggle.com/code/uranium53/laya-wikiracing-base-vs-fine-tuned-results>

## Read this first: the local `main` ref moved during the audit

At 20:18:53 on 2026-09-26, something outside this worktree ran
`git fetch . codex/laya-wikirace:main: fast-forward` from `/Users/harikp/Developer/scratch/laya`,
which fast-forwarded the shared local `main` ref from `963269d` to `93c0ad2`. The Wikiracing
worktree did not do this, and nothing was pushed: `origin/main` is still `963269d`, and the
other checkout is now sitting on an unrelated branch with uncommitted edits to session-guard
files.

Consequences for you:

- The branch is **not** fully merged into `main` any more. Local `main` contains everything up to
  `93c0ad2`, and only `47e6700`, the 240-route replay, is outstanding.
- The diffstat below is given against `origin/main` = `963269d`, which is the honest full-branch
  baseline. The remaining delta against local `main` is one commit and six files.
- Undoing the ref move is your call, not this branch's. If you want `main` back at `963269d`, say
  so and it can be reset, but the other checkout's in-progress work is the reason to think twice.

## Diffstat versus origin/main (`963269d`)

```
39 files changed, 11729 insertions(+)
```

```
 .gitignore                                         |   5 +
 README.md                                          |   2 +
 app/README.md                                      |  34 +
 app/app.js                                         |  221 +
 app/favicon.svg                                    |   5 +
 app/index.html                                     |  40 +
 app/model_adapter.py                               |  75 +
 app/server.py                                      | 145 +
 app/style.css                                      |  20 +
 data/wikirace/calibration.jsonl                    |  600 ++
 data/wikirace/eval_races.json                      |  302 ++
 data/wikirace/graph.json                           |    1 +
 data/wikirace/provenance.json                      |   19 +
 data/wikirace/test.jsonl                           |  600 ++
 data/wikirace/train.jsonl                          | 6000 +++++++++++++++++++++
 kaggle/package_wikirace.py                         |   52 +
 kaggle/package_wikirace_results.py                 |  402 ++
 kaggle/wikirace-dataset/dataset-metadata.json      |   10 +
 kaggle/wikirace/kernel-metadata.json               |   15 +
 kaggle/wikirace/wikirace.ipynb                     |    1 +
 kaggle/wikiracing-results/kernel-metadata.json     |   16 +
 kaggle/wikiracing-results/wikiracing-results.ipynb |  456 ++
 reports/WIKIRACE_RESULTS.md                        |   20 +
 scripts/browser_check.ts                           |   90 +
 scripts/compare_wikirace.py                        |   64 +
 scripts/race_eval.py                               |   70 +
 scripts/race_smoke.py                              |   74 +
 scripts/setup_wikirace_models.py                   |   42 +
 tests/test_wikirace.py                             |   52 +
 tests/test_wikirace_app.py                         |   84 +
 tests/test_wikirace_server.py                      |   95 +
 wikirace/README.md                                 |   38 +
 wikirace/__init__.py                               |    1 +
 wikirace/build_data.py                             |  109 +
 wikirace/graph.py                                  |   66 +
 wikirace/train.py                                  |  241 +
```

Every one of those 39 files is new except two. The branch is a pure addition plus seven lines of
edits to shared files, and it never touches `src/`, `MODEL_CARD.md`, `verify`, `requirements.txt`,
or any existing `reports/*.json`. The session-guard experiment on `main` is unaffected.

Data volume: 5.2 MB of JSONL and graph, 46 MB including the git-ignored source archive directory
placeholder. `data/wikirace/graph.json` is a single 667 KB line, so any diff of it is unreadable;
compare it by hash instead.

## File-by-file risk notes

Risk is about merge and review friction, not code quality. Nothing here is high risk.

| File | Risk | Note |
| --- | --- | --- |
| `.gitignore` | Low, but the only likely conflict | Appends five Wikirace patterns after `.DS_Store`. Additive. If `main` later appends its own patterns, resolve by keeping both sets. |
| `README.md` | Low | Two lines: a pointer sentence below the title. Additive, resolves automatically against unrelated edits. |
| `app/app.js` | Medium, needs eyes | 221 lines of untested browser code. Race loop, two lanes, mode switching, trail rendering. Verified only through `scripts/browser_check.ts` in headless Chrome, not by unit tests. Read it before merging. |
| `app/server.py` | Medium | Local HTTP server on `127.0.0.1`, request validation, lazy model load, per-lane timing. Binds a port and loads ~840 MB of weights. Covered by `tests/test_wikirace_server.py`. |
| `app/model_adapter.py` | Medium | Wraps the `laya` package and normalizes probabilities. Its calibration behaviour is the difference between a wrong lane and a correct one; the tests cover the normalization and load-serialization paths. |
| `app/index.html`, `app/style.css`, `app/favicon.svg`, `app/README.md` | Low | Static assets. No logic beyond the mode radio buttons. |
| `wikirace/graph.py` | Medium, load-bearing | The shortlist is the experimental control: legal outgoing links only, ranked by title overlap and log indegree, never by distance. A change here silently changes every reported number. Treat as frozen. |
| `wikirace/train.py` | Medium | The training recipe. Requires CUDA and the pinned base revision, so it cannot be smoke-tested locally. Review the loss, masking, and freezing logic rather than running it. |
| `wikirace/build_data.py` | Low | Rebuilds the graph and splits from the SNAP archive. Not run during review; the committed artifacts carry `provenance.json` with the source archive hash. |
| `data/wikirace/*.jsonl`, `graph.json`, `eval_races.json`, `provenance.json` | Low | Frozen measured artifacts, 5.2 MB. `train.jsonl` is 6,000 lines. Do not regenerate casually: the splits are what the published metrics were computed on. |
| `kaggle/wikirace/wikirace.ipynb` | Low, opaque | One line. Generated by `kaggle/package_wikirace.py`, which base64-embeds the three training modules. Verify with the packer, not by reading it. |
| `kaggle/wikiracing-results/wikiracing-results.ipynb` | Low, opaque | 456 lines, generated by `kaggle/package_wikirace_results.py`. Pinned to HF revision `510cb4f…`, so the published numbers are stable as long as those artifacts are. |
| `kaggle/package_wikirace.py`, `kaggle/package_wikirace_results.py` | Low | Deterministic generators for the two notebooks. Both are run by `verify` only for syntax, so re-run one if a notebook looks stale. |
| `kaggle/*/kernel-metadata.json`, `kaggle/wikirace-dataset/dataset-metadata.json` | Low | Point at Hari's Kaggle account and are already live. Editing them post-merge would change what a future push overwrites, not the published kernels. |
| `data/wikirace/eval_races_240.json` | Low | The first 240 routes of the same deterministic ordering, 1,202 lines. The published 60 are a verified prefix, and `scripts/build_wikirace_races.py` asserts that before writing. |
| `scripts/build_wikirace_races.py` | Low | Re-derives the race ordering from the SNAP archive with the same seed and cap. Needs the ignored source archive, so it is not part of `verify`. |
| `scripts/compare_wikirace.py` | Low | Regenerates `reports/WIKIRACE_RESULTS.md` from the run outputs. Only rerunnable with the Kaggle outputs present. |
| `scripts/setup_wikirace_models.py` | Low | Downloads the pinned base revision and writes calibration temperatures. Requires network and the Kaggle outputs. |
| `scripts/race_smoke.py`, `scripts/race_eval.py` | Low | Local verification tools. `race_eval.py` takes about ten minutes on CPU. Not part of `verify`. |
| `scripts/browser_check.ts` | Low | Deno plus headless Chrome over the DevTools protocol. Not part of `verify`, and no npm dependency. |
| `tests/test_wikirace*.py` | Low | 7 new tests. `test_wikirace.py` re-derives shortlists for all 7,200 decisions, which is why `verify` takes a few seconds. |
| `wikirace/README.md`, `app/README.md`, `reports/WIKIRACE_RESULTS.md` | Low | Documentation and the measured results table. `WIKIRACE_RESULTS.md` is generated; edit `scripts/compare_wikirace.py` if it needs to change. |

## Review checklist

Read these in order. Everything else is data or a generator.

- [ ] `wikirace/graph.py`: confirm the shortlist never reads distance to the target, and that
      `exclude` is applied before ranking.
- [ ] `wikirace/train.py`: confirm the act head stays frozen, the calibration split is never used
      for gradient steps, the temperature search uses only calibration data, and the written
      `training` block describes this run rather than the base checkpoint.
- [ ] `app/server.py` request validation: bounds on `current`, `target`, and `visited`, the
      Explore-only `visited` requirements, the 16 KB body cap, and that a failed model load
      returns 503 rather than a heuristic move.
- [ ] `app/model_adapter.py`: probability normalization, rejection of non-finite or
      out-of-candidate answers, and that a checkpoint is loaded once.
- [ ] `app/app.js`: hop limits 8 and 16, the `runId` guard against overlapping races, and that
      Benchmark sends no race history.
- [ ] `reports/WIKIRACE_RESULTS.md` against `outputs/wikirace/tuned/report.json` and the published
      HF artifacts. Numbers must match exactly.
- [ ] Confirm the two public links in the model card and notebook resolve.
- [ ] Run `./verify` and read the 24 test names.

## Commands to run after approval

Nothing below has been run against `main`. Review first, then:

```sh
# 1. Merge. origin/main is an ancestor of the branch, so this is a fast-forward with no rebase.
cd ~/Developer/scratch/laya
git status --short                     # expect clean, and note which branch you are on
git merge --ff-only codex/laya-wikirace
./verify                               # expect 24 tests, OK

# 2. Push main, only if you want the branch public.
git push origin main
git push -u origin codex/laya-wikirace

# 3. Optional: prove the merged tree still serves the published checkpoint.
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt --offline
PYTHONPATH=. .venv/bin/python scripts/setup_wikirace_models.py
LAYA_BASE_MODEL=outputs/wikirace/base LAYA_TUNED_MODEL=outputs/wikirace/tuned \
  LAYA_DEVICE=cpu .venv/bin/python app/server.py --port 8765 &
.venv/bin/python scripts/race_smoke.py http://127.0.0.1:8765 0
.venv/bin/python scripts/race_eval.py http://127.0.0.1:8765   # about ten minutes, 60/60 agreement
```

Publication needs no action. The checkpoint and the public notebook are already live, and neither
is affected by this merge. If you later want to republish after editing a notebook, regenerate it
with `python3 kaggle/package_wikirace_results.py` and push with
`kaggle kernels push -p kaggle/wikiracing-results`; re-uploading the Hugging Face repo requires
`hf upload hxrikp/laya-wikirace-wikispeedia . .` from a staging directory that contains
`model.safetensors`, the configs, and `evaluation/`.

## Known follow-ups, not blockers

- The Hugging Face model card no longer claims the code is on the repository's `main`. It now
  states that the Wikiracing code lives on the unpushed `codex/laya-wikirace` branch, that `main`
  does not contain these files, and that the evaluation artifacts plus the training entry point are
  already recoverable from this repository and the training notebook. Card revisions so far:
  `30408ad` first upload, `fba1fa4` added `evaluation/graph.json`, `510cb4f` added the replay note,
  `62cdb93` and `b9a2b03` card-only text fixes, then `e61c218` and `16b076b` added the 240-route
  records and `81d6d21` added the 240-route card section. The public results notebook is pinned to
  `81d6d21`, which the last Kaggle run confirmed end to end. Every earlier revision is still
  resolvable, and no file other than `README.md` changed after `b9a2b03`.
- The Explore mode's model input now includes the last three visited articles, which was not part
  of the published measurements. Benchmark, and therefore every published number, is unchanged.
- The achievable ceiling is 55 of the 60 Kaggle routes and 222 of the 240 replayed routes. A
  path-aware shortlist is the next experiment worth running; it would change the reported numbers,
  so it belongs in a new branch.
- The 240-route replay was measured locally on CPU, not on Kaggle. It reproduces the Kaggle traces
  exactly on the shared 60 routes, but it is a single-host measurement and is labelled as such in
  both the model card and the public notebook.
