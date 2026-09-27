# Paper-triage demo

A runnable demo of the published router: feed it a title and an abstract, get the
distribution over eight arXiv subject areas and the top choice. It loads
[`hxrikp/laya-arxiv-topic-router`](https://huggingface.co/hxrikp/laya-arxiv-topic-router)
at revision `1c2bb313b6a94b674c2e2c57483760eb5ccab244`, the exact weights that were
released, and runs on CPU. The revision is pinned in `src/router_demo.py` and passed
to `snapshot_download`, so this does not silently pick up a later commit.

| path | what it is |
| --- | --- |
| `src/router_demo.py` | the CLI |
| `scripts/build_demo_cases.py` | builds the verification set from arXiv metadata |
| `scripts/build_demo_view.py` | renders `index.html` from a predictions JSON |
| `demo/verification_cases.jsonl` | 5 real papers from 2025, 5 contrived cases |
| `demo/predictions.json` | the CLI output for that set |
| `demo/index.html` | static view, 17 kB, no server and no network |

## Run it

```sh
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt pyarrow fsspec

# one paper
.venv/bin/python src/router_demo.py \
    --title "Black hole entropy from Euclidean path integrals" \
    --abstract "We compute the entropy of a black hole in three dimensions as the \
expectation value of a Wilson loop in a Euclidean quantum gravity theory."

# a batch, then the static view
.venv/bin/python scripts/build_demo_cases.py
.venv/bin/python src/router_demo.py --file demo/verification_cases.jsonl --out demo/predictions.json
.venv/bin/python scripts/build_demo_view.py
open demo/index.html
```

Other input forms: `--stdin` reads the same JSONL from a pipe, `--max-words` changes
how much of the abstract is kept, `--revision` moves off the pin deliberately, and
`--device` accepts any torch device. Add `--model` to run against a local checkpoint
instead of the Hub.

The CLI writes JSON to stdout by default and prints a one-line summary per case to
stderr, so it pipes cleanly. The HTML view embeds the JSON rather than fetching it,
because a `file://` page cannot `fetch()` its neighbour. There is no CDN and no
build step; the file is 17 kB and opens by double click.

## What the demo actually produces

Ten cases, this machine, batch of 4, 712 to 856 ms per example. The five real papers
come from arXiv metadata submitted between 2025-05-06 and 2025-10-20, sixteen years
after the training window closed.

| case | gold primary category | top choice | probability | margin over runner-up |
| --- | --- | --- | --- | --- |
| AI as IA: the use and abuse of AI for human enhancement | `cs.CY` | `quant-ph` | 0.37 | 0.10 |
| A Variant of Chaitin's Omega function | `math.LO` | `math.PR` | 0.52 | 0.11 |
| Frequency Response Identification of Low-Order Systems | `eess.SY` | `math.PR` | 0.73 | 0.56 |
| On L1-approximation of groups | `math.GR` | `math.AG` | 0.29 | 0.07 |
| Gorenstein Normal tangent cones of integrally closed ideals | `math.AC` | `math.AG` | 0.96 | 0.94 |
| contrived: three-word abstract, no signal | none | `math.AG` | 0.61 | 0.42 |
| contrived: ambiguous transport title | none | `cond-mat.mes-hall` | 0.65 | 0.34 |
| contrived: ribosome profiling, biology | none | `quant-ph` | **0.88** | 0.82 |
| contrived: cloud droplet chemistry | none | `cond-mat.mtrl-sci` | 0.32 | 0.19 |
| contrived: French abstract, English title | none | `gr-qc` | 0.43 | 0.10 |

## Weaknesses the demo makes obvious

**There is no abstain option, so a confident wrong answer is the default failure.**
None of the five real papers has its true category among the eight options: `cs.CY`,
`math.LO`, `eess.SY`, `math.GR` and `math.AC` are all outside the set. Every one of
those five answers is therefore wrong, and the router cannot say so. On a 2025 paper
it will hand you a subject area with a number attached and no indication that the
question was malformed.

**Confidence does not detect out-of-distribution input.** Mean top probability is 0.57
on the real out-of-window papers and 0.58 on the contrived cases, which is the same
range as in-distribution behaviour. The worst case in the whole set is the ribosome
paper at 0.88 for `quant-ph`, and the frequency-response control-systems paper is
0.73 for `math.PR`. Both are confidently wrong, and both are more confident than
several correct in-window predictions. Any caller that thresholds on this number is
gating on nothing.

**A near-flat distribution is the only visible warning.** The cloud-droplet case
spreads across five areas with a top probability of 0.32, and the AI-ethics paper sits
at 0.37 with a 0.10 margin. Those two are visibly hedged. The margin over the runner-up
is more informative than the top probability, and even then it is a weak signal, not a
threshold.

**The label set is a 2007-2009 artefact.** `math.AG` and `math.PR` are the only
options a modern mathematics paper has a chance of landing in, and the run of `math.AG`
on algebraic-geometry-adjacent papers means the router is partly acting as a coarse
"is this pure mathematics" detector. The physics options are pinned to archive codes
that have since split, `astro-ph` in particular.

**Empty and short abstracts still get a confident answer.** Three words of text about
an unspecified problem produced `math.AG` at 0.61. The model has no way to express
"not enough information", and nothing in the pipeline requires an abstract.

**The ambiguity case is handled, which is worth saying.** A transport-coefficient
title with a diffusion-constant abstract split 0.65 `cond-mat.mes-hall` against 0.31
`math.PR`, which is the defensible reading: mesoscopic transport rather than
probability. A French abstract under an English gravitational-wave title still
returned `gr-qc`, with a flat 0.43 against 0.33 `quant-ph`, so a non-English body
degrades the answer into a near-coin-flip rather than breaking it.

## If you want to use this

Read the two caveats in the model card first. The honest summary is that this is a
demonstration of a metadata-only fine-tune on eight 2007-2009 subject areas, useful
for showing the base-versus-fine-tuned gap and for exploring Laya's choice head. It is
not a triage classifier, and the demo above is the evidence for that, not a caveat
understated in a footnote.
