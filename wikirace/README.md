# Laya Wikiracing experiment

This experiment uses Stanford SNAP's archived [Wikispeedia dataset](https://snap.stanford.edu/data/wikispeedia.html), not live Wikipedia links. The game has 4,604 articles. The source archive contains 119,882 links and 51,318 finished human paths. `build_data.py` retains 119,772 directed, non-self links between known articles and 44,280 paths whose steps can be verified against that graph. Article titles in the app link to live Wikipedia for reading, but live edits do not alter a race.

The model chooses one of up to 12 actual outgoing links at each step. The deterministic shortlist uses title overlap with the target and link indegree; it never reads shortest-path distance. The shortlist is shuffled deterministically to remove position cues. Human paths supply current-page/target pairs, and reverse graph search labels the candidate that reaches the target in the fewest remaining hops. Ties prefer the human's next click when that click is optimal, or use a deterministic hash. Paths do not supply labels directly when they wander away from the shortest route.

Targets are disjoint across training, calibration, and test. There are 6,000 training decisions, 600 calibration decisions, 600 held-out decisions, and 60 held-out races with a limit of eight moves each. Of the 60 races, 55 have a route within eight moves through the fixed shortlist, as checked by graph search. That is the maximum possible win count for a perfect agent with this shortlist. The archive is a historical, reduced Wikipedia; success here does not imply performance on today's full Wikipedia.

The Kaggle notebook evaluates pinned English base Laya before updating weights, trains the encoder and choice scorer for three epochs, fits choice temperatures on the calibration targets, and repeats the same decisions and races. The native act head is frozen. The comparison records top-one choice accuracy, Brier score, calibration error, win rate, hops, and warm GPU inference p50/p95 latency. These latency numbers include batch preparation and device transfer but exclude tokenization, graph lookup, browser rendering, and model loading. The app reports locally measured inference timing separately, including tokenization.

## Rebuild and run

Download [Stanford's paths-and-graph archive](https://snap.stanford.edu/data/wikispeedia.html) into `data/wikirace/source/`, extract it there, then run:

```sh
PYTHONPATH=. python3 -m wikirace.build_data
python3 kaggle/package_wikirace.py
kaggle datasets create -p kaggle/wikirace-dataset -t
kaggle kernels push -p kaggle/wikirace --accelerator NvidiaTeslaT4 -t 7200
```

For this run, the private Kaggle dataset is [laya-wikirace-data](https://www.kaggle.com/datasets/uranium53/laya-wikirace-data) and the private notebook is [laya-wikiracing-base-vs-fine-tuned](https://www.kaggle.com/code/uranium53/laya-wikiracing-base-vs-fine-tuned). The [measured results](../reports/WIKIRACE_RESULTS.md) compare both checkpoints. Download and prepare the checkpoints before starting the [app](../app/README.md):

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
mkdir -p outputs/wikirace
kaggle kernels output uranium53/laya-wikiracing-base-vs-fine-tuned -p outputs/wikirace
mv outputs/wikirace/laya-wikirace outputs/wikirace/tuned
PYTHONPATH=. .venv/bin/python scripts/setup_wikirace_models.py
.venv/bin/python app/server.py
```

The source archive, Kaggle dataset copy, and checkpoints are ignored by Git. The derived graph, decision splits, race list, provenance, code, and notebook are in the repository.
