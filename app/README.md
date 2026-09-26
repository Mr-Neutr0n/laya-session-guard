# Wikirace app

The app races base and fine-tuned Laya checkpoints on the archived Wikispeedia article graph. Both receive the same shortlist and question from `wikirace/graph.py`. The browser never fetches Wikipedia pages during a race; article titles link to live Wikipedia for optional reading.

Run from the repository root:

```sh
LAYA_BASE_MODEL=outputs/wikirace/base \
LAYA_TUNED_MODEL=outputs/wikirace/tuned \
LAYA_DEVICE=cpu python3 app/server.py
```

Open `http://127.0.0.1:8765`. The app loads `data/wikirace/graph.json` and `data/wikirace/eval_races.json`. It shows an unavailable checkpoint as such, without substituting a heuristic result. Model files stay in local directories and the server loads each checkpoint on its first move.

`/api/races` lists the evaluation routes, checkpoint availability, and recorded timing. `POST /api/step` accepts `{"model":"base","current":0,"target":1,"visited":[0]}` and returns a legal next article, all candidate probabilities, and model inference time. `/api/timing` returns observed p50 and p95 inference time for each checkpoint. Timing starts after lazy loading, so it excludes model download/load, browser work, and page fetch time.
