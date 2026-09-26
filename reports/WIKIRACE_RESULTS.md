# Wikiracing results

Source: archived Stanford Wikispeedia graph. The same 600 held-out choices and 60 held-out races were used for both checkpoints. Test targets do not occur in training or calibration.

| Measure | Base Laya | Fine-tuned Laya |
| --- | ---: | ---: |
| Choice accuracy, 600 decisions | 37.0% | 47.2% |
| Choice accuracy without direct target, 489 decisions | 23.5% | 35.2% |
| Choice Brier score | 0.765 | 0.632 |
| Choice ECE, 10 bins | 0.051 | 0.046 |
| Races won within 8 hops, 60 races | 13 (21.7%) | 23 (38.3%) |
| Model latency p50, T4 | 34.5 ms | 33.8 ms |
| Model latency p95, T4 | 37.5 ms | 36.3 ms |

Paired choice-accuracy difference (tuned minus base): +10.2%; 95% paired bootstrap interval [6.2%, 14.2%].
Paired race-win difference: +16.7%; 95% paired bootstrap interval [5.0%, 30.0%]. The fine-tune won 13 races that base lost and lost 3 that base won.

The shortlist permits a path within eight moves for 55 of 60 races. Model latency measures warm GPU inference including batch preparation and transfer, but excluding tokenization, checkpoint load, graph lookup, browser rendering, and live network requests. The game uses a fixed historical subset of Wikipedia, so these results do not establish performance on the current full site.

Training took 1068.6 seconds on Tesla T4. Base revision: `1c5edc17a7acd8701df6fc341c0d179f1c62c982`. See `outputs/wikirace/tuned/report.json`, `race_traces.json`, and `decision_predictions.json` for per-case results.
