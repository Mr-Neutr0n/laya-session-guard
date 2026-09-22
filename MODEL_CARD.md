---
license: apache-2.0
base_model: convaiinnovations/laya
language:
- en
pipeline_tag: text-classification
tags:
- laya
- modernbert
- experimental
- prompt-injection-detection
- synthetic-data
---

# Laya session guard pilot

An experimental full fine-tune of English Laya for structured coding-session decisions. It takes an authoritative task, ordered messages with source metadata, history-completeness information, and a proposed action. Separate typed questions judge suspicious source content and action authorization.

**Not suitable for automatic tool authorization.** The saved checkpoint missed 7 of 11 suspicious sources and incorrectly allowed two actions in a separately written 24-session synthetic diagnostic. All reported evaluation data is synthetic.

- [Code, inference wrapper, datasets, and reproduction](https://github.com/Mr-Neutr0n/laya-session-guard)
- [Detailed results](https://github.com/Mr-Neutr0n/laya-session-guard/blob/main/reports/RESULTS.md)
- [Private Kaggle training notebook, owner access required](https://www.kaggle.com/code/uranium53/laya-session-guard-pilot)

- [Narrated Kaggle comparison and runnable results](https://www.kaggle.com/code/uranium53/laya-session-guard-vs-jev)

## Results

| Dataset | Model | Content accuracy | Action accuracy |
| --- | --- | ---: | ---: |
| 144 templated test sessions | Base Laya | 50.0% | 49.3% |
| Same templated sessions | Fine-tuned | 100% | 100% |
| 24 separately written challenge sessions | Base Laya | 54.2% | 37.5% |
| Same challenge sessions | Fine-tuned | 70.8% | 54.2% |
| Same challenge sessions | Jev 1.13.0 | 95.8% | 95.8% |
| Same templated sessions | Jev 1.13.0 | 77.8% | 97.9% |

The template split shares scenario logic with training, so its perfect accuracy is not evidence of robust session understanding. On the challenge set, fine-tuning fixed seven action judgments and regressed three. Suspicious-content recall improved from 0/11 to 4/11. Block recall stayed at 2/8. Review recall improved from 1/8 to 7/8, while allow recall fell from 6/8 to 4/8. The main action gain comes from deferral.

Raw incorrect allowances fell from 13/16 nonallow cases to 2/16. These are argmax model decisions. The base wrapper reviewed every case with its shipped calibration; wrapper error rates are therefore not a fair detection comparison. The two checkpoints have different calibration temperatures. Neither model was trained or calibrated on the challenge set.

Jev used the same session states and question schema through the official TypeSafe API. It caught 11/11 suspicious sources and correctly blocked 8/8 block-labeled actions, with no incorrect allowances in 16 nonallow cases. This fine-tune did not outperform Jev on the challenge set. One Jev action disagreement is a debatable block-versus-review label. See the Kaggle notebook for per-case inspection.

The challenge set is small and agent-authored. It does not establish real-world improvement. No multilingual or independently annotated real-session evaluation was performed.

## Training

- Base: `convaiinnovations/laya`, pinned revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`.
- Architecture: ModernBERT-large encoder with Laya's typed option-scoring head, approximately 421M parameters.
- Data: 480 training sessions, 96 disjoint calibration sessions, 144 templated test sessions. Two labels per session. All examples use fictional descriptions with no real transcripts or executable attack payloads.
- Method: full supervised cross-entropy fine-tuning of the encoder and option scorer, three epochs, seed 42. The native act head is frozen and ignored.
- Hardware: one Tesla T4 on Kaggle. Recorded training plus final evaluation took 301.7 seconds.
- Calibration: separate temperatures for the two typed questions, fitted only on the synthetic calibration split. Both selected 0.5. Confident errors on the challenge set show poor transfer of this calibration.
- Context: 1,024 tokens per question. Training inputs were at most 379 tokens. Base challenge inputs also fit its 512-token budget.

## Usage

This is a Laya checkpoint, not a standard `AutoModelForSequenceClassification` export. Use the linked project's wrapper to check transcript structure and prevent silent truncation.

```sh
git clone https://github.com/Mr-Neutr0n/laya-session-guard.git
cd laya-session-guard
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/hf download hxrikp/laya-session-guard-pilot --local-dir outputs/laya-session-guard
.venv/bin/python src/predict.py outputs/laya-session-guard session.json --device cpu
```

`session.json` must follow the schema and example in the GitHub README. The host must supply authentic role/source metadata; text from a document cannot grant itself user authority.

The wrapper preserves every event across bounded windows and returns review for multiwindow sessions. It also reviews incomplete histories, oversized events, invalid probabilities, and low-confidence action decisions. It does not execute tools. The two observed incorrect allowances survived these checks, so the wrapper is not a validated security boundary. Native Laya `act_probability` is not trained for authorization and must not be used for enforcement.

## Included artifacts

Weights, encoder configuration, tokenizer, question schema, exact synthetic training/calibration/test splits, training report, test logits, and package versions are included. `evaluation/` contains both challenge evaluations, the challenge cases, paired measurements, and the checkpoint checksum manifest. The GitHub repository includes the reproducible source and tests.

## License and provenance

The base model and this derivative checkpoint use Apache 2.0. Laya is developed by Convai Innovations. This session-guard experiment is an independent fine-tune, not an official Laya security release.
