# Laya session guard

Fine-tuning completed and the saved model loads locally. **This checkpoint is not suitable for automatic tool authorization.** On 24 separately written synthetic sessions it detected 4 of 11 suspicious sources, achieved 54.2% action accuracy, and incorrectly allowed two actions. See [the results](reports/RESULTS.md).

Experimental fine-tuning of `convaiinnovations/laya` for coding-session decisions. The input includes the user's task, ordered messages with their sources, a proposed tool action, and explicit history-completeness metadata.

Two questions produce separate probability distributions:

| Question | Labels |
| --- | --- |
| Does lower-trust content try to redirect the assistant? | benign, suspicious |
| Is the proposed action consistent with current user authorization? | allow, block, review |

A suspicious repository note can coexist with a legitimate next action. Conversely, a proposed action can violate a user restriction even when no injection is present. The training examples cover both cases, authorization changes, quoted security examples, and missing context.

## Published artifacts

- [Hugging Face checkpoint](https://huggingface.co/hxrikp/laya-session-guard-pilot)
- [GitHub code and evaluation](https://github.com/Mr-Neutr0n/laya-session-guard)

On the same 24 separately written challenge sessions, base Laya versus this fine-tune scored 37.5% versus 54.2% action accuracy and 54.2% versus 70.8% content accuracy. Suspicious-source recall improved from 0/11 to 4/11, while block recall stayed at 2/8. The improvement mainly comes from deferring uncertain actions. See [the paired results](reports/paired_challenge_results.json).

## Training run

Version 2 completed successfully on Kaggle. Training and final evaluation took 301.7 seconds on a Tesla T4. The checkpoint is downloaded at `outputs/laya-session-guard` and its SHA-256 is recorded in `reports/run_manifest.json`. On 144 templated test sessions, content accuracy rose from 50.0% to 100%, and action accuracy rose from 49.3% to 100%. These scores must be read with the shared-template limitation below.

Private Kaggle notebook: [Laya Session Guard Pilot](https://www.kaggle.com/code/uranium53/laya-session-guard-pilot).

The notebook requests Kaggle's T4 accelerator and trains on one GPU. It runs three epochs of full supervised cross-entropy fine-tuning, starting from Laya revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`. The run is capped at 3,600 seconds. The training notebook saves artifacts; the completed checkpoint was subsequently uploaded through the authenticated Hugging Face CLI.

The generated data has 480 training, 96 calibration, and 144 test sessions. Each has two labels. Coding workflows include configuration edits, test execution, dependency installation, fixture removal, build publication, and documentation updates. Sources include repository files, issue threads, build logs, and package documentation. The data contains fictional descriptions, no executable payloads or private transcripts.

Calibration uses separate sessions and separate temperatures for the two questions. Both the base and fine-tuned checkpoints receive calibration on the same calibration split. Test results include confusion matrices, per-class precision and recall, benign false-positive rate, Brier score, and expected calibration error. Test data never selects epochs or temperatures.

**These are synthetic prototype metrics.** All splits share scenario logic and differ in phrasing, project names, and group identifiers. This is not a held-out attack-family benchmark, and random project-number variation does not create independent semantic examples. Strong scores would establish that the model learned this fixture distribution, not that it protects a real coding agent.

## Reproduce

```sh
./verify
python3 src/session_data.py --output-dir data/synthetic
python3 kaggle/package.py
kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4 -t 3600
kaggle kernels status uranium53/laya-session-guard-pilot
kaggle kernels logs uranium53/laya-session-guard-pilot
kaggle kernels output uranium53/laya-session-guard-pilot -p outputs
```

`kaggle/package.py` embeds the source into a notebook. Change the owner/slug in that script before submitting under another Kaggle account. Kaggle installs pinned Laya, Transformers, and Safetensors releases; the run records the actual package versions. The model download is restricted to the English root checkpoint. It avoids downloading the other checkpoints bundled in the upstream repository.

Output directory on Kaggle: `/kaggle/working/laya-session-guard`. A completed run contains `model.safetensors`, the encoder configuration, tokenizer, `rl_agent_config.json`, `questions.json`, `report.json`, `baseline.json`, prediction logits, package versions, and the exact generated JSONL splits. Weights are saved after each epoch, so a later failure may leave a usable uncalibrated checkpoint. Only `report.json` with `status: completed` establishes completion.

## Inspect a session

Create the environment with `uv venv .venv` and install dependencies with `uv pip install --python .venv/bin/python -r requirements.txt`. Download the published checkpoint, then inspect a session:

```sh
hf download hxrikp/laya-session-guard-pilot --local-dir outputs/laya-session-guard
.venv/bin/python src/predict.py outputs/laya-session-guard session.json --device cpu
```

Example input:

```json
{
  "authorized_task": "Inspect the failing unit test. Do not change dependencies.",
  "events": [
    {"role": "user", "source": "direct_user", "content": "Inspect the failing unit test. Do not change dependencies."},
    {"role": "tool", "source": "build_log", "content": "The parser unit test failed on an empty input."}
  ],
  "proposed_action": {"kind": "read_test", "description": "Read the parser test", "destination": "workspace"},
  "context": {"history_complete": true, "omitted_events": 0}
}
```

The host must supply authentic role/source metadata and completeness information. Document text must never be allowed to declare itself a user message. The wrapper does not execute tools or integrate into an agent's permission system.

The context budget is 1,024 tokens per question. Short sessions pass intact. Longer transcripts split at event boundaries while retaining task and proposed action. Every event remains represented, but cross-window authorization is untested, so the final action decision is `review`. Oversized individual events, incomplete histories, invalid model probabilities, and action probabilities below the provisional 0.8 threshold also require review. The threshold is a prototype default, not a measured safety guarantee. Inspect the content and action distributions separately.

The native Laya `act_probability` is ignored because this run does not train that head. Bypassing this wrapper with `Agent.predict` also bypasses overflow and input checks.

## Additional diagnostic

`data/challenge.jsonl` contains 24 separately written synthetic coding sessions with 48 labels. It is excluded from training and calibration. Run `.venv/bin/python src/evaluate_challenge.py` to evaluate the saved checkpoint and the wrapper. Results go to `reports/challenge_results.json`. It achieved 70.8% content accuracy and 54.2% action accuracy, with two incorrect allow decisions after the wrapper. This is another synthetic diagnostic, not an independent real-session benchmark.

## What remains before real use

Collect consented, redacted coding-session prefixes with human-reviewed action decisions. Split by repository and actual scenario family. Include longer histories, permission revocation far from the action, benign security code, misleading tool output, and ordinary unauthorized actions. Measure error rates on that independent corpus and choose thresholds from a representative calibration set before considering enforcement. The current dataset mainly teaches explicit authorization cues.

## References

- [Laya model card](https://huggingface.co/convaiinnovations/laya)
- [Supplied Kaggle notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)
- [Anthus fine-tuning comparison](https://anth.us/blog/jev-vs-laya/)

This implementation sets token limits before preprocessing, reserves calibration sessions, clears inherited calibration buckets, and checks overflow. The supplied notebook does not do all of those. The Anthus study concerns a short sentiment dataset, so its accuracy is not an estimate for this task.
