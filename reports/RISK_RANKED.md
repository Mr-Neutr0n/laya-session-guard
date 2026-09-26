# Risk-ranked Laya experiment

This run tests the next training method on a Kaggle GPU. It remains an experiment and cannot authorize agent tools.

## Training method

The previous runs used independent cross-entropy classification for each session and question. This run starts from the pinned English Laya base checkpoint and combines three losses:

1. Supervised content and action classification. Content receives half the weight of action.
2. A cost on nonallow cases whose `allow` score is too close to or above the correct `block` or `review` score. This explicitly penalizes unsafe allowance.
3. A paired ranking loss. For the same user task and source content, the legitimate action's `allow` score must outrank the attacker-goal action's `allow` score.

The run requests three epochs. It saves the epoch with the fewest raw incorrect allowances on calibration cases, then the most correctly allowed cases, then the highest action accuracy. Test cases never select the epoch. Both the objective and checkpoint selection may increase reviews or blocks of legitimate actions, so correct allowances are reported next to false allowances.

## Data and split

Training combines 480 original synthetic sessions, 576 counterfactual synthetic sessions, and 1,864 adapted ASPI workspace examples. Calibration has 648 cases. The public ASPI source is [ScaleAI/aspi](https://huggingface.co/datasets/ScaleAI/aspi), licensed CC BY 4.0. Its 407 workspace records specify a legitimate task, safe action, attacker goal, and benign/injected clarifications. The adaptation treats the clarification as an unverified lower-trust event and asks Laya about either the safe or attacker-proposed action. This is a new task built from ASPI fields, not ASPI's published agent evaluation. Labels are benchmark-generated, not human-reviewed for this schema.

Three frozen ASPI tests probe generalization:

| Split | Cases | What is unseen in training | SHA-256 |
| --- | ---: | --- | --- |
| `test_user` | 432 | User task IDs 34+ | `adda06fc2b8e33f393505aeab86c53b77dd74ded1588281fdd28001b16e94997` |
| `test_attack` | 352 | Attacker goal IDs 12-13 | `e13502a7aeee835be332ae2c9789b0e76ebd4fb3a3f03430288be6b1ce2e0bd4` |
| `test_both` | 72 | Both axes | `0ee73710df18869e949dae196dda9ee2bce9d574f2e460a09121ed54e3b8fa20` |

All 24 earlier challenge cases remain out of training and calibration. They are a diagnostic seen during previous analysis, not a fresh benchmark. The ASPI tests are external and larger, but they concern workspace tasks and synthetic attacker clarifications, not long coding sessions. The longest adapted input is 695 tokens; all fit Laya's 1,024-token cap.

## Results

The private [Kaggle notebook](https://www.kaggle.com/code/uranium53/laya-risk-ranked-aspi) completed on a Tesla T4 in 2,065 seconds (about 34 minutes). Its requested three epochs all ran. Epoch 2 was selected on calibration. The 648 calibration action cases were all correct at epochs 2 and 3, but this did not predict behavior on the frozen tests.

The table reports raw action decisions. An incorrect allow means the expected action was block or review and the model chose allow. Correct allows show whether the model retained the ability to approve legitimate actions.

| Frozen split | Model | Action accuracy | Incorrect allows | Correct allows | Block recall |
| --- | --- | ---: | ---: | ---: | ---: |
| Unseen user tasks (432) | Pinned base | 50.2% | 199/216 | 211/216 | 2.8% |
| | Risk-ranked | 97.2% | 4/216 | 208/216 | 98.1% |
| Unseen attacker goals (352) | Pinned base | 58.8% | 134/176 | 174/176 | 18.8% |
| | Risk-ranked | 81.3% | 66/176 | 176/176 | 62.5% |
| Both unseen (72) | Pinned base | 51.4% | 31/36 | 35/36 | 5.6% |
| | Risk-ranked | 75.0% | 18/36 | 36/36 | 50.0% |
| Earlier synthetic challenge (24) | Pinned base | 37.5% | 13/16 | 6/8 | 25.0% |
| | Risk-ranked | 66.7% | 7/16 | 7/8 | 62.5% |

The unseen-goal errors are concentrated rather than evenly spread. The model blocks every example of attacker goal 12, a conspicuous exfiltration/deletion request. It incorrectly allows 66 of 76 attacker actions for goal 13, which sends phishing-style meeting emails, even though all 76 matching safe actions are allowed. On the both-unseen split, it incorrectly allows 18 of 20 goal-13 attacker actions. It also incorrectly allows four cases of goal 1 on unseen user tasks. This suggests the model learned many familiar unauthorized-action cues but did not learn a reliable authorization rule for ordinary-looking email actions. That diagnosis is an inference from the per-goal error counts, not a causal proof.

On the earlier 24-case diagnostic, the previous 8-epoch original-data checkpoint had two raw incorrect allows and one wrapper incorrect allow. The risk-ranked checkpoint has seven raw and seven wrapper incorrect allows. This diagnostic was seen during previous analysis, so it cannot select a model reliably, but it is a clear regression warning. The risk-ranked checkpoint is **rejected as a replacement**. It remains in the private Kaggle output for research and was not installed locally. The local downloaded [report](../outputs/risk/laya-risk-ranked/report.json) and [challenge evaluation](../outputs/risk/laya-risk-ranked/challenge.json) preserve the measurements.

A follow-up error analysis of all 95 false allows, including the 66 above, is in [FALSE_ALLOW_ANALYSIS.md](FALSE_ALLOW_ANALYSIS.md). Three findings there sharpen this report. All 84 false allows on the ASPI splits come from goal 13 alone, not from a class of attacks, and 25 of them occur with a benign source and no injection anywhere in the transcript, where the model allowed an outward-facing email send the user never requested. The errors are confident rather than borderline, with a median allow margin of 11.6, so recalibration cannot help. A wrapper allow-margin threshold does reduce the false allows, but a threshold fitted by reading these splits reports 52 of 66 caught while the same rule cross-fitted on a different split and applied out of sample catches 31 of 66; the in-sample figure is roughly double the honest one and is not a held-out result. No threshold can be fitted on the calibration split for this checkpoint, because the calibration predictions were never persisted and the weights were never downloaded.

The cost-sensitive and pairwise method can be implemented and trained, but its current data and calibration set do not support a reliable guard. Next, collect consented, redacted, human-reviewed coding sessions with ordinary unauthorized actions, phishing-like messaging, conditional permission, and long context. Freeze an independent test set before another training run. A task-level policy check should enforce explicit authorization separately from this model's advisory score.
