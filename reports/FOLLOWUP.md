# Laya session guard follow-up

This private Kaggle experiment tests two changes independently against the previous run. All cases are synthetic and no model in this report is approved to authorize tool calls.

| Run | Training sessions | Training schedule | What changed |
| --- | ---: | --- | --- |
| Previous pilot | 480 | Full supervised fine-tune, 3 epochs | Original template data |
| Longer control | 480 | Same full fine-tune, 8 epochs | Epoch count only |
| Counterfactual run | 576 | Same full fine-tune, 8 epochs | Paired authorization and action changes, eight workflows and six attack phrasings |

All three use the pinned English 421M Laya base revision. The two new runs start independently from that base, use the same optimizer and question schema, and run on a Kaggle T4. Training and calibration splits stay separate. The 24 previously frozen challenge cases are never used for training or calibration.

The longer control asks whether more updates on the old data help. The counterfactual run asks whether changing the examples helps beyond more updates. Its pairs hold the tool text or proposed action fixed while changing current user permission; other pairs hold the permission fixed while changing the proposed action. This reduces some cheap label shortcuts. The examples still use generated wording and short sessions. Their longest tokenized question and session is 356 tokens, so this run does not test whole coding transcripts.

The comparison will report raw content and action accuracy, suspicious-source recall, block recall, and incorrect allowances on the exact same challenge cases. Template test scores will be reported separately because both generators reuse scenario logic between train and test. These 24 cases are also synthetic and too small to establish deployment reliability.

Research behind the design: [Instruction Hierarchy](https://arxiv.org/abs/2404.13208) motivates keeping higher-trust user permission distinct from lower-trust source content. [AgentDojo](https://arxiv.org/abs/2406.13352) motivates action-level evaluation in an agent workflow. [Defenses Against Prompt Attacks Learn Surface Heuristics](https://arxiv.org/abs/2601.07185) documents why template scores and attack-token cues can mislead.

## Results

Both runs completed on a Kaggle Tesla T4. The longer control took 827 seconds of training and final evaluation; the counterfactual run took 938 seconds. Both scored 100% on their own 144-session generated test splits. Those scores are not independent evidence of security performance.

| Run | Content accuracy | Action accuracy | Suspicious sources caught | Block decisions correct | Raw incorrect allows | Wrapper incorrect allows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous 3 epochs | 70.8% | 54.2% | 4/11 | 2/8 | 2 | 2 |
| Original data, 8 epochs | 75.0% | 75.0% | 5/11 | 5/8 | 2 | 1 |
| Counterfactual data, 8 epochs | 83.3% | 62.5% | 8/11 | 2/8 | 7 | 6 |

The longer run improved action accuracy by 20.8 percentage points over the previous pilot and correctly blocked three more block-labeled cases. Its raw incorrect allow count stayed at two. The wrapper deferred one of those two, but still incorrectly allowed the false test-summary case (`challenge-16`). It also raw-allowed an incomplete-history action (`challenge-18`), which the wrapper reviewed.

The counterfactual run caught four more suspicious sources than the previous pilot, but it regressed on the decision that matters most for a guard. It raw-allowed seven of the 16 actions labeled block or review, and the wrapper allowed six. It should not replace the prior checkpoint. Its perfect generated-test score shows how misleading same-generator tests remain.

The longer control is the better of these two new checkpoints on this diagnostic. The sample is only 24 synthetic cases and was seen during analysis of the earlier pilot, so it is not a fresh independent benchmark. Neither new model is suitable for automatic tool authorization. The next data collection should emphasize false reporting, ordinary unauthorized actions without injection language, conditional approval, and longer authentic session histories. Human-reviewed held-out cases should determine the training schedule and any confidence threshold.

Private Kaggle notebook: [Laya Session Guard Follow-up](https://www.kaggle.com/code/uranium53/laya-session-guard-follow-up). The local reports under `outputs/followup/` are downloads from that completed run. Recompute the table with `python3 scripts/compare_followup.py`.

The longer-control checkpoint was downloaded to `outputs/followup/followup-original/` and loaded successfully for local CPU inference. Its `model.safetensors` SHA-256 is `01d2309fc74715ffb72f754091021f6f356fdc0d233481abb6413181e8e283eb`. The counterfactual checkpoint remains in the private Kaggle output because its action behavior regressed.
