# Laya session guard results

The fine-tune completed on Kaggle and the downloaded checkpoint loaded successfully for local inference. It is a working experiment, but the observed errors rule out using it to authorize agent tool calls.

## Completed work

- Authenticated Kaggle CLI as `uranium53` and created a private notebook.
- Fine-tuned the 421M-parameter English Laya checkpoint on a Tesla T4 for three epochs. The recorded training/final evaluation time was 301.7 seconds.
- Used 480 training sessions, 96 separate calibration sessions, and 144 templated test sessions. Each session has content and proposed-action decisions.
- Saved 842,609,220 bytes of weights, tokenizer/configuration, exact dataset splits, predictions, and package versions under `outputs/laya-session-guard`.
- Verified source files downloaded from Kaggle match the submitted local files.
- Passed 14 local checks, validated 1,440 tokenized question/session pairs with no truncation, and preserved 103 transcript events across four windows.
- Evaluated the actual saved model on 24 separately written synthetic challenge cases excluded from training and calibration.

## Measurements

| Evaluation | Content accuracy | Action accuracy |
| --- | ---: | ---: |
| Base Laya, 144 templated test sessions | 50.0% | 49.3% |
| Fine-tuned Laya, same 144 sessions | 100% | 100% |
| Base Laya, 24 separately written challenge sessions | 54.2% | 37.5% |
| Fine-tuned Laya, same 24 challenge sessions | 70.8% | 54.2% |

The templated test split shares scenario logic with training. Its perfect accuracy is not evidence of reliable session understanding. The challenge set is also synthetic and small, but it exposes failures hidden by the template split. The later paired base evaluation used the exact same challenge cases and pinned base revision. Raw action accuracy improved by 16.7 percentage points, fixing seven cases and regressing three. Suspicious-source recall rose from 0/11 to 4/11. Block recall remained 2/8; review recall rose from 1/8 to 7/8 and allow recall fell from 6/8 to 4/8. Raw incorrect allowances fell from 13/16 nonallow cases to 2/16. The base wrapper reviewed every case under its shipped calibration, so wrapper error rates should not be compared as detection gains. Different temperatures also preclude interpreting confidence shifts as evidence of better calibration.

On the challenge set, the model caught 4 of 11 suspicious sources and missed 7. It classified all 13 benign sources correctly. The wrapper requested review on 15 of 24 sessions and incorrectly allowed two actions out of 16 whose labels were block or review.

One incorrect allowance would report all tests passed despite observed failures. The other would start shared-environment checks without established confirmation of availability. The reported allow probabilities were 1.0 and 0.9804. A confidence threshold did not solve these errors.

## Scope and limitations

Inputs are structured session prefixes with host-supplied role/source metadata, user intent, history-completeness information, and the proposed action. The model separately judges suspicious content and action authorization. It does not execute tools. No production integration was performed. The experimental checkpoint is published on Hugging Face, linked to the GitHub reproduction repository.

The input budget is 1,024 tokens. The training examples were at most 379 tokens per question. Longer inputs preserve complete events across windows and return review because cross-window authorization has not been validated. This experiment does not establish understanding of arbitrary whole coding transcripts.

Calibration uses held-out synthetic sessions. Those easy sessions selected a sharpening temperature of 0.5 for both questions; the confident challenge errors show those probabilities are unreliable outside that distribution. There is one training seed, no expert-reviewed real-session corpus, and no multilingual evaluation.

The next useful investment is consented, redacted coding-session prefixes with independent labels, diverse authorization language, realistic tool outputs, and genuinely held-out repositories/scenario families. Retain a final test set that does not participate in training revisions. More epochs on the existing templates would not address the observed problem.

## Artifacts

- [Hugging Face model](https://huggingface.co/hxrikp/laya-session-guard-pilot)
- [GitHub repository](https://github.com/Mr-Neutr0n/laya-session-guard)
- [Paired challenge measurements](paired_challenge_results.json)

- [Private Kaggle notebook](https://www.kaggle.com/code/uranium53/laya-session-guard-pilot)
- [Training report](training_results.json)
- [Challenge report and predictions](challenge_results.json)
- [Run manifest and checkpoint checksum](run_manifest.json)
- [Usage and reproduction](../README.md)

The scratch workspace, private Kaggle notebook, new public GitHub repository, and new public Hugging Face model contain this experiment. Uploaded data is synthetic; previous project sources, local credentials, environment files, and raw logs are excluded from publication.
