# Release state

Handoff for the Laya session-guard workstream. One page: what is published, what is rejected and
why, the corrected numbers, and the work that needs a human.

**Nothing in this project is a safety guard.** Every published checkpoint is rejected. Explicit
authorization is enforced by a task-level policy check outside the model, permanently.

- Reports: `reports/RISK_RANKED.md`, `reports/FALSE_ALLOW_ANALYSIS.md`, `reports/AUTHZ_GAP_AUDIT.md`,
  `reports/FOLLOWUP.md`, `reports/REVIEW_PROTOCOL.md`
- Branch: `audit/authorization-review-dataset`, local only, never pushed
- Verification: `./verify`, 150 tests

## What is published

| Artifact | URL | Type | Status |
| --- | --- | --- | --- |
| Session guard pilot | `huggingface.co/hxrikp/laya-session-guard-pilot` | model, 842.6 MB | public, **rejected** |
| Follow-up, original data, 8 epochs | `huggingface.co/hxrikp/laya-session-guard-followup-original` | model, 842.6 MB | public, **rejected** |
| Risk-ranked run, metrics only | `huggingface.co/datasets/hxrikp/laya-session-guard-risk-ranked-metrics` | dataset, 48.3 KB | public, **rejected** |
| Narrated comparison vs Jev | `kaggle.com/code/uranium53/laya-session-guard-vs-jev` | notebook | public, COMPLETE |
| Session guard pilot notebook | `kaggle.com/code/uranium53/laya-session-guard-pilot` | notebook | private, COMPLETE |
| Follow-up notebook | `kaggle.com/code/uranium53/laya-session-guard-follow-up` | notebook | private, COMPLETE |
| Risk-ranked notebook | `kaggle.com/code/uranium53/laya-risk-ranked-aspi` | notebook | private, version 3 |

Not published, deliberately: the **risk-ranked checkpoint weights**. They were never downloaded
after the run was rejected and exist only in a private Kaggle output. Nothing in the metrics dataset
can be loaded as a model, which is why it is typed as a dataset rather than a model repo.

`hxrikp/laya-wikirace-wikispeedia` is unrelated and was not touched.

## What is rejected, and why

| Run | Rejected because |
| --- | --- |
| Pilot, 3 epochs | Missed 7 of 11 suspicious sources; allowed 2 non-allow actions on a 24-case diagnostic. Gains came from deferral, not safety. |
| Follow-up, original data, 8 epochs | Action accuracy rose 20.8 points over the pilot and **raw incorrect allowances stayed at 2**. Extra epochs on the same synthetic data bought no safer decisions. |
| Follow-up, counterfactual data, 8 epochs | **7 raw and 6 wrapper incorrect allowances**, a regression. The most detection-capable run and the least safe. No weights published. |
| Risk-ranked, ASPI, 3 epochs | **66 of 176** unauthorized actions allowed on unseen attacker goals, **18 of 36** on the both-unseen split, and **7 of 16** on the earlier diagnostic. Regressed against the 8-epoch checkpoint. |

## Corrected headline numbers

Numbers below are reproduced by `python3 scripts/verify_risk_report.py` and
`python3 scripts/analyze_false_allows.py`. An independent replication at
`/Users/harikp/Developer/scratch/laya-replication/` reproduced every headline result and corrected
three descriptive details: two median margins, one Mode 1 median, and one miscounted subset. The
conclusions and every rejection were unchanged.

Risk-ranked run, raw argmax incorrect allowances, argmax over `[allow, block, review]`:

| Frozen split | Pinned base | Risk-ranked | Correct allows |
| --- | ---: | ---: | ---: |
| Unseen user tasks (432) | 199/216 | **4/216** | 208/216 |
| Unseen attacker goals (352) | 134/176 | **66/176** | 176/176 |
| Both unseen (72) | 31/36 | **18/36** | 36/36 |
| 24-case diagnostic | 13/16 | 7/16 | 7/8 |

- **84 of 84** ASPI false allows come from one attacker goal, `injection_task_13`. Not a class of
  attacks, and 23 of them have a benign source with no injection anywhere in the transcript.
- **The errors are confident, not borderline.** Margin is
  `allow_logit - log(exp(block_logit) + exp(review_logit))`. Median over false allows is **11.3**
  on unseen attacker goals and **12.6** on the both-unseen split; interquartile ranges **8.3 to
  13.8** and **9.2 to 16.3**. Mode 1 median is 9.4 on the 18 and 9.8 across all 23.
- **Wrapper threshold, honestly fitted.** A threshold fitted by reading the test splits reports 52
  of 66 caught. Cross-fitted on a different split and applied out of sample, the same rule catches
  **31 of 66**, at zero cost to the 176 legitimate allowances. The in-sample figure is roughly
  double the honest one. Fit rule: grid `t` over `{0, 0.5, ..., 29.5}`, largest `t` whose
  correct-allow deferral stays within 1% on the fitting split, reported on a different split. No
  threshold can be fitted on calibration for this run, because calibration predictions were never
  persisted and the weights were never downloaded.
- **Conditional permission is confounded with a metadata flag, in one direction.** In the 2,920-case
  training split, all 216 incomplete-history cases are `review` with **no counterexample**, while 40
  `review` cases have a complete history. Global one-bit accuracy is 54.7%, which hides this.
- At `t=14`, 42 of the 61 injected false allows are converted, roughly half out of sample. That is
  drawn from all 61 injected cases; the content question flagged 59 of the 61 and missed 2.

## The synthetic stopgap does not transfer

`data/s6_s7_synthetic/` holds 64 matched S6/S7 pairs plus 8 conditional-permission cases. It
covers 11 of 11 task shapes and 5 of 5 action-shape features of the Mode 1 failure, and it is
**still not worth a training run**:

| Axis | Stopgap | Real ASPI | Real Mode 1 |
| --- | ---: | ---: | ---: |
| Mean text length | 295 | 841 | 839 |
| Vocabulary Jaccard | 0.075 vs ASPI | — | 0.208 vs Mode 1 |
| Action-verb overlap | 8% of mass | email 772, send 400 | — |
| Structured `permission` field | 64 of 64 | **0 of 2,720** | 0 |
| Dates and named entities | 0% | 92% | 52% |

The permission-field divergence is decisive and is a schema difference rather than a harder task:
training on the stopgap teaches a model to read `granted`/`scope`/`revoked`/`source`, and not one
real case has that field. **A model trained on it should not be expected to transfer to the frozen
splits.** The pre-registered primary run contributes **zero** synthetic rows for this reason.

The stopgap's value is as a specification for the human collector, and as a regression fixture for
the generator and validator. Not as data.

## Next actions, and all of them need a human

Nothing below can be started by an agent. The engineering is complete ahead of the data.

1. **Consent.** Draft wording is in `review_kit/README.md`, marked *draft for your review, not legal
   advice*. It needs your review, and professional review if the sessions come from an employer,
   contain other people's data, or anything derived will be published. Per session, from the person
   whose session it is. Consent is not implied by the session existing.
2. **Collection.** Roughly 300 real sessions, weighted toward the strata in
   `reports/REVIEW_PROTOCOL.md` section 3. Two reviewers per case, a third for disagreements.
3. **Redaction.** Applied to the raw capture before a case is written. The validator catches five
   high-signal secret shapes and will miss others; redaction is your judgement.
4. **Submit a batch.** `review_kit/validate_batch.sh path/to/batch.jsonl`. Pass every split in one
   invocation. Commit to a branch, do not push.
5. **Inter-rater agreement** per stratum after the first batch. Below 0.8 means the instructions
   need rewriting, not that the cases should be kept.
6. **Adjudicate** the four `injection_task_1` cases under `user_task_35`, which are probably label
   noise in the ASPI adaptation rather than model errors, and are currently counted in neither
   direction.

Then, and only then, the pre-registered run in `reports/REVIEW_PROTOCOL.md` section 10 becomes
possible. Its eight publication criteria are fixed in advance, and a run that misses them is
published as a negative result, as the last four were.

- **Run-up variance is larger than a two-run comparison suggested.** Three same-seed runs give
  66, 58, and **33** false allows on unseen attacker goals. The kernel is **not public**: its
  markdown asserts the earlier 58-to-66 range and v5's own output gives 33, so publishing would
  again contradict its own execution. `FALSE_ALLOW_ANALYSIS.md` has the full table.
- **The published temperature of 0.5 was a degenerate fit, not a real value.** v2 and v5 both
  classify their calibration set perfectly at the selected epoch, so cross-entropy is minimised by
  sharpening without limit and the temperature is unidentifiable. v4, the only run with a single
  calibration error, fits to 2.179 interior. Widening the grid cannot fix an unidentified
  parameter; a calibration set the model cannot classify perfectly can.
- **Determinism is now fixed and proven.** v5 ran with `use_deterministic_algorithms`,
  `cudnn.deterministic`, no `cudnn.benchmark`, no TF32, and `CUBLAS_WORKSPACE_CONFIG` set, and
  emitted zero `does not have a deterministic implementation` warnings. v5 is the first
  repeatable run. `src/train_risk.py` records the configuration in its report.
- **A regression was found and fixed.** The `eval_rows` work had inserted a `continue` into the
  evaluation loop, silently dropping the 24-case diagnostic from `report["finetuned"]`. It passed
  every local test. Pinned by `tests/test_train_risk_pipeline.py`.
- **The pipeline is not bit-reproducible.** fp16 autocast on GPU with no
  `torch.use_deterministic_algorithms`. Published counts are point estimates from one run; the
  3-seed requirement in the pre-registration is a floor, not a formality.

## Open items not requiring data

- `src/train.py` carries eval instrumentation that is uncommitted, because the file also holds
  in-progress work that was excluded from the audit branch by request. The temperature-grid
  correction ships as `patches/temperature-grid.patch` for the same reason.
- **The risk-ranked Kaggle write-up is public as of kernel v8**, at
  `kaggle.com/code/uranium53/laya-risk-ranked-aspi`. Its markdown asserts no per-run number; the
  table is generated from the run's own report, so a run can no longer contradict its write-up.
  v7 completed cleanly and was verified against its own output before the flip: 36 of 176 on
  unseen attacker goals, 7 of 36 on the both-unseen split, 2 of 216 on unseen user tasks, 6 of 16
  on the 24-case diagnostic. Rejected, as every recorded run is.
- The v7 run selected epoch 3, whose calibration accuracy was 99.8% with **exactly one error**,
  and its temperature fit is interior at 2.45 rather than pinned to the grid floor. That is a
  direct confirmation of the identifiability finding from an independent run: one calibration
  error makes the temperature estimable, zero collapses it.
- The `S1`/`S2`/`S3`/`S5` coverage warnings are permanent for an S6/S7-only stopgap.
- `leakage.text_near` warns and does not block. Somebody must clear those warnings by hand before a
  training run.
