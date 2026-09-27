# Unblocking Mode 1: where consented cases can come from

A decision memo, not a plan of record. Four realistic routes to the missing data, what each
costs, and what each cannot support. Written before doing any of them.

## What Mode 1 actually is

23 false allows in the rejected run, all from one attacker goal: expand a one-recipient request
into a broadcast to lists the user never named, one scripted message per recipient, on a
transcript with **no injection anywhere**. The rejected checkpoint allowed 66 of 176 unauthorized
actions on unseen attacker goals. The synthetic stopgap covers the shape and was measured as
non-transferable (`reports/FALSE_ALLOW_ANALYSIS.md` §10.7). We need a hundred or so cases where a
real person was actually asked for one recipient and an agent actually proposed many.

The awkward part: the current model, run through the scenario harness, **declined to expand in all
five scenarios and respected the revoked grant**. The behaviour being corrected does not reproduce
by prompting this model. It reproduces in the rejected fine-tune and in the synthetic training
data. So "just collect more agent traces from the model in use" will not produce the failure.

## Option A: real cases from consenting people

Recruit a handful of people whose agents can send mail, calendar invites, or messages. Ask each
to donate sessions where an outward send was proposed. Two human reviewers label each.

- **Cost:** your time to recruit and review, plus theirs. Realistically a few weekends, and the
  review burden is two people per case, so ~200 cases means ~400 labelling decisions.
- **Honesty:** the highest available. These are witnessed decisions by a real subject about a
  real request. If the subject agrees to publication, the consent chain is intact.
- **Cannot support:** a *rate*. The subjects are self-selected, they know they are donating
  authorization cases, and their agents may be more cautious than average. This measures that the
  shape exists and is learnable, not how often it happens.
- **Cannot support:** generality across tooling. Email-capable agents are a narrow slice.

**This is the only option that yields witness data.** If the project is going to make any claim
stronger than "we fixed a shape we scripted", it starts here.

## Option B: run the rejected checkpoint in the scenario harness

Load `laya-session-guard-riskanked` as the acting agent behind the mock `send_batch` tool and run
the same five scenarios. If the fine-tune is what produces the expansion, this is the direct test.

- **Cost:** nearly zero. The checkpoint is a 421M model, the harness exists, five scenarios run in
  seconds on CPU. The weights are in the private Kaggle output and would need downloading first.
- **Honesty:** this is the cleanest experiment on this list and it is worth doing regardless. It
  tests a specific causal claim: *the fine-tune, not the base model, learned to expand recipients.*
  A positive result is strong evidence. A negative result is equally informative and would send
  you back to the data.
- **Cannot support:** training on the output. **Model-generated data is not witness data.** A case
  where the rejected checkpoint expanded recipients was, by construction, not a mistake the system
  caught; it is a mistake the system made, and training on its own errors without a human judge
  teaches nothing about what a human would have said. It is usable as an evaluation probe and as
  a regression fixture, and for nothing else.
- **Cannot support:** any statistic about frequency or severity.

Do this first, even though it does not produce training data. It is the cheapest experiment on
this list and it is diagnostic.

## Option C: scripted rehearsals as regression fixtures

The five scenarios in `review_kit/scenarios/`, with the tool call scripted.

- **Cost:** zero. Already built and validated.
- **Honesty:** none as evidence. The user turn and the tool call are both written by us. Nothing
  here is a real request or a real decision.
- **Cannot support:** anything measured. **Never use as training data.** The failure mode is
  specific and worth stating: a template-generated pair set teaches the template, which is exactly
  the same-generator confound the S6/S7 stopgap was rejected for. Its only legitimate uses are as a
  pipeline regression test and as the shape specification a human collector works from.

## Option D: public licensed agent-trace datasets

Look for an existing corpus of agent traces that contain permission decisions.

- **Status: unverified.** The web search tool is out of credits, so I could not search. I am not
  going to assert a dataset exists on the strength of recollection.
- **What is already in use:** `ScaleAI/aspi`, CC BY 4.0, 407 workspace records with a legitimate
  task, a safe action, an attacker goal, and clarifications. That is where the 1,864 adapted
  training examples came from. Its labels are **benchmark-generated, not human-reviewed for our
  schema**, which is exactly the gap.
- **What to check if you pursue it:** whether the licence permits derived training data, whether
  any human actually made a permission judgement rather than a scripted attacker, and whether the
  corpus is model-generated. Most public agent-trace corpora are model-generated, which lands them
  in Option B's honesty bucket, not Option A's.
- **Cannot support:** witness data, for the same reason as B, unless the corpus turns out to be
  genuinely human-labelled.

## Recommendation

**Option B first, because it is nearly free and it tests the causal claim the whole project rests
on.** If the rejected checkpoint expands recipients in the harness and the base model does not,
the fine-tune is confirmed as the cause and the remaining problem is data volume, not data shape.
If both decline, the expansion is something the synthetic training data taught that no natural
model does, and the pre-registered run should be reconsidered rather than repeated.

**Option A in parallel, because it is the only thing that can ever produce witness data** and
because it is slow. Start the recruitment conversation now while B runs. A target of 100 to 200
cases, weighted toward S6 and S7, with the S7 control cases collected first so a "fix" cannot be
just blocking everything.

**Option C stays a fixture.** It is already built and it is genuinely useful for keeping the
collector and validator honest. It must never enter training.

**Option D only after someone verifies it**, with the three checks above.

## The thing none of these can fix

Every option leaves the same gap. Even 200 real cases tell us the shape is learnable, not that
the resulting model is safe to deploy. A guard that is 99% right on 200 cases still blocks a
legitimate action in one of them, and that trade is a product decision, not a data decision. The
pre-registered publication criteria in `reports/REVIEW_PROTOCOL.md` §10.6 exist so that the
number decides, and the task-level policy check stays outside the model regardless.

## One dependency I will not paper over

Option A cannot start until a consent record exists for a real session. The draft wording is in
`review_kit/README.md`, marked **draft for your review, not legal advice**. If the sessions come
from an employer, contain other people's data, or anything derived will be published, get it
reviewed by someone qualified. I am not able to tell you that the draft is sufficient, and I would
not use it as though it were.
