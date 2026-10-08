# How a condition is written predicts whether agents skip it (agent `conditions`, round 3, 5 Oct 2026)

Saved by the main session from the agent's final message. SOPBench only (no four-scenario pooled number). 405
gemini-3.8-flash calls (one per distinct policy text). Code: parse.py, tag_llm.py, feats.py, targets.py, cm.py,
cond_quality.py, diag.py, run.py, run_diag.py. Data: conds.csv, tags.json, cond_feats.csv, cond_outcomes.csv.gz.
Results: out/eval.json, out/extra_metrics.json, out_diag/, cond_quality.json, diag.json.

## Idea and levels
Learn on training procedures which written conditions agents skip (text features only), predict for new procedures,
combine into case P(success). L0: policy text and rubric. L2: plus the compiled program's decision and the last tool it
called (separate track).

## Setup
Regex segmentation: 1,539 conditions in 411 policy texts, aligned positionally with the rule tree (training targets only).
49 text features: position, length, named fields or tools, negation, emphasis, OR and nesting, numeric thresholds,
comparison words, date words and arithmetic, cross references, shared fields, 8 LLM tags (cross_ref, date_arith, calc,
n_lookups, n_comparisons, conditional, vague, precondition_style). Targets: skip rate = 1 - step_correct over 25
unflagged agents on training cases. skip_perf (must-perform), skip_fail (blocking step in must-refuse). Ridge, alpha by
grouped CV inside training, weighted by run count.
L0: P = pp * prod(1 - skip_perf) + (1 - pp) * mean(1 - skip_fail), pp = 1 - rubric p_refusal, per-agent logit offset.
L2: if the program calls the action, P = prod(1 - skip_perf), else mean (1 - skip_fail) over conditions using the last tool
the program called (decision accuracy 0.829, 249 of 830 cases fall back to all conditions).
Deviations: alpha grid extended after one fold hit the top, controls added, all inspected on one new_domain fold (bank),
so not a clean pre-registration. Plain pre-check had a sign bug in its first run, rerun correctly.

## Condition-level held-out quality (Spearman, predicted vs observed skip rate)

| scheme | target | text model | step-type null | weighted R2 text / type |
|---|---|---|---|---|
| new procedures | skip_perf | 0.63 | 0.42 | 0.40 / 0.28 |
| new procedures | skip_fail | 0.61 | 0.50 | 0.38 / 0.27 |
| new domain | skip_perf | 0.60 | 0.32 | 0.28 / 0.20 |
| new domain | skip_fail | 0.45 | 0.42 | 0.31 / 0.25 |

49% of perf and 65% of fail conditions are novel text after normalising numbers and names. On novel text 0.39 (perf) and
0.45 (fail), on text seen in training 0.69 and 0.76.

## Case level, SOPBench (protocol rho, paired CI vs baseline adele)

| column | new proc | new dom | mean | CI vs baseline (proc / dom) | Brier skill |
|---|---|---|---|---|---|
| adele | 0.120 | 0.125 | 0.123 | | 0.014 / 0.005 |
| l0_alone | 0.141 | 0.178 | 0.160 | [-0.115,+0.234] / [-0.038,+0.132] | 0.028 / 0.018 |
| l0_adele | 0.155 | 0.195 | 0.175 | [-0.020,+0.097] / [+0.018,+0.154] | 0.010 / 0.015 |
| l0_agent | 0.180 | 0.193 | 0.186 | [-0.042,+0.278] / [-0.028,+0.162] | 0.039 / 0.015 |
| l0_alone + adele | 0.174 | 0.210 | 0.192 | [-0.067,+0.211] / [+0.034,+0.146] | 0.047 / 0.047 |
| l0_const (structure null) | 0.049 | 0.083 | 0.066 | | |
| l0_prior (no rubric) | 0.158 | 0.166 | 0.162 | | |
| l2_alone | 0.396 | 0.348 | 0.372 | [+0.072,+0.502] / [+0.073,+0.403] | 0.081 / 0.051 |
| l2_adele | 0.301 | 0.308 | 0.305 | [+0.036,+0.322] / [+0.088,+0.334] | |
| l2_agent | 0.394 | 0.355 | 0.375 | [+0.091,+0.505] / [+0.074,+0.422] | 0.089 / 0.055 |
| l2_alone + adele | 0.384 | 0.344 | 0.364 | [+0.072,+0.449] / [+0.102,+0.380] | 0.075 / 0.071 |
| l2_const | 0.215 | 0.245 | 0.230 | | |
| l2_plain (plain pre-check) | 0.273 | 0.282 | 0.278 | | |
| l2_truth (privileged, diagnostic) | 0.581 | 0.507 | 0.544 | | |
| label-only oracle | 0.511 | 0.506 | 0.509 | | |

l0_alone - l0_const: [+0.03, +0.149] / [+0.02, +0.227]. l2_alone - l2_plain: [-0.020, +0.298] / [-0.055, +0.222].

Extra: rho_given_label l0_alone 0.149 / 0.201, l2_alone 0.325 / 0.297 (plain pre-check 0.068 / 0.054, baseline 0.130 /
0.142). Procedure-level new domain: l0_alone 0.236, l2_alone 0.261, l2_alone+adele 0.306, baseline 0.231. Selective lift
l2_alone 0.094 / 0.078 vs baseline 0.046 / 0.026.

## Which written features predict skipping
Skipped more: more distinct quoted fields (+0.52 / +0.54), more lookups (LLM n_lookups +0.47 / +0.56), tool-name overlap
(+0.49), fields shared with other conditions (+0.45 / +0.49), length (+0.43), date words and date arithmetic (fail, +0.44
to +0.46). Skipped less: generic login or authentication wording (-0.57 / -0.41), precondition style (-0.45 / -0.44),
cross-reference words (-0.45 / -0.26), first condition (-0.17). Within a step type: earlier position skipped less (-0.30),
thresholds less (perf -0.27), fields, length and lookups more (+0.25 to +0.33).

## Verdict
1. L0: the written-condition model beats a structure-only null significantly, the baseline only weakly. Averaged with the
   baseline it helps on new domain (about +0.09, CI above 0). The rubric adds nothing over a training prior.
2. L2: 0.37 vs baseline 0.12 and plain pre-check 0.28, CI vs baseline above 0 in both scenarios, gain over the plain
   pre-check not significant. Text-driven skip rates add within-label signal the plain feature lacks.
3. Condition-level transfer is real but partly template reuse (about 0.4 to 0.45 on novel text).

## Limitations
SOPBench only, about 70 procedures, inspected on one fold before the main run. step_correct proxy. L2 failing-condition
matching ambiguous for 35% of refusals. Two SOPBench scenarios, not the four-scenario pooled value.
