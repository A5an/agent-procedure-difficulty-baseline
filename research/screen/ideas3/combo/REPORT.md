# Combined cheap predictor, pre-registered (agent `combo`, round 3, 5 Oct 2026)

Saved by the main session from the agent's final message. No LLM calls, no L2 features. PREREG.md written before any
evaluation. Files: PREREG.md, run.py, eval_main.py, score_combo.py, out_main/ (eval.json, eval.txt, extra_metrics.json),
out_top5/, score_main.txt, score_top5.txt, logs.

## Families
ADELE (18 demand levels, L0, both), RUB (12 procedure-rubric columns, L0, both), COND (logit of the written-condition
skip score, L0, SOPBench, refitted per fold), CMP (len_mean, branch_mean of compiled programs, L0, SOPBench), PLAN (9
planned-action features, real tool list on tau2, L1, both). Grouped ridge, one penalty per family from {1, 10, 100, 1000,
10000} by 5-fold inner CV grouped by procedure.
Variants: c_L0 (ADELE + RUB + COND + CMP, tau2 only ADELE + RUB), c_L1 (c_L0 + PLAN), c_lad (rubric ladder L2 with global
columns ADELE + PLAN + CMP, tau2 ADELE + PLAN, no COND).

## Main target (pooled within-domain rho; sop-proc, sop-dom, tau2-proc, tau2-dom)

| column | pooled | 95% CI | worst | scenarios | paired CI vs adele | vs lad2_adele* |
|---|---|---|---|---|---|---|
| adele | 0.131 | [0.027, 0.255] | 0.120 | .120 .125 .128 .150 | | -0.051 [-0.082, +0.016] |
| lad2_adele | 0.182 | | | .181 .179 .168 .199 | +0.051 [-0.016, +0.082] | |
| c_L0 | 0.159 | [0.046, 0.277] | 0.085 | .216 .225 .085 .110 | [-0.046, +0.077] | -0.023 [-0.038, +0.024] |
| c_L1 | 0.196 | [0.114, 0.312] | 0.121 | .223 .228 .211 .121 | [-0.020, +0.168] | +0.014 [-0.008, +0.091] |
| c_lad | 0.258 | [0.148, 0.362] | 0.212 | .237 .212 .298 .285 | [+0.010, +0.198] | +0.076 [+0.037, +0.128] |
| c_L1 + adele | 0.194 | [0.118, 0.332] | 0.110 | .217 .230 .220 .110 | [-0.003, +0.160] | |
| c_lad + adele | 0.209 | [0.119, 0.322] | 0.201 | .206 .205 .225 .201 | [+0.013, +0.144] | |

*Bootstrap over procedures from tiers/score.py (300 draws), not the harness bootstrap, not a criterion.
Per scenario paired CI vs adele, c_lad: [-0.025, +0.271], [+0.019, +0.181], [-0.065, +0.357], [-0.066, +0.302].
Brier / log-loss skill c_lad: .081/.079, .081/.087, .032/.055, .013/.060 (adele .014/.011, .005/.003, .003/-.000, -.021/-.028).
Procedure level SOPBench new domain: c_L0 0.450, c_L1 0.442, c_lad 0.486 [+0.025, +0.462] vs adele 0.231, lad2_adele 0.411.
rho_given_label c_lad 0.253 / 0.218. Selective lift c_lad 0.054 vs 0.033.

## Decision rule (EVALUATION_PROTOCOL section 8)

| variant | 1 pooled CI above 0 | 2 no scenario significantly worse | 3 no benchmark worse | 4 Brier and log-loss skill not lower |
|---|---|---|---|---|
| c_L0 | FAIL | pass | pass | FAIL (tau2-dom) |
| c_L1 | FAIL | pass | pass | FAIL (tau2) |
| c_lad | pass (+0.127 [+0.010, +0.198]) | pass | pass | pass |

Criterion 5 (blind test) open.

## Secondary: IRT on the top-5 tier, scored on fail_top5
adele 0.113, lad2_adele 0.187, c_L0 0.185, c_L1 0.170, c_lad 0.171 (+0.057 [+0.007, +0.105] vs refit adele, -0.016
[-0.043, +0.030] vs refit lad2_adele). Procedure level c_lad 0.488, refit adele 0.047. No combo beats refit lad2_adele.

## Diagnostics
tau2: RUB inside a grouped ridge hurts (c_L0 0.085 / 0.110), inside the ladder helps (c_lad 0.298 / 0.285). SOPBench
penalties: COND got the strongest penalty in 25 of 32 fits, PLAN in 18 of 32, CMP weak penalty. c_lad gain over lad2_adele:
tau2 new procedures +0.130 [+0.048, +0.200], sop-proc +0.056 [+0.029, +0.091], sop-dom +0.034 [+0.012, +0.073], tau2 new
domain +0.086 [-0.057, +0.225]. Gain not attributed to a single family (no ablation after results).

## Verdict
1. c_lad alone passes criteria 1 to 4 on the main target (0.258, +0.127 [+0.010, +0.198]) and criteria 1 to 3 on the
   top-tier target. Gain over lad2_adele +0.076 [+0.037, +0.128] on the main target, vanishes on the top-tier target.
2. Pure documentation (c_L0) fails: helps on SOPBench, flat or negative on tau2, fails criterion 4.
3. Confidence moderate to low: lower bound +0.010, three of four per-scenario CIs include 0.

## Limitations
COND missing from c_lad. Different bootstrap for the lad2_adele comparison. One seed per scheme in the obs files.
Secondary criterion 4 not computed. Features chosen after seeing round-3 results, so the blind test is essential.
