# tau2 planner with the real tool list (agent `plan_tools`, round 3, 5 Oct 2026)

Saved by the main session from the agent's final message. Level L1. 507 gemini-3.8-flash calls (thinking medium).
tau2-bench cloned at commit 5bfa7e37b36656b37dc6d022156be6563c1007f3 (MIT), agent tool schemas exported with
get_environment().get_tools(): airline 14, retail 16, telecom 13. Telecom user tools (30) are not visible to the agent,
replaced by one pseudo-tool ask_customer_to_act. 169 distinct prompts x 3 plans (t0 plus 2 at temperature 1), 0% invented
tool names. SOPBench features reused from plan_tau (features/plan3_sopbench.csv is a copy).
Code: dump_tools.py, build_tools.py, gen2.py, feats3.py, run3.py, validate3.py, perdom3.py. Results: out/run/eval.json,
eval.txt, extra_metrics.json, validation3.json.

## Main table (pooled within-domain rho; sop-proc, sop-dom, tau2-proc, tau2-dom)

| column | pooled | 95% CI | worst | scenarios | paired CI vs baseline |
|---|---|---|---|---|---|
| adele (baseline) | 0.131 | [0.027, 0.255] | 0.120 | .120 .125 .128 .150 | |
| plan3 | 0.202 | [0.118, 0.336] | 0.154 | .154 .176 .291 .187 | [-0.041, +0.222] |
| plan3sd | 0.183 | [0.102, 0.319] | 0.132 | .154 .176 .272 .132 | [-0.067, +0.216] |
| plan3_cat_adele | 0.214 | [0.115, 0.355] | 0.179 | .213 .186 .279 .179 | [+0.021, +0.165] |
| plan3_cat_adele+adele | 0.181 | [0.089, 0.296] | 0.163 | .180 .170 .212 .163 | [+0.001, +0.112] |

tau2 paired CIs: plan3 new_proc [-0.045, +0.527], new_dom [-0.205, +0.361]; plan3_cat_adele new_proc [+0.032, +0.362],
new_dom [-0.132, +0.213]. Brier and log-loss skill near zero or negative on tau2 (plan3 new_dom -0.096): rank features,
not calibrated probabilities.

tau2 rho new_proc / new_dom: baseline 0.128 / 0.150, plan_tau plan 0.147 / 0.181, plan_tau plan_cat_adele 0.184 / 0.117,
plan3 0.291 / 0.187, plan3_cat_adele 0.279 / 0.179, gold oracle 0.470 / 0.401. About half of the gold gap closed on new
procedures, almost no gain on new domain.

## Validation against gold (tau2)

| domain | calls vs gold agent len | writes vs gold writes | distinct tools | Jaccard of tool names | plan calls / gold agent len |
|---|---|---|---|---|---|
| airline | 0.23 | 0.62 | 0.28 | 0.50 | 3.84 / 2.84 |
| retail | 0.17 | 0.82 | 0.19 | 0.57 | 5.85 / 4.82 |
| telecom | 0.15 (0.44 with user actions) | 0.48 | 0.19 | 0.13 | 5.18 / 1.08 |

plan_calls vs b: airline 0.48 (gold 0.51), retail 0.30 (gold 0.07), telecom 0.28 (gold 0.36), telecom plan_writes vs b 0.55.
The planner over-plans reads by 1 to 2 calls.

Per domain (new_proc / new_dom): adele airline 0.33 / 0.37, retail 0.26 / 0.15, telecom -0.45 / 0.03; plan3 0.47 / 0.35,
0.23 / 0.22, 0.13 / 0.16; plan3_cat_adele 0.48 / 0.44, 0.34 / 0.28, -0.08 / 0.04.
Stability of plan_calls (t0 vs 3-plan mean): 0.98 airline, 0.95 retail, 0.89 telecom. Spread features did not help.

## Verdict
1. The real tool list lifts tau2 new_proc (0.15 to 0.29) but not new_dom (0.18), about half of the gold oracle.
2. plan3_cat_adele is the only variant significantly above the baseline pooled (+0.021 to +0.165, lower bound barely positive).
3. Modest evidence: 50 to 164 tasks per tau2 scenario in 3 folds, result rests on a stack with ADeLe.
