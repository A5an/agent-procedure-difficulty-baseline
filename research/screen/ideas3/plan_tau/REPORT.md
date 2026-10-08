# Planned-action features from documentation (agent `plan_tau`, round 3, 5 Oct 2026)

Saved by the main session from the agent's final message. Information level L1 (one LLM call per case, inputs: policy,
scenario, tool list where available). About 1,215 gemini-3.8-flash calls.
Code: gen.py, feats.py, validate.py, gold.py, run.py, perdom.py. Results: out/run/eval.json, eval.txt, extra_metrics.json,
out/gold/eval.json, validation.json, stability.json, perdomain.json. Plans: plans_t0.json, plans_t1.json.

## Setup
One call per case (temperature 0, thinking low, JSON): ordered tool plan with a writes_db flag, policy rules relied on,
facts to ask, outcome (complete, refuse, transfer), ambiguity 0 to 2. tau2 gets the domain policy (telecom: main policy plus
tech-support manual), SOPBench its statement. 9 features: plan_calls, plan_writes, plan_reads, plan_distinct_tools,
plan_rules, plan_facts, plan_refuse, plan_transfer, plan_ambig. Two temperature-1 samples on 100 tasks for stability.
Variants fixed before the run: plan, plan_cat_adele, plan_grp_adele, averages with the baseline. Oracle (gold, tau2 only):
gold_len, gold_cat_adele. Deviation: tau2 prompts had no tool schema (only recoverable from gold), tool names invented.

## Main table (pooled within-domain rho; scenarios sop-proc, sop-dom, tau2-proc, tau2-dom)

| column | pooled | 95% CI | worst | scenarios | paired CI vs baseline |
|---|---|---|---|---|---|
| adele (baseline) | 0.131 | [0.027, 0.255] | 0.120 | .120 .125 .128 .150 | |
| plan | 0.165 | [0.078, 0.333] | 0.147 | .154 .176 .147 .181 | [-0.035, +0.177] |
| plan_cat_adele | 0.175 | [0.073, 0.352] | 0.117 | .213 .186 .184 .117 | [+0.019, +0.151] |
| plan_grp_adele | 0.161 | [0.078, 0.318] | 0.081 | .217 .198 .149 .081 | [-0.008, +0.125] |
| plan_cat_adele+adele | 0.161 | [0.056, 0.302] | 0.147 | .180 .170 .147 .147 | [+0.006, +0.087] |

tau2 scenario paired CIs: plan new_proc [-0.075, +0.316], new_dom [-0.223, +0.355]; plan_cat_adele new_proc
[+0.014, +0.258], new_dom [-0.178, +0.156]. SOPBench plan_cat_adele new_proc [-0.027, +0.208], new_dom [-0.002, +0.147].
Brier and log-loss skill near zero for every column.

| column | procedure-level rho SOPBench new dom | paired CI | rho_given_label (proc / dom) | hard recall 20% |
|---|---|---|---|---|
| adele | 0.231 | | 0.130 / 0.142 | 0.281 |
| lad2_adele | 0.411 | [-0.008, +0.370] | 0.190 / 0.173 | 0.333 |
| plan | 0.141 | [-0.440, +0.299] | 0.205 / 0.203 | 0.214 |
| plan_cat_adele | 0.313 | [-0.084, +0.271] | 0.236 / 0.197 | 0.256 |
| plan_grp_adele | 0.346 | [-0.111, +0.332] | 0.222 / 0.182 | |
| plan_cat_adele+adele | 0.353 | [+0.015, +0.256] | 0.200 / 0.190 | |

## Validation against gold (tau2)
Planned calls vs gold length: airline 0.27, retail 0.11. Planned writes vs gold writes: airline 0.52, retail 0.73.
Correlation with IRT b: gold_len airline 0.51 retail 0.07, plan_calls 0.45 / 0.32, plan_writes 0.37 / 0.29,
plan_facts 0.50 / 0.16.

Gold oracle (tau2, labelled as using gold): gold_len new_proc 0.470 [0.156, 0.608], new_dom 0.401 [0.082, 0.605],
gold_cat_adele 0.466 / 0.436. Plan features reach only 0.147 to 0.184 on tau2.

## Diagnostics
Per tau2 domain (new_domain): plan airline 0.49 vs adele 0.37, retail 0.15 vs 0.15, telecom 0.06 vs 0.03. new_procedures
plan_grp_adele airline 0.45, retail 0.34 vs 0.33 and 0.26. Telecom: 5 distinct texts, difficulty sits in the hidden fault
(gold length vs b 0.69 there). Stability: SOPBench very stable (rank vs sample mean 0.97), tau2 less (0.75 calls, 0.63
writes). Instability vs difficulty: 0.05 SOPBench, 0.13 tau2. SOPBench plan_refuse vs label AUC 0.556. plan_reads best
single SOPBench feature (0.25 pooled, 0.14 within domain).

## Verdict
1. Plan features give pooled 0.16 to 0.18 vs 0.131, plan_cat_adele just above zero on the paired CI ([+0.019, +0.151]).
   The gain comes mostly from SOPBench (rho_given_label 0.20 to 0.24 vs 0.14), not tau2 as hypothesised.
2. On tau2 the plan is a weak copy of gold length: matches gold writes in airline and retail, not call count, useless in
   telecom. The gold oracle reaches 0.40 to 0.47, so most of the length signal is not recovered.
3. Modest evidence: one variant with lower bound +0.019, per-scenario intervals include 0.

## Limitations
No real tau2 tool schema in the prompt. Telecom hidden fault invisible. Instability measured on 100 tasks only. Gold write
regex approximate.
