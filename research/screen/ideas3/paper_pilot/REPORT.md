# Paper pilot: one-call plans without tools as a cheap probe (agent `paper_pilot`, round 3, 5 Oct 2026)

Saved by the main session from the agent's final message. Level L1. About 2,990 calls (gemini-2.5-flash-lite,
gemini-3.8-flash, gemini-2.5-flash for the model-specific check), 992 unique prompts per model.
Files: pp_lib.py, collect.py, build_features.py, features/, plans_parsed.json, run_screen.py, out/run/ (eval.txt,
eval.json, extra_metrics.json), dump_real.py, real_h0.json, analyze_model.py, model_specific.json, diagnostics.json.

## Setup
SOPBench prompt: statement with the domain tool list from task_initializer. tau2: domain policy, scenario, tool list built
from tool names in trajectories. Output: tool_calls with arguments, final_decision, refusal_conditions. Required checks per
SOPBench case from the compiled program (required_tools, same rule as harness/run.py). Features: plan shape, missing
required checks, checks after the action, decision, refusal path, model disagreement. Variants fixed before the run.

## Main table (pooled within-domain rho; sop-proc, sop-dom, tau2-proc, tau2-dom)

| variant | pooled [95% CI] | scenarios | paired CI vs baseline | proc-rho SOP new dom | rho given label SOP |
|---|---|---|---|---|---|
| adele | 0.131 [0.027, 0.255] | .120 .125 .128 .150 | | 0.231 | 0.13 / 0.14 |
| pf_lite | 0.050 | .165 .173 -.137 -.002 | [-0.299, +0.051] | 0.142 | 0.16 / 0.18 |
| pf_lite_cat_adele | 0.144 [0.036, 0.271] | .196 .196 .084 .101 | [-0.043, +0.067] | 0.155 | 0.19 / 0.20 |
| pf_lite + adele | 0.136 | .180 .188 .073 .103 | [-0.069, +0.059] | 0.215 | 0.18 / 0.20 |
| pf_g38 | -0.062 | .076 .065 -.141 -.246 | [-0.374, +0.109] | -0.121 | 0.14 / 0.13 |
| pf_g38_cat_adele | 0.099 | .108 .114 .065 .109 | [-0.082, +0.037] | 0.127 | 0.15 / 0.17 |
| pf_both_cat_adele | 0.105 | .170 .180 -.002 .070 | [-0.089, +0.071] | 0.124 | 0.20 / 0.21 |
| pf_both_cat_adele + adele | 0.126 | .160 .167 .077 .101 | [-0.052, +0.052] | 0.158 | 0.18 / 0.20 |

No pooled gain. Small non-significant SOPBench gain (0.17 to 0.20 vs 0.12). tau2 nothing. Procedure level nothing above
0.231 (pf_g38 alone significantly worse).

## Model-specific check (own plan vs own real outcomes)

| outcome set | score | AUC [CI] | within-domain rho | rho given label |
|---|---|---|---|---|
| 2.5-flash public 830 | population baseline | 0.49 | +0.035 | +0.005 |
| | own plan, logistic | 0.50 | +0.074 | +0.082 |
| | lite plan, logistic | 0.62 [0.57, 0.65] | +0.145 [0.06, 0.22] | +0.132 |
| 2.5-flash real H0, 150 | population baseline | 0.45 | +0.089 | +0.023 |
| | own plan, fixed | 0.67 [0.58, 0.76] | +0.207 | +0.306 |
| | 3.8 plan, fixed | 0.71 | +0.294 | +0.374 |
| 3.8-flash real H0, 150 | population baseline | 0.43 | +0.022 | -0.087 |
| | own plan, logistic | 0.71 [0.60, 0.81] | +0.351 | +0.404 |
| | lite plan, fixed | 0.71 | +0.244 | +0.351 |

Own plan beats the population baseline on AUC on 150 cases, but another model's plan does just as well: the signal is
general difficulty carried by omitted checks, not model-specific. Strong models rarely omit checks (3.5 to 4% of steps).

## Diagnostics
Plans choose "perform" in 89 to 92% of SOPBench cases (true 35%), decision accuracy 0.40 to 0.42: without the database the
plan cannot identify must-refuse cases. Checks never placed after the action. A check the 3.8 plan omits is skipped in real
runs far more often (0.60 to 0.75 vs 0.07 to 0.25, phi 0.20 to 0.34), but omissions are rare.

## Verdict
1. Case level: small non-significant SOPBench signal, nothing on tau2 or at procedure level.
2. Model-specific probe not supported: any model's plan predicts a given model about equally.
3. Mechanism confirmed where omissions occur, but they are too rare in strong-model plans to rank cases.

## Limitations
tau2 tool descriptions rebuilt from trajectories. tau2 scenario text includes hidden user instructions. Required checks
from one compiled program per unit. 150-case analyses small, case bootstrap only.
