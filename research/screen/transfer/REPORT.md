# transfer: Terminal-Bench 2.0 and SWE-bench Verified screening (4 Oct 2026)

No LLM calls. Scripts: `common.py` (fit code, fixed hyper-parameters), `m1.py` (method 1), `m23.py` (methods 2 and 3).
Outputs: `m1_tb.json` (includes the full agent-to-(harness, model) mapping and the dropped agents), `m1_swe.json`, `m23.json`.
Run with the repo venv, `PYTHONHASHSEED=0`. This is not the baseline protocol (no new-process scenarios, no harness): these are
self-contained checks of three claims from Agent psychometrics, so none of the numbers is comparable to the 0.131 rho.

Data: `third_party/agent-psychometrics/data/terminalbench/responses_per_attempt.jsonl` (143 submissions x 89 tasks, successes/trials,
12,727 filled cells, 62,682 attempts; 11,376 cells have 5 trials, the rest 1 to 10) and `swebench_verified/responses.jsonl` (134 agents x 500 tasks, 1 attempt).

Fixed before looking at any held-out result (all in `common.py`): binomial likelihood, logit = mu + ability - difficulty, L2 priors
(precision 1 on abilities / model / harness effects, 0.25 on task difficulty, 4 on log-discrimination, 1 on rank-2 factors), L-BFGS,
5-fold random cell split with seed 0, smoothing pseudo-count 2 for shares. No hyper-parameter was tuned.

## Method 1. Ability decomposition theta_agent = theta_model + theta_harness (Eq. 4)

Setup. Harness = `metadata.agent`, model = `metadata.model` of the leaderboard row (model strings normalised: "Claude 4.5 Sonnet" = "Claude Sonnet 4.5",
"Claude 4.6 Opus" = "Claude Opus 4.6", the two MiniMax m2.5 spellings merged). 9 rows with model "Multiple" (Warp x3, LemonHarness x2, JJAgent, Polaris, Junie CLI
multiple, Abacus AI Desktop) are dropped as ambiguous. Left: 134 rows, 46 models, 40 harnesses. Leave-one-out unit = a (harness, model) pair (so a duplicate
submission of the same pair cannot leak). Evaluated only when both the model and the harness occur in other training rows: 101 of the held-out units
(the others are singleton models or harnesses, where the decomposition cannot be identified). For each held-out unit everything is refit on the rest,
including task difficulties. Compared on all 89 tasks of the held-out agent:
- dec: mu + theta_model + theta_harness - beta (decomposition fit)
- mean (a): mean ability of all other agents (free-theta 1PL fit)
- same (b): mean free ability of the other agents with the same model (other harnesses only)
- modonly (extra): theta_model of the decomposition, harness effect set to 0

Terminal-Bench 2.0, 101 held-out agents, attempt-level scores:

| predictor | log-loss | pooled AUC | mean abs error of success share | bias of share |
|---|---|---|---|---|
| dec (model + harness) | 0.393 | 0.902 | 0.042 | +0.001 |
| (a) mean of all agents | 0.547 | 0.800 | 0.182 | +0.016 |
| (b) same model, other harnesses | 0.396 | 0.900 | 0.048 | +0.005 |
| model effect only | 0.405 | 0.899 | 0.066 | +0.045 |

Paired log-loss difference, dec minus comparator, bootstrap over held-out agents (2000 resamples): vs (a) -0.155 [-0.191, -0.120], dec better for 87% of agents;
vs (b) -0.0034 [-0.0073, +0.0005], dec better for 55% of agents. AUC is pooled over agents because within one agent the three predictors rank tasks identically.

SWE-bench Verified (1 attempt). Only names that literally contain harness_model could be parsed (33 agents, hand mapping in `m1.py` `SWE_MAP`; "claude-3.5-sonnet" old and
the 20241022 update are kept as different models). Eligible held-out: 19.

| predictor | log-loss | AUC | MAE share | bias |
|---|---|---|---|---|
| dec | 0.342 | 0.929 | 0.079 | +0.002 |
| (a) mean | 0.557 | 0.815 | 0.229 | -0.039 |
| (b) same model | 0.336 | 0.931 | 0.059 | -0.002 |
| model only | 0.353 | 0.923 | 0.083 | 0.000 |

dec minus (a): -0.215 [-0.318, -0.125]; dec minus (b): +0.006 [-0.043, +0.051]. Small sample, wide intervals.

Verdict. Knowing the model identity transfers well to a new harness (log-loss 0.55 to 0.39, share error 0.18 to 0.05), but that gain is already obtained by (b), the
ability of the same model under other harnesses. Adding a harness term improves on (b) by only 0.003 log-loss on Terminal-Bench, with a CI that includes zero (better for 55% of agents, a coin flip) and not at all on SWE-bench.
Decomposition is supported as "model carries the ability"; the harness term is not shown to add measurable transfer here. Caveat: many harnesses appear with only 1 to 3 models, so the harness effect is weakly identified.
Relevance to our task: this transfers across agents (new agent, known tasks), not to new tasks, so it does not bear on the 0.131 baseline.

## Method 2. Plain shares against IRT on Terminal-Bench 2.0

All 143 rows, 12,727 cells. Rank correlation on full data (1PL fit): Spearman(theta, agent success share) = 0.9997, Spearman(-beta, task success share) = 0.99997.
That is close to tautological: with a near-complete grid and similar trial counts, 1PL ability is nearly a monotone function of the share. IRT adds no new ranking here.

5-fold random cell holdout, same folds for all models, held-out log-loss per attempt (lower is better):

| model | fold 1 | 2 | 3 | 4 | 5 | pooled | paired diff vs 1PL (cell bootstrap 95%) |
|---|---|---|---|---|---|---|---|
| (i) 1PL | 0.390 | 0.370 | 0.385 | 0.389 | 0.379 | 0.3826 | reference |
| (ii) additive logit of smoothed shares | 0.411 | 0.395 | 0.405 | 0.408 | 0.402 | 0.4041 | +0.0216 [+0.0193, +0.0238] |
| (iii) 2PL | 0.374 | 0.359 | 0.368 | 0.372 | 0.368 | 0.3681 | -0.0144 [-0.0170, -0.0121] |
| (iv) 1PL + rank-2 interaction | 0.376 | 0.357 | 0.363 | 0.375 | 0.375 | 0.3689 | -0.0136 [-0.0179, -0.0091] |

2PL and rank 2 beat 1PL in every fold; shares lose to 1PL in every fold. Verdict. Rankings of agents and tasks are the same from shares and IRT; for predicting
single cells 1PL beats smoothed shares by 0.02 log-loss and richer models (discrimination or interaction) add another 0.014. The gain of 2PL and rank 2 over 1PL
is small and consistent but a constant-probability reference was not computed. IRT is not
needed for ranking, is moderately useful for cell prediction. Caveat: smoothing pseudo-count 2 was fixed and not tuned; a tuned additive model could close part of the gap.

## Method 3. Overdispersion of repeated attempts on Terminal-Bench 2.0

Pearson statistic sum (s - n p)^2 / (n p (1-p)) over cells with n >= 2 (12,412 cells) under the fitted 1PL.

| quantity | value |
|---|---|
| in-sample dispersion ratio (chi2 / (cells - agents - tasks) = 29,254 / 12,180) | 2.40 |
| same, rank-2 model (dof corrected for its extra parameters) | 2.04 |
| null: simulated binomial data from the fitted 1PL, refit, same statistic (20 sims) | 0.951 +- 0.022 |
| cross-fitted (held-out cells) ratio, 1PL / 2PL / rank-2 | 2.66 / 5.66 / 13.9 |
| null for cross-fitted ratio (20 sims) | 0.984 +- 0.022 |

The ratio 2.4 is far outside the binomial null (0.95), so attempts within a cell are positively correlated beyond what an additive agent-task logit explains.
Rank 2 removes only a small part, so most of it is within-cell, not a missing low-rank interaction. (The cross-fitted ratios for 2PL and rank 2 are inflated by
overconfident held-out probabilities near 0 or 1, which the Pearson statistic punishes heavily; do not read them as worse dispersion.)

Implication for pass^k. On the 11,376 cells with exactly 5 attempts, unbiased observed pass^k (C(s,k)/C(5,k), cell-bootstrap CI) against the 1PL independence prediction mean p^k (cross-fitted):

| k | observed | 1PL independent | rank-2 independent |
|---|---|---|---|
| 1 | 0.462 [0.454, 0.470] | 0.462 | 0.462 |
| 2 | 0.382 [0.374, 0.391] | 0.340 | 0.354 |
| 3 | 0.341 [0.333, 0.349] | 0.279 | 0.298 |
| 4 | 0.314 [0.306, 0.323] | 0.241 | 0.263 |
| 5 | 0.295 [0.287, 0.303] | 0.214 | 0.237 |

Treating attempts as independent with the 1PL probability underestimates pass^5 by about 8 points (0.21 versus 0.295, a 28% relative shortfall), and the error grows with k.
Reliability is therefore more all-or-nothing per cell than the 1PL p implies: pass^k must be estimated from repeated attempts, not extrapolated as p^k.
Verdict: overdispersion is real and large here. Caveat: the cause is not identified (it can be stable agent-task idiosyncrasy, shared environment or seed effects, or leaderboard
submissions pooling different runs); only the size of the departure from binomial is established.

## What was not run / limits
- SWE-bench: only 33 of 134 names were parseable into harness and model; 19 evaluable held-out agents.
- Method 1 uses the leaderboard `agent` and `model` strings as harness and model; harness version and reasoning settings are ignored.
- Nothing here is on the supervisor's four new-process scenarios, so there is no verdict under section 8 of EVALUATION_PROTOCOL.md.
