# targets: plain shares instead of IRT (E1), beta-binomial variant, pass^k reliability (E6), combinations

All numbers use exactly the baseline protocol (harness.py: same folds, fold IRT read from the repository cache, repository evaluator). No LLM calls. Baseline: pooled rho 0.131 [0.027, 0.255] over the four new-process scenarios (reproduced exactly).

## 1. What was run

**E1, plain shares instead of IRT (supervisor's question).** Predictor `SharesPredictor` in `shares.py`. Inside each fold, using only training-task outcomes: agent strength = logit((s+0.5)/(n+1)), s and n summed over the agent's training cells (binomial cells for tau2 and tauk_banking count each attempt). Task difficulty target = logit((f+0.5)/(n+1)), f failures over agents on that training task. Ridge from features to that target is the baseline's own `FeatureBasedPredictor` (StandardScaler, RidgeCV, alphas 0.01 to 1e5, cv=5), unchanged. P(a, j) = sigmoid(strength_a - difficulty_hat_j + c), where one constant c is fitted by maximum likelihood (binomial likelihood) on training cells, using the shrunken-free training difficulties, so probabilities are calibrated on average. Run with three feature sets: ADeLe (the baseline's 18 levels), human_time, length. Columns: `shares_adele`, `shares_human_time`, `shares_length`. Deviation from the paper: this is not in a paper, it is the supervisor's control; no IRT is fitted in the model itself.

**Variant, beta-binomial empirical Bayes.** Same pipeline, but +0.5/+1 is replaced by a Beta(a, b) prior fitted by maximum likelihood on the training counts (beta-binomial marginal), separately for agent success shares and for task failure shares (a, b clipped to [0.05, 50]). Columns `bb_adele`, `bb_human_time`, `bb_length`.

**E6, reliability over repeated attempts (tau2, tauk_banking).** Script `passk.py`. Input: the baseline's out-of-fold P (the `adele` column of `run/<bench>/obs_*.csv.gz`). For k = 2, 3, 4 two predictions per (agent, task) cell: independent attempts p^k, and beta-binomial pass^k = prod_{i<k}(p*phi + i)/(phi + i) with mu = p. phi is fitted by maximum likelihood (beta-binomial likelihood of the cell's s successes out of n attempts, mu = p) on the cells of the other folds of the same seed only, then applied to the held-out fold. Caveat: those other-fold P values were themselves produced by models that saw some of the held-out fold's tasks as training, so phi is "training folds only" for the cell outcomes being scored but not fully nested. Observed target: responses.jsonl stores only successes and trials per cell (no attempt order), so the indicator "all k attempts succeeded" is taken as the share of k-subsets of the n attempts that are all successes, C(s,k)/C(n,k); the Brier score is the exact mean over those subsets of (pred - indicator)^2. Cells with n < k are skipped. Pairing: bootstrap over tasks (2000 draws), difference beta-binomial minus independent (negative = beta-binomial better).

**Combinations.** Equal-weight logit averages formed by the repository evaluator (`combos`): shares_adele+adele, bb_adele+adele, shares_human_time+adele, shares_adele+shares_human_time+adele.

## 2. Results (E1, variant, combinations)

Scenario order: sopbench new_procedures, sopbench new_domain, tau2 new_procedures, tau2 new_domain. "diff" = paired 95% CI of (method minus baseline) pooled rho. Brier and log-loss skill are means over the four scenarios. Verdict by section 8 of EVALUATION_PROTOCOL.md: rules 1 to 4 checked here; rule 5 (held-out benchmark confirmation) was not run for any method, so no method can be called "better" in the full sense.

| method | pooled rho [95% CI] | worst | rho by scenario | diff to baseline [95% CI] | Brier skill | log-loss skill | verdict |
|---|---|---|---|---|---|---|---|
| adele (BASELINE) | 0.131 [0.027, 0.255] | 0.120 | 0.120, 0.125, 0.128, 0.150 | - | +0.000 | -0.003 | baseline |
| bb_adele+adele | 0.131 [0.028, 0.251] | 0.121 | 0.121, 0.125, 0.121, 0.156 | [-0.004, +0.007] | +0.021 | +0.029 | not better |
| shares_human_time+adele | 0.130 [0.033, 0.262] | 0.116 | 0.116, 0.123, 0.134, 0.147 | [-0.026, +0.040] | +0.022 | +0.030 | not better |
| shares_adele+adele | 0.129 [0.029, 0.253] | 0.122 | 0.122, 0.124, 0.123, 0.149 | [-0.004, +0.005] | +0.019 | +0.026 | not better |
| shares_adele+shares_human_time+adele | 0.129 [0.036, 0.260] | 0.119 | 0.119, 0.120, 0.128, 0.150 | [-0.019, +0.033] | +0.026 | +0.037 | not better |
| shares_adele | 0.128 [0.025, 0.251] | 0.112 | 0.120, 0.123, 0.112, 0.155 | [-0.009, +0.005] | +0.033 | +0.046 | not better |
| bb_adele | 0.127 [0.029, 0.251] | 0.108 | 0.120, 0.126, 0.108, 0.153 | [-0.008, +0.008] | +0.035 | +0.050 | not better |
| shares_human_time | 0.106 [-0.002, 0.281] | 0.079 | 0.094, 0.127, 0.079, 0.124 | [-0.131, +0.106] | +0.027 | +0.042 | not better |
| bb_human_time | 0.106 [-0.002, 0.281] | 0.079 | 0.094, 0.127, 0.079, 0.124 | [-0.131, +0.106] | +0.030 | +0.046 | not better |
| bb_length | 0.043 [-0.187, 0.230] | -0.051 | -0.020, 0.032, -0.051, 0.211 | [-0.318, +0.047] | -0.000 | +0.023 | not better |
| shares_length | 0.043 [-0.187, 0.230] | -0.051 | -0.020, 0.032, -0.051, 0.211 | [-0.318, +0.046] | -0.007 | +0.017 | not better |

Exact differences in pooled rho to the baseline (0.1308): shares_adele 0.1276 (-0.003, CI [-0.009, +0.005]); bb_adele 0.1267 (-0.004, CI [-0.008, +0.008]); shares_human_time 0.1059 (-0.025, CI [-0.131, +0.106]); shares_length 0.0430 (-0.088, CI [-0.318, +0.046]). The baseline's own human_time and length rows are in the repository results, not repeated here.

Findings:
- Answer to the supervisor: with ADeLe features, replacing fold IRT by plain smoothed shares changes pooled rho by -0.003 (interval contains zero, and the interval is narrow, about +-0.007). The IRT step adds nothing measurable for ranking new tasks; the signal comes from the features.
- Beta-binomial prior gives the same rho as +0.5/+1 (0.1267 vs 0.1276). For human_time and length the two are identical to three decimals in rho (they differ only in Brier and log-loss skill).
- Brier and log-loss skill of the shares models (+0.033 and +0.046 for shares_adele) are higher than the baseline's (+0.0003 and -0.003), because c is calibrated by maximum likelihood on training cells and the IRT-based P is not. This is a calibration gain, not a ranking gain.
- Combinations with the baseline do not move rho (0.129 to 0.131); they raise Brier and log-loss skill to about +0.02 and +0.03.
- human_time alone is lower pooled (0.106, interval includes zero) but is the best on random_cases in sopbench (0.119 vs 0.092) and tauk_banking (0.563 vs 0.523). Length alone is unstable (worst -0.051).
- Rule 1 fails for every method (no interval above zero), so every verdict is "not better". Rule 2 holds for all (no scenario interval entirely below zero); rule 4 holds for all except the length variants.

random_cases (sanity only, rho within domain): sopbench adele 0.092, shares_adele 0.090, bb_adele 0.091; tau2 0.171, 0.168, 0.171; tauk_banking 0.523, 0.525, 0.524.

## 3. E6 results: pass^k from the baseline's P

Mean Brier of the pass^k prediction (lower is better), mean predicted vs mean observed pass^k. "phi" is the mean over seeds and folds of the fitted overdispersion (range over folds in brackets). Diff = beta-binomial minus independent Brier, task-bootstrap 95% CI.

| bench / scheme | phi | k | Brier indep | Brier beta-bin | diff [95% CI] | mean pred indep | mean pred beta-bin | mean observed |
|---|---|---|---|---|---|---|---|---|
| tau2|new_procedures | 1.31 [1.17, 1.49] | 2 | 0.2180 | 0.2197 | +0.0018 [-0.0016, +0.0048] | 0.703 | 0.756 | 0.671 |
| tau2|new_procedures | 1.31 [1.17, 1.49] | 3 | 0.2379 | 0.2406 | +0.0027 [-0.0039, +0.0089] | 0.609 | 0.713 | 0.610 |
| tau2|new_procedures | 1.31 [1.17, 1.49] | 4 | 0.2490 | 0.2533 | +0.0043 [-0.0052, +0.0131] | 0.535 | 0.683 | 0.564 |
| tau2|new_domain | 1.19 [0.88, 1.38] | 2 | 0.2473 | 0.2459 | -0.0014 [-0.0046, +0.0018] | 0.733 | 0.783 | 0.671 |
| tau2|new_domain | 1.19 [0.88, 1.38] | 3 | 0.2708 | 0.2730 | +0.0022 [-0.0038, +0.0079] | 0.652 | 0.749 | 0.610 |
| tau2|new_domain | 1.19 [0.88, 1.38] | 4 | 0.2824 | 0.2914 | +0.0089 [+0.0004, +0.0172] | 0.590 | 0.725 | 0.564 |
| tau2|random_cases | 1.40 [1.36, 1.53] | 2 | 0.2055 | 0.2088 | +0.0033 [+0.0003, +0.0061] | 0.702 | 0.754 | 0.671 |
| tau2|random_cases | 1.40 [1.36, 1.53] | 3 | 0.2254 | 0.2292 | +0.0038 [-0.0021, +0.0096] | 0.608 | 0.710 | 0.610 |
| tau2|random_cases | 1.40 [1.36, 1.53] | 4 | 0.2370 | 0.2416 | +0.0045 [-0.0041, +0.0128] | 0.533 | 0.678 | 0.564 |
| tauk_banking|random_cases | 0.80 [0.73, 0.91] | 2 | 0.2439 | 0.2123 | -0.0316 [-0.0446, -0.0184] | 0.134 | 0.231 | 0.328 |
| tauk_banking|random_cases | 0.80 [0.73, 0.91] | 3 | 0.2364 | 0.1942 | -0.0422 [-0.0600, -0.0250] | 0.070 | 0.195 | 0.281 |
| tauk_banking|random_cases | 0.80 [0.73, 0.91] | 4 | 0.2237 | 0.1799 | -0.0438 [-0.0626, -0.0262] | 0.041 | 0.172 | 0.250 |

Findings:
- tau2: independent attempts is better or equal. Beta-binomial raises predicted pass^k (0.76, 0.71, 0.68 for k = 2, 3, 4 against observed 0.67, 0.61, 0.56) and over-shoots; independent p^k is closer to the observed mean (0.70, 0.61, 0.53 in new_procedures). Differences in Brier are small (0.002 to 0.009), significant only for k=4 new_domain and k=2 random_cases (both in favour of independent). The baseline's mean p on tau2 is already close to the observed mean success, so extra dispersion pushes predictions above the truth.
- tauk_banking: beta-binomial is clearly better (Brier 0.212, 0.194, 0.180 vs 0.244, 0.236, 0.224; diff -0.032, -0.042, -0.044, all intervals below zero). The reason is a calibration failure of the baseline P on this benchmark, not the dispersion itself: independent p^k predicts pass^2 = 0.134 against 0.328 observed; the baseline P is too low on average, and the beta-binomial form with phi about 0.8 (strong overdispersion, attempts within a cell highly correlated) lifts the product tail (0.231, 0.195, 0.172). It still under-predicts pass^k by 0.09 to 0.08.
- Fitted phi: tau2 1.2 to 1.4 (new_domain 0.88 to 1.38 across folds), tauk_banking 0.8 (0.73 to 0.91). All are small, i.e. attempts of the same agent on the same task are strongly correlated, far from independent. Small phi alone does not make beta-binomial better when the underlying p is mis-calibrated, as seen on tau2.
- Verdict: no uniform winner. Beta-binomial wins when p is biased low (tauk_banking), independent wins slightly when p is roughly calibrated (tau2). A fair test of the overdispersion idea would first recalibrate p (for example with the shares-model constant c), which was not done here.

## 4. What could not be run / limits
- Per-attempt order is not in responses.jsonl, so the "all k attempts succeeded" indicator is the k-subset average, not one fixed set of the first k attempts.
- phi for E6 comes from other folds' out-of-fold P (not fully nested, see above).
- Rule 5 of the improvement rule (confirmation on an unused held-out benchmark) was not run; none of the methods passes rule 1 anyway.
- tauk_banking has one domain and unique procedures, so it only has random_cases; it is not in the pooled four-scenario figure.

## 5. Paths
- Scripts: research/screen/targets/shares.py (E1, variant, combos), research/screen/targets/passk.py (E6)
- Results: research/screen/targets/run/eval.json, run/eval.txt, run/<bench>/obs_*.csv.gz; E6: passk.json; run log shares.log; table.json
