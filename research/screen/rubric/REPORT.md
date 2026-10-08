# Rubric group: procedure step rubric, LARA/T-IPO, RPA criteria (4 Oct 2026)

All numbers use the shared harness and the baseline protocol (same folds, fold IRT from the repository cache, repository evaluator). The harness reproduces the baseline: pooled rho 0.131 [0.027, 0.255]. LLM: gemini-3.8-flash on Vertex, temperature 0, thinking level low. 3,715 calls in total (1,205 per rubric plus 100 for the repeat), every response cached in cache/*.jsonl. Not a single answer needed a retry. No key or project id is written anywhere.

## 1. What was run

**Procedure step rubric (our own).** The model reads statements.jsonl text and lists the steps the agent must do, each labelled with one of seven types (check_condition, lookup_document, ask_customer, write_system, irreversible_action, branch, refuse_or_handover) and a nesting depth. It also gives the number of refusal conditions, a 0/1 flag "decision depends on data not in the request", and the probability that the correct outcome is a refusal. Features (12): total steps, 7 counts by type, max nesting depth, refusal conditions, hidden-data flag, p_refusal. The prompt has a definition and one short example per label (appendix file RUBRIC_PROMPTS.md). Deviation to note: the statements of tau2 and tauk_banking contain only a customer scenario and no policy, so there the model was told to use the policy such a domain would normally apply, which means it relies on its own knowledge of the domain. SOPBench statements contain the policy text.

**LARA / T-IPO readiness (Li and Ye 2026).** Dimensions D1 cognitive complexity (Bloom anchored), D2 data dependency, D3 interaction diversity, D4 compliance sensitivity, D5 innovation requirement, each scored 1 to 5 with the paper's anchors. From the scores I compute the paper's weighted mean (weights 1, 1, 1, 1.5, 1), the level L1 to L4 with the paper's thresholds (2.0, 3.0, 4.0) and the D4 floor rule (D4 at least 4 gives at least L3). Features (7): D1 to D5, weighted mean, level. Deviation: the paper rates tasks of a securities firm by human raters; here the "task" is "handle this customer request as an agent" and the rater is one LLM call. The paper's weights and thresholds were calibrated for financial services, I used them unchanged.

**RPA selection criteria (Wellmann et al. 2020).** The 13 criteria of the process characteristics evaluation framework (standardization, maturity, determinism, failure rate, frequency, duration, urgency, structuredness, interfaces, stability, number of systems, resources, proneness to human error), each scored 0 to 2 (2 = supports automation). Features (14): 13 scores and their sum. Deviation: the paper measures most criteria from event logs (number of variants, exceptions, users). We have no logs, so the model estimates each from the request text and the kind of process; frequency, urgency, stability, resources, maturity and failure rate are therefore pure guesses about the process type and carry almost no task-level information.

**Feature variance.** Within a benchmark many LARA and RPA columns are nearly constant. SOPBench: D5 has standard deviation 0.12 and D1 0.03; 6 of 13 RPA columns have standard deviation below 0.06. tau2/tauk: D3 is constant 2. The hidden-data flag is constant 1 for tau2 and tauk_banking. These features cannot separate tasks inside a benchmark, which limits what they can add.

**Runs through the harness.** Each set alone with the baseline ridge (rub_proc, rub_lara, rub_rpa). Each combined with ADeLe in two forms: one ridge on the concatenated columns (cat_*), and the repository's grouped ridge with one penalty grid per family (grp_*, both families use the repository's "LLM Judge" grid). Controls: total steps only (len_ridge, level 1 as ridge), counts by type (types_ridge, level 2 as ridge). Full: procedure rubric + ADeLe + human_time (cat_full, grp_full). Missing features: none (0 failed parses), so no imputation was needed.

**Formula ladder (ladder.py).** Logistic models fitted on the training cells of each fold (L-BFGS, binomial likelihood for repeated attempts), features standardised on the training tasks, ridge penalty chosen by inner 5-fold CV over training tasks only (log-loss). Agent abilities theta_a are fitted jointly (not taken from the IRT cache).
- L1: logit P = theta_a - delta * n_steps.
- L2: logit P = theta_a - sum_t delta_t n_t - v'g, with g = the four global rubric features (max depth, refusal conditions, hidden-data flag, p_refusal). I read "v'g" this way; the spec did not define g. A variant lad2_adele puts the 18 ADeLe levels into g as well.
- L3: L2 plus per-agent deviations gamma_at on the type counts, penalty on gamma chosen from a strong grid (0.3, 3, 30, applied to the mean squared gamma).
- Added control lad0: the same logistic fit with no task feature at all (agent intercepts only).
Predicted difficulty for rho: the task part of the logit (for L3 averaged over agents).

## 2. Results (pooled over the four new-process scenarios: SOPBench new procedures, SOPBench new domain, tau2 new procedures, tau2 new domain)

Brier skill, log-loss skill and within-agent AUC are means over the four scenarios. "Within-agent AUC" is the repository's auc_within_config. Verdict by EVALUATION_PROTOCOL section 8: the rule needs the paired pooled interval above zero; criterion 5 (confirmation on an untouched benchmark) was not run, so even a pass would only be "candidate".

| method | pooled rho | 95% CI | worst scen. | sop-proc | sop-dom | tau2-proc | tau2-dom | paired diff vs baseline 95% CI | Brier skill | logloss skill | within-agent AUC | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adele | +0.131 | [+0.027, +0.255] | +0.120 | +0.120 | +0.125 | +0.128 | +0.150 | - | +0.000 | -0.003 | 0.562 | baseline |
| rub_proc | +0.088 | [-0.025, +0.188] | +0.067 | +0.102 | +0.091 | +0.091 | +0.067 | [-0.190, +0.049] | +0.005 | +0.002 | 0.571 | not better |
| rub_lara | +0.056 | [-0.026, +0.082] | -0.005 | -0.005 | +0.021 | +0.066 | +0.143 | [-0.235, +0.007] | +0.002 | -0.000 | 0.535 | not better |
| rub_rpa | +0.023 | [-0.038, +0.136] | -0.017 | +0.040 | -0.017 | +0.065 | +0.004 | [-0.202, +0.013] | +0.003 | -0.002 | 0.522 | not better |
| len_ridge | -0.005 | [-0.140, +0.189] | -0.094 | +0.028 | +0.049 | -0.094 | -0.004 | [-0.276, +0.006] | -0.007 | -0.007 | 0.536 | not better |
| types_ridge | +0.086 | [-0.015, +0.190] | +0.063 | +0.087 | +0.102 | +0.090 | +0.063 | [-0.169, +0.049] | +0.004 | +0.001 | 0.571 | not better |
| cat_proc_adele | +0.148 | [+0.032, +0.261] | +0.104 | +0.165 | +0.168 | +0.104 | +0.156 | [-0.063, +0.076] | +0.015 | +0.009 | 0.583 | not better |
| grp_proc_adele | +0.172 | [+0.053, +0.269] | +0.133 | +0.194 | +0.179 | +0.181 | +0.133 | [-0.060, +0.091] | -0.002 | -0.020 | 0.578 | not better |
| cat_lara_adele | +0.133 | [+0.032, +0.255] | +0.122 | +0.122 | +0.135 | +0.129 | +0.145 | [-0.025, +0.029] | +0.014 | +0.008 | 0.572 | not better |
| grp_lara_adele | +0.114 | [+0.014, +0.224] | +0.084 | +0.123 | +0.137 | +0.112 | +0.084 | [-0.065, +0.018] | -0.002 | -0.024 | 0.573 | not better |
| cat_rpa_adele | +0.115 | [+0.034, +0.231] | +0.091 | +0.130 | +0.103 | +0.136 | +0.091 | [-0.049, +0.045] | +0.008 | +0.003 | 0.559 | not better |
| grp_rpa_adele | +0.115 | [+0.060, +0.217] | +0.088 | +0.136 | +0.119 | +0.119 | +0.088 | [-0.083, +0.070] | -0.023 | -0.051 | 0.559 | not better |
| cat_full | +0.140 | [+0.031, +0.266] | +0.105 | +0.163 | +0.161 | +0.105 | +0.132 | [-0.062, +0.073] | +0.016 | +0.010 | 0.576 | not better |
| grp_full | +0.161 | [+0.049, +0.266] | +0.111 | +0.192 | +0.180 | +0.159 | +0.111 | [-0.056, +0.088] | +0.002 | -0.012 | 0.571 | not better |
| lad0 | +0.036 | [+0.000, +0.071] | +0.000 | +0.070 | +0.074 | +0.000 | +0.000 | [-0.235, +0.017] | +0.040 | +0.059 | 0.500 | not better |
| lad1 | -0.008 | [-0.142, +0.187] | -0.094 | +0.022 | +0.045 | -0.094 | -0.004 | [-0.279, +0.003] | +0.039 | +0.058 | 0.536 | not better |
| lad2 | +0.135 | [+0.012, +0.218] | +0.118 | +0.118 | +0.127 | +0.141 | +0.154 | [-0.142, +0.110] | +0.052 | +0.070 | 0.580 | not better |
| lad2_adele | +0.182 | [+0.046, +0.270] | +0.168 | +0.181 | +0.179 | +0.168 | +0.199 | [-0.061, +0.101] | +0.059 | +0.076 | 0.589 | not better |
| lad3 | +0.137 | [+0.010, +0.219] | +0.118 | +0.118 | +0.127 | +0.147 | +0.156 | [-0.141, +0.106] | +0.056 | +0.073 | 0.585 | not better |


Reading the table:
- No method passes the improvement rule. All paired intervals of the difference to the baseline contain zero. Largest point gain: lad2_adele (+0.051 rho, interval [-0.061, +0.101]); next grp_proc_adele (+0.041, [-0.060, +0.091]).
- The procedure rubric alone (rho 0.088) is below ADeLe alone (0.131) but its interval overlaps; step counts alone (len_ridge, lad1) give rho about 0 and are the worst, so the "only length matters" control does not work: more steps does not mean harder in these data.
- Counts by type (level 2) lift rho from about 0 to 0.09 (ridge) and 0.135 (logistic L2, which also uses the four global features), so the type information carries the signal, not the length. Within-agent AUC rises 0.536 (L1) to 0.580 (L2).
- LARA alone (0.056) and RPA alone (0.023) are near zero; combined with ADeLe they do not help (0.133 and 0.115, intervals contain the baseline).
- Adding the procedure rubric to ADeLe moves rho from 0.131 to 0.148 (concatenated) or 0.172 (grouped) with a visible but non-significant gain, and the grouped version has worse log-loss skill (-0.020) than the baseline, so criterion 4 fails for it. Human_time added on top does not help (0.140, 0.161).

## 3. Ladder, held-out comparison (per scenario: sop-proc, sop-dom, tau2-proc, tau2-dom)

| model | log-loss skill | within-agent AUC | rho (within domain) |
|---|---|---|---|
| baseline adele | +0.011, +0.003, -0.000, -0.028 | 0.580, 0.549, 0.556, 0.562 | 0.120, 0.125, 0.128, 0.150 |
| lad0 (no feature) | +0.040, +0.063, +0.045, +0.087 | 0.500 x4 | 0.070, 0.074, 0, 0 (ties) |
| L1 steps | +0.042, +0.060, +0.045, +0.086 | 0.543, 0.534, 0.545, 0.520 | 0.022, 0.045, -0.094, -0.004 |
| L2 types + globals | +0.064, +0.072, +0.050, +0.095 | 0.600, 0.557, 0.587, 0.578 | 0.118, 0.127, 0.141, 0.154 |
| L2 + ADeLe | +0.068, +0.081, +0.055, +0.098 | 0.609, 0.571, 0.593, 0.583 | 0.181, 0.179, 0.168, 0.199 |
| L3 (gamma_at) | +0.066, +0.073, +0.061, +0.094 | 0.605, 0.557, 0.606, 0.571 | 0.118, 0.127, 0.147, 0.156 |

- **The log-loss gains of the ladder are mostly calibration, not features.** lad0, which has no task feature, already reaches log-loss skill +0.04 to +0.09 and Brier skill +0.04, against about 0 for the baseline and for "constant". The reason: "constant" and the baseline take theta from the fold IRT, where sigmoid(theta) is not the agent's marginal success rate; a plain refit of agent intercepts on the training outcomes fixes that. Beyond this free gain, L1 adds nothing (0.058 vs 0.059), L2 adds about +0.011 log-loss skill, and L3 adds +0.003 over L2. This is a finding about the baseline pipeline that is independent of the rubric: the Brier/log-loss skills of the ridge methods are held back by the frozen-IRT calibration. Criterion 4 of the improvement rule (skill not lower than the baseline) is therefore easy for any logistic refit and should not be read as evidence for the rubric.
- L2 over L1: rho +0.14 and within-agent AUC +0.04 to +0.07 (consistent over 4 scenarios): the type decomposition matters, the total does not.
- L3 over L2: within-agent AUC changes by +0.005, 0.000, +0.019, -0.007 and rho by 0, 0, +0.006, +0.002. Per-agent sensitivities to step types do not add held-out predictive power here (the fitted penalty is strong; with 28 agents and 7 counts the deviations are poorly identified).
- The best point estimate, L2 + ADeLe (rho 0.182, interval [0.046, 0.270], paired difference [-0.061, +0.101]), is still not significant, and the ladder's penalties and the choice g were fixed before seeing results, but only one variant of g was tried besides the two shown.

## 4. Repeatability of the procedure rubric (100 random tasks pooled over benchmarks, same prompt, two independent calls at temperature 0)

- Pooled Cohen's kappa over the seven type-count columns (counts as categories): 0.62; linearly weighted 0.77.
- Total steps: Pearson 0.77, exact equality in 30% of tasks. Exact same sequence of type labels: 14% of tasks.
- By column (kappa on count categories, Pearson of counts):
```
feature,kappa,kappa_linear_weighted,pearson,spearman,identical_share
n_steps,0.225,0.564,0.772,0.788,0.3
n_check_condition,0.569,0.759,0.884,0.825,0.65
n_lookup_document,0.445,0.624,0.78,0.805,0.58
n_ask_customer,0.649,0.842,0.945,0.954,0.83
n_write_system,0.583,0.629,0.666,0.715,0.72
n_irreversible_action,0.593,0.609,0.583,0.687,0.78
n_branch,0.376,0.315,0.165,0.331,0.61
n_refuse_or_handover,0.493,0.423,0.34,0.454,0.85
max_depth,0.519,0.55,0.547,0.548,0.63
refusal_conditions,0.923,0.967,0.99,0.977,0.94
needs_hidden_data,0.662,0.662,0.704,0.704,0.99
p_refusal,,,0.955,0.872,0.52
```
- Refusal conditions are very stable (r 0.99); ask_customer, check_condition and lookup counts are stable (r 0.78 to 0.95); branch and refuse_or_handover counts are the noisiest (r 0.17 and 0.34), so features built on them mostly carry noise. p_refusal r 0.96 but only 52% identical. The test-retest check is of API/sampling noise only, since the prompt was the same; it does not measure sensitivity to prompt wording.

## 5. What could not be run or is weak

- Criterion 5 of the improvement rule (confirmation on a held-out benchmark with frozen settings) was not run; no HANDBOOK data touched.
- tauk_banking appears only in the random_cases sanity scheme (single domain, 97 tasks), which is not in the pooled figure. There, rho: ADeLe 0.523, rub_proc 0.520, L2 0.491, L2+ADeLe 0.589 (sanity only).
- The two-run agreement uses identical prompts; no paraphrase variant was tried.
- Wellmann criteria that need logs were guessed from text (see section 1), LARA was rated by one LLM rather than human experts, and tau2/tauk policy was supplied from the model's own knowledge. Low-variance features limit what LARA and RPA could show; this says little about the original frameworks.
- The ladder's g and penalty grids were my own choices; nothing was tuned on test folds (penalties by inner CV on training tasks).

## 6. Paths

- Folder: research/screen/rubric/
- Prompts: prompts.py, appendix RUBRIC_PROMPTS.md. LLM runner and cache: llmrun.py, cache/{proc,lara,rpa}.jsonl (proc.jsonl also holds the repeat run, key rep 2).
- Features (case_id first): features/rub_proc_<bench>.csv, rub_len, rub_types, rub_lara, rub_rpa; concatenations in features/concat/. Built by build_features.py.
- Ladder: ladder.py. Harness run: run_screen.py (eval.json, eval.txt in out/run/), control run_lad0.py (out/run_lad0/). Agreement: agreement.py -> out/agreement.csv, out/agreement.json. Results table: make_table.py -> out/results_table.md.
- Main eval: research/screen/rubric/out/run/eval.json (lad0 is only in out/run_lad0/eval.json).
