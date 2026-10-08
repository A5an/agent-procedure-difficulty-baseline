# Model tricks: within-domain centering, borrowing from other benchmarks, stacking

## 1. Idea
Three cheap changes to the baseline ridge (beta from features, P = sigmoid(theta - beta_hat)), no new features and no LLM calls.
(1) Fit the ridge on within-domain centered data, because the metric ignores between-domain differences. (2) Add tasks of the other benchmarks to the training set. (3) Stack four ridges with non-negative least squares (Wolpert 1992, van der Laan et al. 2007).
The three tricks are chosen by us, not taken from a paper that applies them to this problem.

## 2. Setup
Code: mt.py (predictors), run.py (run), debug.py (one-fold check). Output: out/ (eval.json, eval.txt, extra_metrics.json, stack_log.json).
One harness run, benches sopbench and tau2, schemes new_procedures and new_domain only (no random cases, no banking evaluation), under cpu_lock. No LLM calls.
Every variant is a custom predictor that reads the feature tables directly and keeps the baseline's StandardScaler + RidgeCV(RB.ALPHAS, cv=5). Fold IRT abilities come from the repository cache as usual. A plain predictor of mine on ADeLe reproduced the baseline betas to 5e-7 on a debug fold, and the baseline column adele gives 0.131 pooled.

Variants (fixed before the run, all reported):
- wd_center_adele: subtract the domain mean from beta and each feature (means over training tasks), ridge, then at prediction subtract the mean of the test task's domain computed over all tasks of that domain in the benchmark (texts only), add back the overall training mean of beta.
- wd_z_adele: the same with division by the domain standard deviation (zero std replaced by 1).
- wd_center_adele_rub: wd_center on ADeLe 18 + 12 rubric columns.
- xbench_adele, xbench_adele_rub: for sopbench add all tau2 and tauk_banking tasks, for tau2 add all sopbench and banking tasks. Borrowed beta (full-data IRT items.csv) is z-scored inside each domain, rescaled to the training fold's beta std, shifted to the training fold's beta mean (my choice, the brief did not say), weight 0.5 passed as sample_weight to RidgeCV. Own training tasks have weight 1. The test benchmark's own test tasks never enter. Caveat: the inner 5-fold CV of RidgeCV picks alpha on a mix of own and borrowed rows.
- stack4: four base ridges (ADeLe, 12 rubric, human_time + length, ADeLe + rubric). Inner out-of-fold predictions by GroupKFold(5) over procedure_id, NNLS with a free intercept (done by centering), base learners refit on the whole training set.
- cat_adele_rub: plain ridge on ADeLe + rubric, the untricked version of the first three variants and the xbench rubric variant.
- Combos: equal-weight logit average of (wd_center_adele_rub, adele) and (stack4, adele); with lad2_adele (from rubric/out/run) in extra metrics.
Side effect: the debug run created one stray IRT cache directory in the repository results (seed0_fold0of1_1pl), which I deleted afterwards.

## 3. Results (pooled within-domain rho over the four new-process scenarios)
Scenarios: sop-proc, sop-dom, tau2-proc, tau2-dom. Brier and log-loss skill are means over the four scenarios. Paired CI is the difference to adele.

| method | pooled rho | 95% CI | worst | sop-proc | sop-dom | tau2-proc | tau2-dom | paired CI vs adele | Brier skill | log-loss skill | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| adele (baseline) | 0.131 | [0.027, 0.255] | 0.120 | 0.120 | 0.125 | 0.128 | 0.150 | - | 0.000 | -0.004 | - |
| cat_adele_rub (plain) | 0.148 | [0.032, 0.261] | 0.104 | 0.165 | 0.168 | 0.104 | 0.156 | [-0.063, 0.076] | 0.015 | 0.009 | not better |
| wd_center_adele | 0.117 | [0.012, 0.227] | 0.083 | 0.083 | 0.097 | 0.145 | 0.144 | [-0.059, 0.016] | 0.004 | 0.004 | not better |
| wd_z_adele | 0.098 | [-0.011, 0.224] | 0.081 | 0.081 | 0.105 | 0.090 | 0.115 | [-0.076, 0.005] | 0.001 | 0.002 | not better |
| wd_center_adele_rub | 0.135 | [0.019, 0.245] | 0.116 | 0.144 | 0.126 | 0.116 | 0.153 | [-0.091, 0.062] | 0.012 | 0.010 | not better |
| xbench_adele | 0.114 | [0.031, 0.222] | 0.062 | 0.062 | 0.079 | 0.131 | 0.185 | [-0.102, 0.050] | 0.005 | 0.003 | not better |
| xbench_adele_rub | 0.189 | [0.011, 0.263] | 0.095 | 0.095 | 0.120 | 0.224 | 0.316 | [-0.115, 0.117] | 0.026 | 0.025 | not better (point estimate highest) |
| stack4 | 0.095 | [-0.035, 0.223] | -0.026 | 0.151 | 0.137 | 0.117 | -0.026 | [-0.130, 0.044] | -0.011 | -0.008 | not better |
| wd_center_adele_rub + adele | 0.141 | [0.035, 0.258] | 0.128 | 0.152 | 0.146 | 0.138 | 0.128 | [-0.050, 0.059] | 0.011 | 0.010 | not better |
| stack4 + adele | 0.122 | [0.008, 0.253] | 0.020 | 0.161 | 0.162 | 0.143 | 0.020 | [-0.064, 0.051] | 0.017 | 0.014 | not better |

The verdict column applies "paired CI entirely above 0" as the brief states it. No variant satisfies it. Pooled rho for the combos with lad2_adele is not computed by the extra module (only the extra metrics below).

### Extra metrics (metrics_extra.report)
SOPBench new domain, procedure-level rho (CI vs baseline), hard-case recall at 20% (chance 0.2), selective lift at 50% coverage:

| method | proc-rho | paired CI vs adele | hard@20 | lift50 |
|---|---|---|---|---|
| adele | 0.231 | - | 0.281 | 0.033 |
| cat_adele_rub | 0.360 | [-0.041, 0.318] | 0.304 | 0.039 |
| lad2_adele (earlier) | 0.411 | [-0.008, 0.370] | 0.333 | 0.041 |
| grp_proc_adele (earlier) | 0.438 | [-0.040, 0.428] | 0.295 | 0.040 |
| wd_center_adele | 0.309 | [-0.056, 0.207] | 0.283 | 0.032 |
| wd_z_adele | 0.279 | [-0.136, 0.222] | 0.261 | 0.029 |
| wd_center_adele_rub | 0.472 | [-0.016, 0.477] | 0.275 | 0.032 |
| xbench_adele | 0.313 | [-0.020, 0.180] | 0.260 | 0.032 |
| xbench_adele_rub | 0.341 | [-0.078, 0.307] | 0.347 | 0.041 |
| stack4 | 0.310 | [-0.219, 0.337] | 0.235 | 0.023 |
| wd_center_adele_rub + adele | 0.398 | [0.031, 0.312] | 0.290 | 0.036 |
| stack4 + adele | 0.322 | [-0.069, 0.264] | 0.262 | 0.035 |
| wd_center_adele_rub + lad2_adele | 0.482 | [0.044, 0.453] | 0.316 | 0.039 |
| stack4 + lad2_adele | 0.382 | [-0.100, 0.382] | 0.293 | 0.035 |

Label-conditional rho on SOPBench (rho inside fold, domain and allow/refuse cells), new procedures / new domain:
adele 0.130 / 0.142, cat_adele_rub 0.178 / 0.174, wd_center_adele 0.101 / 0.120, wd_z_adele 0.121 / 0.139, wd_center_adele_rub 0.156 / 0.135, xbench_adele 0.074 / 0.109, xbench_adele_rub 0.127 / 0.140, stack4 0.175 / 0.148, lad2_adele 0.190 / 0.173.
Only the procedure-level paired intervals of the two wd_center_adele_rub combos lie above 0 (+0.031 and +0.044 lower ends), a single metric out of many, not the protocol's.

## 4. Diagnostics
- Stacking weights (mean of the NNLS weights over the outer folds, share of folds with weight > 0 in brackets), order ADeLe / rubric / human_time+length / ADeLe+rubric:
  - sopbench (32 fits): 0.079 [0.31] / 0.367 [0.91] / 0.022 [0.19] / 0.623 [1.00], sum about 1.09.
  - tau2 (28 fits): 0.055 [0.21] / 0.066 [0.14] / 0.073 [0.18] / 0.279 [0.61], sum about 0.47.
  So on SOPBench the stack leans on the rubric and the ADeLe+rubric ridge. On tau2 the weights sum to under 0.5, meaning the stack shrinks predictions toward the mean, almost all of tau2's signal being noise. The stack fails in tau2 new domain (rho -0.026), where the inner procedure-grouped folds do not mimic a domain shift.
- Centering inside domains does not help ADeLe (0.117 and 0.098 vs 0.131). With the rubric it is 0.135 vs 0.148 for the plain version, so centering alone gives nothing pooled. A possible reason: the rubric features are per procedure and already differ little between domains, so there is little between-domain variance to remove. I did not test this reason.
- Where centering does show an effect is the procedure level on SOPBench new domain (0.472 vs 0.360 plain, 0.231 baseline), but the CI vs baseline includes 0 and the pooled case-level metric does not move.
- Borrowing helps the rubric variant on tau2 (tau2-dom 0.316, tau2-proc 0.224, against 0.156 and 0.104 plain) and hurts on SOPBench (0.095 and 0.120 vs 0.165 and 0.168). The ADeLe-only borrow also hurts SOPBench (0.062, 0.079 vs 0.120, 0.125). Reading: tau2 has few tasks and gains from the 830 SOPBench rows, SOPBench gets diluted by tau2 rows with a different difficulty structure. This is a post hoc reading from 4 scenarios.
- Noise level: paired CI half-widths are about 0.07 to 0.12, so the +0.058 of xbench_adele_rub over the baseline and the +0.041 over plain cat_adele_rub are inside noise. 5 seeds x 5 folds over the same procedures are not independent samples.

## 5. Verdict
- None of the three tricks passes the improvement rule. Within-domain centering and stacking do not beat their untricked versions (centering about equal, stacking worse, mainly from tau2 new domain).
- Borrowing from other benchmarks with the rubric features has the highest pooled rho (0.189, +0.058 over the baseline, paired CI [-0.115, 0.117]) and the best Brier and log-loss skill, but it comes from tau2 only and is negative on SOPBench versus the plain version. Worth a rerun only as a tau2-specific idea, with more than one split before believing it.
- The plain ridge on ADeLe + rubric (0.148) beats the baseline by 0.017 within noise, as in the rubric screen. Confidence in all of the above: low, the sample cannot resolve differences below about 0.08.
