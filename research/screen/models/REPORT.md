# Models screen: new model parts on the baseline's ADeLe features

Date: 4 Oct 2026. Folder: prototype/screen/models/. No LLM calls. Same folds, fold IRT, metrics and bootstrap as the baseline (harness.py); the baseline column `adele` reproduces pooled rho 0.131.

## 1. What was run

Only the model part (features -> difficulty -> P) was changed. Features are the 18 ADeLe levels (data/<bench>/features/adele.csv) unless stated. All inner cross-validation uses training tasks of the fold only, grouped by procedure (GroupKFold, 5 folds, shuffle, seed 42), unlike the baseline's plain inner KFold; on banking (one task per procedure) it is the same thing. All benchmarks (sopbench, tau2, tauk_banking) and all schemes (random_cases, new_procedures, new_domain; banking only random_cases) were run, 5 seeds.

1. **Random forest and gradient boosting** (`rf_adele`, `gb_adele`; Krsteski and Meyer 2026: tree regressors from features to fold-IRT difficulty, P = sigmoid(theta - beta_hat)). XGBoost is not installed in the repo venv, so sklearn HistGradientBoostingRegressor is used. Grids chosen by inner CV (MSE): RF 200 trees, min_samples_leaf {5,20,50} x max_features {0.33,1.0}; GB learning rate 0.05, max_depth {2,3} x max_iter {50,150} x min_samples_leaf {20,50}. Deviation: their exact model settings are not known to me, the grid is my choice.
2. **Pairwise ranking loss** (`rank_adele`). Linear scorer on standardised ADeLe, trained with logistic loss on pairs of training tasks of the same domain (order from fold-IRT beta; at most 20,000 random pairs per fit, symmetrised, no intercept). L2 strength C in {1e-4,...,1} chosen by inner-CV pairwise accuracy on held-out training tasks. The score is mapped to the beta scale by a linear fit on training tasks (beta ~ a + c*score), then P = sigmoid(theta - beta_hat).
3. **Multidimensional IRT from ADeLe** (`mirt_adele_d2`, `mirt_adele_d4`). The repo's AmortizedMIRT (methods.py, IRT-Router-style) with the ADeLe source instead of the bge embedding, 2 and 4 dimensions, repo defaults (PCA keeps all 18 components, weight decay 0.01, 600 epochs, seed 0). Trained jointly on the training responses; no tuning.
4. **ADeLe-style per-agent assessor** (`agent_lr_adele`; Zhou et al. 2026, section 3.4). For each agent a separate L2 logistic regression from standardised ADeLe levels to that agent's success on training tasks (binomial counts for tau2 and banking). The penalty is shared across agents and chosen by inner CV log loss over {0.3,3,30,300,3000} (sum-loss scale), a deviation from fully separate tuning to keep it stable. P comes straight from the agent's model; the IRT ability is not used. Difficulty for the rank metric = minus mean logit of P over agents. **Shrunk version** (`agent_shrunk_adele`): pooled logistic with agent intercepts (penalty in {1,10,100}), then per-agent deviations (intercept and coefficients) fitted with the pooled logit as offset and a stronger L2 in {10,100,1000,1e5}; both penalties chosen jointly by inner CV log loss.
5. **LLTM with extra features** (`lltm_adele_ht`, `lltm_adele_ht_len`). The repo's AmortizedIRT (joint fit with abilities) on ADeLe + log human minutes, and ADeLe + human time + log characters. The existing `lltm_adele` is in the repo reference results (pooled 0.113) and is not rerun.
6. **Combinations.** `ridge_adele_ht`, `ridge_adele_ht_len`: the baseline's ridge on concatenated features. `grp_adele_ht`: their GroupedRidgePredictor, ADeLe with the "LLM Judge" grid [0.01,0.1,1,10], human time with the "Embedding" grid [100,1000,10000]; `grp_adele_ht_lowgrid`: human time also on the LLM Judge grid (the Embedding grid shrinks a single column almost to zero). Equal-weight logit averages `X+adele` for every new method, and `mirt_adele_d2+ridge_adele_ht+adele` (the best two new standalone models by pooled rho, with the baseline). Caveat: that choice of two was made after seeing the test-fold numbers, so it is exploratory.

## 2. Results

Pooled over the four new-process scenarios (SOPBench and tau2, new procedures and new domain). Columns: rho_mean, bootstrap 95% CI, worst scenario, rho by scenario (sop new proc / sop new domain / tau2 new proc / tau2 new domain), paired 95% CI of the difference to the baseline, mean Brier skill and log-loss skill over the four scenarios, verdict.

Verdict rule (EVALUATION_PROTOCOL section 8): "better" needs criterion 1 (higher mean, paired CI above zero), 2 (no scenario with paired CI entirely below zero), 4 (Brier and log-loss skill not lower than the baseline's). Criterion 3 (across-benchmark test) is not supported by the harness and was not run; criterion 5 (frozen confirmation on a held-out benchmark) was not run. So no method could be called "better" in full even if 1, 2, 4 passed; none passed 1.

| method | rho_mean | 95% CI | worst | by scenario | minus baseline, paired 95% CI | Brier skill | log-loss skill | verdict |
|---|---|---|---|---|---|---|---|---|
| agent_lr_adele+adele | +0.135 | [+0.027, +0.252] | +0.120 | +0.120 / +0.130 / +0.133 / +0.155 | [-0.013, +0.014] | +0.031 | +0.043 | not better |
| mirt_adele_d2+adele | +0.133 | [+0.029, +0.250] | +0.123 | +0.123 / +0.132 / +0.153 / +0.124 | [-0.030, +0.043] | +0.023 | +0.022 | not better |
| mirt_adele_d2+ridge_adele_ht+adele | +0.132 | [+0.024, +0.252] | +0.122 | +0.122 / +0.134 / +0.146 / +0.125 | [-0.028, +0.040] | +0.017 | +0.019 | not better |
| mirt_adele_d4+adele | +0.131 | [+0.016, +0.244] | +0.122 | +0.122 / +0.127 / +0.146 / +0.130 | [-0.037, +0.030] | +0.025 | +0.033 | not better |
| adele | +0.131 | [+0.027, +0.255] | +0.120 | +0.120 / +0.125 / +0.128 / +0.150 |  | +0.000 | -0.003 | baseline |
| ridge_adele_ht+adele | +0.131 | [+0.021, +0.258] | +0.119 | +0.119 / +0.124 / +0.133 / +0.148 | [-0.010, +0.013] | +0.001 | -0.003 | not better |
| gb_adele+adele | +0.128 | [+0.054, +0.251] | +0.086 | +0.086 / +0.111 / +0.127 / +0.186 | [-0.030, +0.055] | +0.006 | +0.005 | not better |
| mirt_adele_d2 | +0.127 | [+0.021, +0.244] | +0.112 | +0.120 / +0.130 / +0.146 / +0.112 | [-0.045, +0.038] | +0.024 | +0.007 | not better |
| ridge_adele_ht | +0.126 | [+0.026, +0.256] | +0.116 | +0.116 / +0.122 / +0.121 / +0.145 | [-0.018, +0.021] | -0.000 | -0.003 | not better |
| rf_adele+adele | +0.125 | [+0.043, +0.241] | +0.102 | +0.107 / +0.102 / +0.132 / +0.162 | [-0.064, +0.064] | +0.008 | +0.005 | not better |
| agent_lr_adele | +0.125 | [+0.024, +0.248] | +0.102 | +0.117 / +0.129 / +0.102 / +0.150 | [-0.031, +0.025] | +0.047 | +0.064 | not better |
| mirt_adele_d4 | +0.123 | [+0.009, +0.243] | +0.104 | +0.121 / +0.126 / +0.140 / +0.104 | [-0.050, +0.033] | +0.020 | +0.022 | not better |
| agent_shrunk_adele+adele | +0.123 | [+0.027, +0.235] | +0.115 | +0.122 / +0.132 / +0.115 / +0.122 | [-0.037, +0.023] | +0.027 | +0.039 | not better |
| ridge_adele_ht_len+adele | +0.122 | [+0.009, +0.257] | +0.112 | +0.112 / +0.119 / +0.131 / +0.126 | [-0.027, +0.012] | +0.005 | +0.002 | not better |
| grp_adele_ht+adele | +0.120 | [+0.013, +0.234] | +0.108 | +0.121 / +0.126 / +0.125 / +0.108 | [-0.045, +0.021] | -0.004 | -0.009 | not better |
| grp_adele_ht_lowgrid+adele | +0.119 | [+0.013, +0.237] | +0.106 | +0.119 / +0.127 / +0.123 / +0.106 | [-0.048, +0.023] | -0.004 | -0.008 | not better |
| grp_adele_ht | +0.116 | [+0.011, +0.235] | +0.104 | +0.121 / +0.119 / +0.121 / +0.104 | [-0.057, +0.024] | -0.018 | -0.028 | not better |
| lltm_adele_ht+adele | +0.115 | [+0.014, +0.233] | +0.100 | +0.114 / +0.127 / +0.118 / +0.100 | [-0.044, +0.020] | +0.017 | +0.029 | not better |
| agent_shrunk_adele | +0.111 | [+0.018, +0.234] | +0.092 | +0.121 / +0.129 / +0.102 / +0.092 | [-0.042, +0.024] | +0.033 | +0.051 | not better |
| ridge_adele_ht_len | +0.111 | [+0.003, +0.255] | +0.106 | +0.106 / +0.107 / +0.116 / +0.113 | [-0.041, +0.021] | +0.006 | +0.003 | not better |
| grp_adele_ht_lowgrid | +0.108 | [+0.008, +0.231] | +0.083 | +0.117 / +0.119 / +0.112 / +0.083 | [-0.061, +0.022] | -0.018 | -0.027 | not better |
| rank_adele+adele | +0.108 | [+0.006, +0.219] | +0.091 | +0.115 / +0.109 / +0.116 / +0.091 | [-0.057, +0.004] | -0.000 | -0.008 | not better |
| lltm_adele_ht | +0.098 | [+0.000, +0.230] | +0.072 | +0.110 / +0.117 / +0.091 / +0.072 | [-0.061, +0.019] | +0.023 | +0.032 | not better |
| lltm_adele_ht_len+adele | +0.094 | [-0.014, +0.223] | +0.056 | +0.106 / +0.114 / +0.101 / +0.056 | [-0.062, +0.010] | +0.022 | +0.034 | not better |
| gb_adele | +0.089 | [+0.016, +0.228] | +0.042 | +0.055 / +0.104 / +0.042 / +0.155 | [-0.138, +0.092] | +0.001 | -0.002 | not better |
| rf_adele | +0.086 | [-0.003, +0.233] | +0.043 | +0.065 / +0.089 / +0.043 / +0.148 | [-0.159, +0.102] | +0.009 | +0.004 | not better |
| rank_adele | +0.084 | [-0.010, +0.198] | +0.071 | +0.100 / +0.092 / +0.072 / +0.071 | [-0.087, -0.008] | -0.010 | -0.025 | worse |
| lltm_adele_ht_len | +0.075 | [-0.031, +0.216] | +0.024 | +0.096 / +0.104 / +0.075 / +0.024 | [-0.084, +0.012] | +0.017 | +0.036 | not better |

Baseline Brier and log-loss skill are about zero: its probabilities are no better than the constant's.

Key reading:
- No method is better than the baseline. The highest pooled value is `agent_lr_adele+adele` (0.135, difference -0.013 to +0.014); every paired interval contains zero except `rank_adele`, which is worse (-0.087 to -0.008).
- Trees (RF 0.086, GB 0.089) are worse in the mean and in two scenarios (sop new proc, tau2 new proc ~0.04 to 0.07) but better on tau2 new domain (0.148, 0.155 vs 0.150 for the baseline, i.e. equal); averaging with the baseline does not help.
- MIRT on ADeLe (d2 0.127, d4 0.123) is level with the baseline and with the earlier LLTM (0.113); the number of dimensions makes no difference.
- Per-agent assessors rank like the baseline (0.125, shrunk 0.111) but have clearly higher Brier and log-loss skill (+0.047 / +0.064 for the plain one) because they model each agent's own success rate, not only a difficulty shift. This is a calibration gain, not a ranking gain; the improvement rule is on ranking first, so it does not qualify.
- Adding human time or length to ADeLe does not help ranking (ridge 0.126 and 0.111; grouped ridge 0.116 and 0.108; LLTM 0.098 and 0.075). Human time alone was 0.107 in the repo.
- Everything sits inside the baseline's noise band (about +-0.03 paired). The pooled CIs for each method are around 0.2 wide.

## 3. What could not be run

- XGBoost: not installed in the repo venv; sklearn HistGradientBoosting used instead.
- Across-benchmark scenario (criterion 3) and the frozen held-out confirmation (criterion 5): not in the harness; not run. Verdicts therefore cover criteria 1, 2, 4 only.
- Banking enters only through random_cases (single domain, one task per procedure) and is not in the pooled number.
- Not tuned on test folds: all hyperparameters are chosen by inner CV on training tasks. The only post-hoc choice is the pair of models in the three-way average.
- About 30 columns were compared against one baseline without multiplicity correction; a lone positive would need confirmation.

## 4. Paths

- Code: prototype/screen/models/mlib.py (tree, ranking, per-agent models), run_group.py (predictor groups), mkfeat.py (concatenated feature csv in features/), merge_eval.py (merge groups, score, combos), table.py (this table).
- Outputs: prototype/screen/models/out/<group>/<bench>/obs_*.csv.gz; merged and scored: out/all/ (eval.json, eval.txt, obs per bench); table.md; logs/.
- Run: PYTHONHASHSEED=0 .../.venv/bin/python run_group.py <trees|rank|irt|ridge|agent|shrunk> <bench>, then merge_eval.py.
