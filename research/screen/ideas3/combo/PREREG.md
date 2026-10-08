# Pre-registration, agent combo, round 3 (5 Oct 2026). Written before any evaluation.

No LLM calls, no L2 feature. Every input already exists.

## Feature families
| family | columns | level | source | benchmarks |
|---|---|---|---|---|
| ADELE | 18 ADeLe demand levels | L0 | repo data/<bench>/features/adele.csv | sopbench, tau2 |
| RUB | 12 procedure rubric columns (step type counts, depth, refusal conditions, hidden data, p_refusal) | L0 | rubric/features/rub_proc_<bench>.csv | sopbench, tau2 |
| COND | 1 scalar: logit of the written-condition skip model case score (conditions l0 track, pp = 1 - rubric p_refusal). Skip model fitted on TRAINING cases of the fold only, training values are out-of-fold (5 inner folds grouped by procedure) | L0 | conditions/cm.py | sopbench only |
| CMP | 2 columns: len_mean, branch_mean of the compiled programs | L0 | compile_consistency/unit_features.csv via cons_sopbench.csv | sopbench only |
| PLAN | 9 planned-action features (calls, writes, reads, distinct tools, rules, facts, refuse, transfer, ambiguity) | L1 | plan_tools/features/plan3_<bench>.csv (tau2: mean of 3 plans with the real tool list, sopbench: plan_tau one plan) | sopbench, tau2 |

tau2 has no COND and no CMP: its families are ADELE + RUB + PLAN. SOPBench has all five.

## Model
Grouped ridge, one penalty per family, target = fold IRT difficulty b of the training tasks. Each family standardised on the
training tasks, scaled by 1/sqrt(alpha_family), ridge with alpha 1 (same device as the repo's GroupedRidgePredictor). Grid per
family {1, 10, 100, 1000, 10000}, full product, chosen by 5-fold inner CV (MSE, folds grouped by procedure_id) on training tasks of
the fold only. One deviation from the repo grouped ridge: inner folds are grouped by procedure (the repo shuffles tasks, which leaks
procedure-level features into validation). Prediction P(success) = sigmoid(theta_agent - predicted b), as the baseline.

## Variants (3, fixed)
1. `c_L0` (L0 only): grouped ridge on ADELE + RUB + COND + CMP (sopbench), ADELE + RUB (tau2, where L0 offers nothing else).
2. `c_L1` (L0 + L1): c_L0 families + PLAN (tau2: ADELE + RUB + PLAN).
3. `c_lad` (L1, ladder): rubric ladder L2 (step-type counts as step features, 4 global rubric features) with extra global columns
   ADELE + PLAN + CMP (sopbench), ADELE + PLAN (tau2). Ladder uses its own single penalty chosen by inner CV (as lad2_adele). COND is
   NOT included here because it needs a fold-fitted feature and the ladder takes a static table.
Averages with baseline adele are reported as reference only (not part of the decision): c_L1+adele, c_lad+adele.
Harness run: benches sopbench and tau2, schemes new_procedures and new_domain (4 scenarios), seeds and folds as the shared harness.

## Targets
Main: the protocol target (fold IRT b of all agents, all_irt), pooled within-domain rho over the four new-process scenarios.
Secondary: IRT fitted on the top tier only (tiers/refit.py logic, top5 agents), the same three variants refitted, scored on
fail_top5 (and all_irt for reference). Compared to refit adele and refit lad2_adele from tiers/refit/top5.

## Metrics
Pooled rho with 95% CI, worst scenario, four scenario rhos, paired CI vs adele (harness, protocol bootstrap) and vs lad2_adele
(tiers/score.py machinery, bootstrap over procedures, 300 draws), Brier and log-loss skill per scenario, procedure-level rho
(SOPBench new domain) with paired CI, rho_given_label, hard recall at 20%, selective lift at 50%.

## Decision rule (EVALUATION_PROTOCOL.md section 8), applied to each variant
1. Mean primary rho over the four new-process scenarios higher than adele AND paired pooled 95% CI of the difference above 0.
2. No single new-process scenario with a paired CI entirely below 0.
3. No held-out benchmark (sopbench, tau2) where the variant is significantly worse than adele.
4. Brier and log-loss skill not lower than the baseline's in the four new-process scenarios.
5. Blind test: stays open, not evaluated.
The lad2_adele comparison is reported but is not a criterion. No variant is added after results. Bugs are fixed and rerun, and said so.
