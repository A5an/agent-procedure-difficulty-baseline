# Pilot runs instead of none: how much do they help, and who should run them

Scripts: `prep_first.py` (first-attempt outcomes of tau2 from the raw trajectories), `pilot.py` (computation), `analyze.py` (tables, plot, results.json).
Outputs: `results.json`, `tables.txt` (all numbers), `pilot_rho_vs_k.png`, `raw.pkl` (per-case predictions), `run.log`. No LLM calls.

## 1. The idea
A business can afford a few cheap trial runs of a new procedure before deciding. We give k pilot agents one run of every case of the new procedure, treat their outcomes as Rasch evidence about the case difficulty beta, and combine it with a Gaussian prior (the documentation model or a flat one). The choice of pilot follows computerized adaptive testing (Lord 1980, Applications of Item Response Theory; van der Linden and Glas 2010, Elements of Adaptive Testing): the most informative agent is the one with predicted success closest to 0.5. The combination itself is a standard Bayesian update, nothing new.

## 2. Exact setup
- Benchmarks sopbench and tau2, schemes new_procedures (seeds 0 to 4, `run_baseline.splits`) and new_domain. For every fold: `load_dataset_for_fold` with the repository IRT cache (no retraining), same seeding as `run_scheme`. Thetas of the pilots and training betas come from that fold IRT.
- Documentation prior: the baseline ridge (`FeatureBasedPredictor`, `ALPHAS`, 18 ADeLe levels) fitted on training tasks gives beta_hat. Prior sd is the residual sd of an inner 5-fold CV on training tasks. Deviation: the inner folds are grouped by procedure (GroupKFold), which is the honest choice for new procedures and gives a larger sd than a plain KFold. Doc-free prior: mean training beta with the total sd of training betas (so k = 0 gives a constant and rho 0, as for the constant method in the protocol).
- Posterior of beta per test case: Gaussian prior times Rasch likelihood sigmoid(theta_pilot - beta) of the pilot outcomes, posterior mean on a grid of 481 points over [-12, 12], vectorized.
- Pilot outcome: sopbench its single run. tau2: the first attempt only (smallest trial index in the leaderboard trajectories, checked against responses.jsonl: 0 mismatching cells).
- Rules (fixed before the run): weakest, median ability (agents sorted by distance to the median ability of the agents that have the case), strongest, adaptive (per case, the agent with posterior predictive success closest to 0.5, sequentially, updating after each pilot, without replacement). For k > 1 the fixed rules take the k weakest, the k strongest, or the k nearest to the median. k = 0, 1, 2, 3, 5.
- Target: within-domain rank of the smoothed failure share (failures + 0.5)/(runs + 1) over the NON-pilot agents of that case (tau2 uses all four attempts of those agents). rho is computed inside (seed, fold, domain) cells with `evaluate.rank_arrays`, pooled by pairs, mean over seeds. Bootstrap: 500 draws over procedures on seed 0, the evaluator's resampling (same draw for the two schemes of one benchmark), pooled CI from the mean of the four scenario draws.
- Gain over k = 0 is paired and uses the same target: the k = 0 prior is scored against the failure share of exactly the agents left after removing that condition's pilots. This matters because the target changes with the pilots.
- Check that the target is the same quantity: Spearman of the target with the full-data IRT beta (cell-pooled, mean over seeds) at k = 0: sop-proc 1.00, sop-dom 1.00, tau2-proc 0.99, tau2-dom 1.00. It falls only as pilots are removed from the target and only on tau2 (adaptive k = 5: 0.98, 0.99, 0.85, 0.90), because tau2 has just 15 agents. Reproduction check: the k = 0 doc prior scored on the failure-share target gives 0.12, 0.12, 0.13, 0.16 (mean 0.129) against the baseline 0.120, 0.125, 0.128, 0.150 (0.131).
- Correction: a first run of this study had a bug and was replaced. `load_matrices` filled S with NaN and SOPBench has a few missing (agent, case) cells, so the failure share was NaN for those cases, `rank_arrays` returned NaN-poisoned cells that only added pairs to the denominator, and SOPBench rho was biased toward 0 (the target-vs-beta check read 0.60 and 0.42). The log-loss sums were also silently skipping NaN. Fixed by setting S to 0 where N = 0, with asserts that no target or prediction is NaN and that no pilot is a missing cell (Y1 is NaN only where N = 0, and selection is restricted to available agents). All numbers below come from the corrected run. The first-run SOPBench conclusions (small gains, adaptive about equal to median) are superseded.
- Probability check: log-loss of the posterior predictive P(success) = E_post[sigmoid(theta - beta)] for non-pilot agents on test cases, against the prior predictive on the same cells.
- P2 (SOPBench, new procedures): one adaptive pilot on 25% or 50% of the cases of each procedure (at least one case left unpiloted, fixed rng seed [777, seed, fold]). Others are predicted as prior plus the mean posterior shift of the piloted cases of the procedure. Evaluated only on cases without a pilot, against the same prior, target all agents.

## 3. Main result: pooled rho over sop-proc, sop-dom, tau2-proc, tau2-dom
Rows: rho [95% CI] and the paired gain over k = 0 of the same prior on the same target (bootstrap over procedures, 500 draws). Baseline for orientation: 0.131.

| rule | k | doc prior rho | doc gain over k=0 | no-doc rho |
|---|---|---|---|---|
| none | 0 | 0.129 [0.017, 0.255] | 0 | 0.000 |
| weakest | 1 | 0.344 [0.236, 0.437] | +0.213 [0.117, 0.297] | 0.373 |
| weakest | 2 | 0.395 | +0.270 [0.124, 0.395] | 0.418 |
| weakest | 3 | 0.423 | +0.304 [0.182, 0.407] | 0.450 |
| weakest | 5 | 0.532 [0.372, 0.636] | +0.401 [0.277, 0.471] | 0.542 |
| median | 1 | 0.313 [0.163, 0.448] | +0.187 [0.107, 0.249] | 0.326 |
| median | 2 | 0.421 | +0.297 [0.180, 0.364] | 0.458 |
| median | 3 | 0.477 | +0.354 [0.234, 0.428] | 0.503 |
| median | 5 | 0.593 [0.414, 0.681] | +0.476 [0.348, 0.516] | 0.621 |
| strongest | 1 | 0.336 [0.227, 0.478] | +0.208 [0.154, 0.296] | 0.332 |
| strongest | 2 | 0.445 | +0.315 [0.225, 0.400] | 0.469 |
| strongest | 3 | 0.489 | +0.358 [0.249, 0.440] | 0.498 |
| strongest | 5 | 0.572 [0.423, 0.674] | +0.441 [0.321, 0.515] | 0.588 |
| adaptive | 1 | 0.453 [0.350, 0.518] | +0.317 [0.216, 0.386] | 0.476 [0.383, 0.532] |
| adaptive | 2 | 0.583 [0.469, 0.645] | +0.440 [0.297, 0.546] | 0.599 |
| adaptive | 3 | 0.637 [0.544, 0.690] | +0.485 [0.335, 0.586] | 0.666 |
| adaptive | 5 | 0.696 [0.609, 0.732] | +0.539 [0.393, 0.590] | 0.698 [0.599, 0.735] |

The gain of the no-doc prior is over a constant (rho 0), so the useful comparison is section 4.

By scenario (sop-proc, sop-dom, tau2-proc, tau2-dom), doc prior:
- k = 0: 0.12, 0.12, 0.13, 0.16.
- adaptive k = 1: 0.37, 0.47, 0.46, 0.52. k = 5: 0.74, 0.79, 0.61, 0.65.
- weakest k = 1: 0.21, 0.22, 0.45, 0.49. median k = 1: 0.36, 0.42, 0.26, 0.21. strongest k = 1: 0.44, 0.48, 0.19, 0.24.
On tau2 the weakest agent is as good as adaptive and the strongest is poor (it succeeds nearly everywhere). On SOPBench the weakest is the worst and median or strongest are close to adaptive. Full numbers for all rows are in `tables.txt` and `results.json`.

## 4. Value of documentation
Paired comparisons (pooled):
- With no pilot, documentation is worth +0.129 [0.017, 0.255] (doc minus no-doc, same target).
- Free minus doc at the same k stays within +-0.04 from k = 1 on, CIs include 0 (adaptive k = 1: +0.023 [-0.058, +0.089], k = 5: +0.002 [-0.036, +0.049]; weakest k = 1 +0.029, median +0.013, strongest -0.005). Rule for "matches": CI of free minus doc contains 0 and the point estimate is at least -0.03. By this rule documentation is unnecessary from **k = 1 for all four rules**. The CI half-width at k = 1 is about 0.07 to 0.10, so a documentation value of that size at k = 1 cannot be excluded.
- Pilots without documentation beat documentation without pilots (lower CI end of free_k minus doc_0 above 0): **k = 1 for all four rules**: adaptive +0.342 [0.182, 0.467], weakest +0.242 [0.086, 0.371], median +0.200 [0.027, 0.292], strongest +0.204 [0.052, 0.354].

## 5. Which agent
Pooled, adaptive is highest at every k (k = 1: 0.453 against weakest 0.344, median 0.313, strongest 0.336; k = 5: 0.696 against 0.532, 0.593, 0.572). No paired CI between rules was computed and the CIs overlap, so the ranking is suggestive, but the direction is the same in all four scenarios at k = 5. The best fixed agent depends on the benchmark (weakest on tau2, median or strongest on SOPBench), which adaptive selection handles automatically. A blind extreme agent loses a lot on one benchmark. Gains are large on both benchmarks once the pilot is informative.

## 6. Probability check (log-loss of P(success) of non-pilot agents, pooled over the scenarios, prior to posterior)
- Adaptive, doc: k = 1 0.5771 to 0.5593, k = 2 0.5668 to 0.5398, k = 3 0.5548 to 0.5245, k = 5 0.5347 to 0.4941. No-doc: 0.5813 to 0.5642 at k = 1, 0.5421 to 0.4976 at k = 5.
- Median and strongest also improve at every k (k = 5 median 0.5983 to 0.5404). Weakest improves at k = 1 only (0.5838 to 0.5708) and is flat or slightly worse at k >= 3 (k = 5: 0.5563 to 0.5602), since the Gaussian-Rasch posterior is overconfident when all pilots are weak agents.

## 7. P2: pilot on a fraction of the cases of a procedure (SOPBench, new procedures, adaptive k = 1)
Evaluated on the unpiloted cases, doc prior: 25% pilot rho 0.120 against 0.136 prior only (gain -0.016 [-0.071, +0.044]), 50% pilot 0.126 against 0.150 (gain -0.024 [-0.082, +0.078]). No-doc: +0.017 [-0.061, +0.098] and +0.036 [-0.108, +0.113] over a constant. A procedure-level shift does not rank cases inside the procedure, and SOPBench case difficulty is mostly within the procedure. Piloting a share of the cases does not replace piloting every case (P1 k = 1 adaptive on sop-proc: gain +0.243 doc, +0.432 no-doc).

## 8. Caveats
- When pilots are removed from the target it loses agents. At k = 5 on tau2 only 10 non-pilot agents remain and the target-vs-beta Spearman falls to 0.74 to 0.90 for the weakest and adaptive rules, which deflates tau2 rho at k = 5.
- Adaptive pilots differ per case and between priors, so the target also differs between the doc and no-doc adaptive rows (the fixed rules share the target exactly).
- This is a within-benchmark simulation with 15 (tau2) or 28 (SOPBench) agents. Pilot thetas come from fold IRT fitted on training tasks only. A real pilot agent must already be placed on the ability scale.
- Bootstrap: procedures on seed 0, 500 draws (the evaluator uses 1000).
- Decision thresholds ("matches": CI contains 0 and point at least -0.03, "beats": lower CI above 0) were fixed before the run. Nothing was tuned on the test folds.

## 9. Verdict
- One pilot run per case adds +0.19 to +0.32 to pooled rho over documentation alone (adaptive 0.129 to 0.453, paired CI of the gain [0.216, 0.386]), five pilots +0.54 (to 0.696). Both benchmarks gain.
- With one pilot per case the ADeLe documentation prior adds no measurable value (difference -0.01 to +0.03, CI about +-0.08), so documentation matters only when no pilot is possible.
- Best choice: the adaptive agent (closest to 50% predicted success). A fixed extreme agent is fine on one benchmark and poor on the other. Piloting only part of a procedure's cases does not help.

## 10. Procedure-level rho for the P2 deployment question (SOPBench)
Script `proc_level.py`, numbers in `proc_level.json`. Per procedure: mean predicted difficulty over its cases against mean full-data beta, Spearman inside each domain across procedures (70 procedures), pooled by pairs, mean over seeds. Bootstrap: 500 draws over procedures on seed 0, paired. P2 prediction per case is prior plus the procedure's mean posterior shift from one adaptive pilot on 25% or 50% of its cases (same subsets as section 7). Two truths: "all cases" (mean beta over all cases of the procedure) and "non-pilot" (mean beta over the cases without a pilot only, predictions averaged over the same cases, k = 0 matched on the same cases). The non-pilot truth is the clean one: the all-cases truth contains the pilot agent's own outcomes on the piloted cases, which also feed the prediction. For new_procedures, procedures of different folds are pooled inside a domain, so predictions of different fold models are compared (the protocol avoids this), and the free prior then has fold-specific constants that give it a small spurious rho (0.045). new_domain is the clean scheme (one model per domain).

| scheme | truth | prediction | rho | prior only | gain over prior only (paired CI) |
|---|---|---|---|---|---|
| new_domain | all | doc + P2 25% | 0.466 | 0.231 | +0.235 [-0.152, +0.632] |
| new_domain | all | doc + P2 50% | 0.434 | 0.231 | +0.203 [-0.259, +0.615] |
| new_domain | all | no-doc + P2 25% / 50% | 0.386 / 0.432 | -0.009 | +0.395 [0.088, 0.624] / +0.442 [0.125, 0.681] |
| new_domain | all | doc + P1 adaptive k=1 (all cases) | 0.593 | 0.231 | +0.362 [-0.046, +0.738] |
| new_domain | all | no-doc + P1 adaptive k=1 | 0.534 | -0.009 | +0.543 [0.219, 0.751] |
| new_domain | non-pilot | doc + P2 25% / 50% | 0.286 / 0.060 | 0.549 / 0.377 | -0.263 [-0.591, +0.034] / -0.317 [-0.657, +0.046] |
| new_domain | non-pilot | no-doc + P2 25% / 50% | 0.214 / 0.046 | 0.070 / -0.006 | +0.144 [-0.200, +0.421] / +0.052 [-0.274, +0.349] |
| new_procedures | all | doc + P2 25% / 50% | 0.400 / 0.424 | 0.157 | +0.243 [-0.166, +0.510] / +0.268 [-0.182, +0.522] |
| new_procedures | all | doc + P1 adaptive k=1 | 0.499 | 0.157 | +0.343 [-0.115, +0.493] |
| new_procedures | all | no-doc + P1 adaptive k=1 | 0.581 | 0.045 | +0.536 [0.410, 1.133] (seed-0 interval, mean-over-seeds point) |
| new_procedures | non-pilot | doc + P2 25% / 50% | 0.179 / 0.226 | 0.185 / 0.243 | -0.005 [-0.303, +0.401] / -0.016 [-0.475, +0.208] |
| new_procedures | non-pilot | no-doc + P2 25% / 50% | 0.084 / 0.210 | 0.037 / 0.013 | +0.047 [-0.163, +0.703] / +0.197 [-0.124, +0.801] |

Reading: at procedure level, with the clean (non-pilot) truth, piloting a sample of cases does not improve the ranking of procedures over the documentation prior (gains negative or zero, all CIs include 0, 70 procedures give wide intervals). The apparent gains against the all-cases truth are inflated by the pilot outcomes being part of the truth, and none is significant against the documentation prior. The doc-free prior with pilots reaches roughly the level of documentation alone on the clean truth (0.05 to 0.21 against 0.18 to 0.55), so documentation is not replaced by a 25% or 50% pilot at procedure level. P1 (every case piloted once) at procedure level reaches 0.50 to 0.59 against the all-cases truth, with the same contamination caveat. The k = 0 documentation values on the non-pilot subsets (0.55, 0.38) differ from the full-set value (0.23) because the subsets are small, which shows how noisy the procedure-level metric is.

## 11. Imperfect judge, cheapest pilot, who the adaptive rule picks
Scripts `judge.py` and `judge_analyze.py`, raw predictions in `judge_raw.pkl`, numbers in `results.json` under `judge_error`, `adaptive_k1_pilot_freq_top5` and `small_model_names`. Same folds, thetas, priors and targets as before. The judge is simulated by flipping each observed pilot outcome with probability e (rng seeded by seed, fold, repetition), 5 repetitions averaged for e > 0 (one for e = 0). The target uses the true outcomes of the non-pilot agents. The posterior uses P(observed = 1 | beta) = (1 - e) sigmoid(theta - beta) + e (1 - sigmoid(theta - beta)). "Naive" uses e = 0 in the likelihood (and in the adaptive choice) on the flipped data. Bootstrap CIs here use 200 draws over procedures (seed 0), averaged over the repetitions. Pooled over the four new-process scenarios. Gain is over k = 0 of the same prior on the same target.

Adaptive pilot, documentation prior, pooled rho (gain over k = 0 [CI]):

| judge error e | k = 1 | k = 2 |
|---|---|---|
| 0 | 0.453 (+0.317 [0.222, 0.383]) | 0.583 (+0.440 [0.311, 0.524]) |
| 0.05 | 0.413 (+0.276 [0.193, 0.342]) | 0.530 (+0.392 [0.276, 0.479]) |
| 0.10 | 0.379 (+0.242 [0.162, 0.297]) | 0.485 (+0.348 [0.242, 0.421]) |
| 0.20 | 0.305 (+0.168 [0.101, 0.215]) | 0.382 (+0.243 [0.140, 0.304]) |
| 0.30 | 0.228 (+0.091 [0.038, 0.137]) | 0.271 (+0.132 [0.032, 0.190]) |
| 0.20 naive | 0.305 (+0.168 [0.098, 0.216]) | 0.387 (+0.246 [0.142, 0.312]) |

Doc-free prior (rho, constant prior has 0): k = 1 0.476, 0.419, 0.373, 0.280, 0.188 for e = 0 to 0.3, naive at 0.2 0.280. k = 2: 0.599, 0.542, 0.482, 0.368, 0.233, naive at 0.2 0.378. Per scenario at e = 0.2, k = 1, doc: sop-proc 0.25, sop-dom 0.31, tau2-proc 0.30, tau2-dom 0.36.
- A judge error rate of 0.2 removes about half of the k = 1 gain and 0.3 removes about 70%, but the gain over documentation alone stays positive (CI above 0) at every e tested, including k = 1 at e = 0.3.
- Gain per unit of error is roughly linear: about -0.075 rho per 0.1 of e at k = 1. A second pilot at e = 0.2 (0.382) is worth about the same as one perfect-judge pilot minus 0.07, so two noisy pilots beat one noisy pilot but not one clean one.
- Ignoring e in the likelihood costs nothing measurable (0.305 against 0.305, 0.387 against 0.382, free 0.280 against 0.280). With symmetric flips the ranking barely changes when the noise is not modelled, because the update direction is the same and only its size differs. This does not say anything about asymmetric judge errors, which were not simulated.
- With a noisy judge the documentation prior and no documentation stay close (free minus doc at k = 1: +0.023 at e = 0, -0.04 at e = 0.3; no CI computed for this difference).

Cost-aware rule "cheapest": the weakest-ability agent (per fold IRT) among agents whose names match 2b to 8b (not preceded by a digit or dot), mini, flash, lite, haiku or small. SOPBench: 13 agents (for example qwen2.5-7b, llama3.1-8b, qwen3.5-2b and 4b, gemma-4-e2b and e4b, the mini and flash models). tau2: gemini-3-flash, gpt-4-1-mini, o4-mini. The name rule counts closed small models too, because tau2 has no open-weight small model. It is close to the weakest rule (the weakest agent of a fold is usually in the set), so it behaves similarly (pooled doc rho 0.327 against 0.344 for weakest).

| k = 1, perfect judge | pooled rho | gain over k = 0 | sop-proc | sop-dom | tau2-proc | tau2-dom |
|---|---|---|---|---|---|---|
| adaptive, doc | 0.453 | +0.317 | 0.37 | 0.47 | 0.46 | 0.52 |
| cheapest, doc | 0.327 [0.192, 0.407] | +0.196 [0.075, 0.281] | 0.21 | 0.22 | 0.40 | 0.47 |
| adaptive, no doc | 0.476 | | 0.43 | 0.48 | 0.45 | 0.54 |
| cheapest, no doc | 0.363 [0.238, 0.430] | | 0.26 | 0.29 | 0.42 | 0.48 |

Always piloting with a cheap model loses 0.13 pooled rho (doc) against the adaptive choice, almost all of it on SOPBench (-0.16 and -0.25), where a weak model fails nearly everything. On tau2 the loss is 0.06 and 0.05. The cheapest pilot with a perfect judge (0.327) is about equal to the adaptive pilot with a 20% error judge (0.305), and still gives a significant gain over documentation alone.

Adaptive pilots chosen at k = 1 (documentation prior, e = 0, all test cases of all seeds and both schemes; share of cases):
- SOPBench: gemma-4-e4b-it 19.0%, claude-3-7-sonnet fc 15.4%, gpt-4.1-mini 14.2%, claude-3-7-sonnet thinking 14.0%, qwen3.5-2b 13.3%.
- tau2: gpt-4-1-mini 45.2%, gpt-4-1 41.5%, o4-mini 9.9%, claude-3-7-sonnet 2.6%, gpt-5-2-none 0.6%.
The rule picks agents in the middle of the SOPBench ability range (theta about -0.3 to 0.5) and the three weakest tau2 agents, whose predicted success is closest to 0.5. The choices are spread over many agents, so the rule is not simply "the median agent". On tau2 it chooses cheap-to-mid models, in line with the finding that the weakest agent works as well there.

## 12. Offline cascade simulation (cheap agent, judge, escalation)
Script `cascade.py`, results in `results.json` under `cascade` (all policies, both benchmarks, CIs, paired differences). Cases: new_procedures seed 0 (5 folds) and new_domain (same folds and fold IRT as before), cases of both schemes pooled per benchmark, first attempts on tau2, real outcomes of the agents. Bootstrap: 1000 draws over procedures, same resample for all policies of a benchmark. Judge error flips A's true outcome with probability e (20 repetitions averaged, success and escalation rates are expectations over the repetitions).
- A = the "cheapest" agent of section 11 (weakest-ability agent among the name-matched small models in the fold: typically qwen2.5-7b on SOPBench and gpt-4-1-mini on tau2). B = the strongest agent by fold ability among the agents that have the case.
- C1: escalate to B when the judge reports fail, else accept A. C2: after A's judged verdict update the posterior (documentation prior, Rasch, error-aware likelihood as in P1), accept A if its posterior success is above tau (0.5 or 0.7), else escalate to the agent with the highest posterior success among the 3 strongest. Under a Rasch model that agent is always the strongest, so C2 differs from C1 only in the accept or escalate decision.
- Cost is a placeholder: cost(A) = 1 per case, cost(B) = 10 per escalated case, A always runs first (always B costs 10). Not a real price. "B fails" is the share of escalated cases where B's real outcome is a failure (these would go to a person). The oracle escalates exactly when A truly failed (equal to C1 at e = 0).

SOPBench (success rate with 95% CI, share escalated, relative cost, share of escalated cases where B also fails):

| policy | success | escalated | cost | B fails |
|---|---|---|---|---|
| always A | 0.090 [0.061, 0.126] | 0 | 1.0 | |
| always B | 0.692 [0.634, 0.747] | 1 | 10.0 | 0.31 |
| oracle (= C1 at e = 0) | 0.699 [0.644, 0.754] | 0.91 | 10.1 | 0.33 |
| C1, e = 0.1 | 0.636 [0.586, 0.690] | 0.83 | 9.3 | 0.33 |
| C1, e = 0.2 | 0.576 [0.529, 0.626] | 0.75 | 8.5 | 0.33 |
| C2 tau 0.5, e = 0 / 0.1 / 0.2 | 0.692 / 0.692 / 0.692 | 0.99 / 1.0 / 1.0 | 10.9 / 11.0 / 11.0 | 0.31 |
| C2 tau 0.7, e = 0 / 0.1 / 0.2 | 0.692 / 0.692 / 0.692 | 1.0 | 11.0 | 0.31 |

tau2:

| policy | success | escalated | cost | B fails |
|---|---|---|---|---|
| always A | 0.547 [0.420, 0.687] | 0 | 1.0 | |
| always B | 0.896 [0.817, 0.939] | 1 | 10.0 | 0.10 |
| oracle (= C1 at e = 0) | 0.942 [0.886, 0.971] | 0.45 | 5.5 | 0.13 |
| C1, e = 0.1 | 0.896 [0.847, 0.924] | 0.46 | 5.6 | 0.12 |
| C1, e = 0.2 | 0.855 [0.810, 0.888] | 0.47 | 5.7 | 0.12 |
| C2 tau 0.5, e = 0 / 0.1 / 0.2 | 0.924 / 0.816 / 0.712 | 0.44 / 0.37 / 0.30 | 5.4 / 4.7 / 4.0 | 0.13 / 0.15 / 0.17 |
| C2 tau 0.7, e = 0 / 0.1 / 0.2 | 0.910 / 0.879 / 0.863 | 0.79 / 0.84 / 0.86 | 8.9 / 9.4 / 9.6 | 0.11 / 0.12 / 0.12 |

Paired differences (bootstrap): C1 beats always A by +0.61 (SOPBench, e = 0) and +0.40 (tau2) at every e. C2 minus C1 on tau2: tau 0.5 -0.018 [-0.031, -0.002] (e = 0), -0.081 [-0.148, -0.010] (0.1), -0.143 [-0.277, -0.013] (0.2). tau 0.7: -0.032 [-0.062, -0.015], -0.017 [-0.048, +0.002], +0.008 [-0.021, +0.028]. On SOPBench C2 minus C1: -0.007 at e = 0, +0.056 [0.043, 0.064] at 0.1, +0.115 [0.100, 0.127] at 0.2.

Findings:
- tau2 is where a cascade pays. The perfect-judge cascade keeps 0.942 success (higher than always B, 0.896, because A sometimes succeeds where B fails on a first attempt) at 55% of B's cost. With a 20% error judge C1 gives 0.855 at 57% of B's cost, which is below always B (0.896), though the unpaired intervals overlap, no paired test against always B was run. A cheap model plus a judge buys about 43% savings for roughly 0.04 of success at e = 0.2.
- SOPBench, with the cheapest model as A, there is no cascade to speak of. A succeeds on 9% of the cases, so 91% of cases escalate and the cost is 10.1 against 10 for always B. Judge errors make C1 worse than always B (0.576 at e = 0.2) because a failure that is reported as pass is accepted. The SOPBench gain from C2 is only that it escalates almost everything, which turns it into always B (0.692 at a cost of 11). A cascade there needs a stronger A, which was not tested.
- Our predictor in the loop (C2) does not beat plain C1 in these runs. On tau2 it only helps at tau 0.7 and e = 0.2 (+0.008, not significant) and it costs success otherwise, because the documentation prior thinks the cheap agent often succeeds and accepts it after a reported fail. Its value in this setup is therefore not shown. With one noisy verdict the posterior moves little compared with the prior, and the thresholds were fixed as given, not tuned.
- Escalated cases that B also fails: 31% to 33% on SOPBench, 10% to 17% on tau2. These would go to a person. On SOPBench that is about 30% of all cases (0.91 x 0.33 at e = 0), the same as in always B (31%).
- Caveats: the prices are placeholders. One fold set (seed 0). The simulated judge is symmetric and independent of the case. A is a weak model chosen by the "cheapest" rule, not a tuned choice.
