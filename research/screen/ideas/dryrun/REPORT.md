# Dry run of the case against the policy: report

## 1. Idea
An LLM plays a compliance officer. For each case it reads the customer request next to the policy and, without database access, marks each relevant policy condition as satisfied, violated, needing system data or not applicable, predicts the outcome, and counts lookups. The resulting numbers are used as case-level features for the baseline predictor. Source: new for this project. The decision-entropy part follows the idea of semantic entropy (Farquhar et al. 2024, Nature) applied to the final decision only.

## 2. Setup (fixed before the full run)
- Model gemini-3.8-flash via Vertex, thinking level "low" (chosen on 10 pilot tasks only: all 10 JSON outputs valid and sensible; medium was not tried).
- Inputs. SOPBench: statement text (contains policy and customer message). tau2: domain policy (airline_policy, retail_policy, telecom_main_policy plus telecom_tech_support_workflow, manual skipped) plus statement. tauk_banking skipped.
- Main call at temperature 0 (JSON: conditions with status, outcome probabilities, n_lookups, n_customer_facts, ambiguous_request, ambiguous_policy). Prompt gives a definition per status, one tiny example, and states that the model has no database and must answer needs_system_data when the request alone does not decide a condition.
- 4 extra calls per task at temperature 1.0 that return one word (perform / refuse / transfer_or_other). `generate_t` in lib.py is a copy of llm.generate with a temperature argument, the repo is untouched.
- Features (11): p_refuse, p_perform (renormalised), n_conditions, n_violated_by_request, n_needs_system_data, n_lookups, n_customer_facts, ambiguous_request, ambiguous_policy, decision_entropy (5 decisions: main argmax plus 4 samples), refuse_share_sampled (share of "refuse" among those 5). Files: features/dry_sopbench.csv, features/dry_tau2.csv.
- LLM calls: 4,994 (cap 6,000). 1,108 tasks, but 109 tasks have a prompt identical to another task, so those share one cached call. Parse failures: 0 on both benchmarks (no imputation needed). One main call needed a retry. Sampled decisions: perform 88%, refuse 11%, other 1%.
- Harness: benches sopbench and tau2, schemes new_procedures and new_domain only (banking and random cases not run), one run, cpu_lock("dryrun"). Variants fixed before the run: `dry` (ridge on 11 features), `dry_cat_adele` (one ridge on dry plus 18 ADeLe), `dry_grp_adele` (H.grouped, both families on the "LLM Judge" grid), `lad2_adele_dry` (formula ladder L2 plus ADeLe with the dry columns added to the global features, ladder.py copied unchanged), combos `dry_cat_adele+adele` and `dry_cat_adele+lad2_adele`.

## 3. Results (pooled within-domain rho over the four new-process scenarios)
Scenarios in order: sop-proc, sop-dom, tau2-proc, tau2-dom. Skills are means over the four scenarios. Verdict by section 8: a gain needs the paired CI of the difference above 0.

| method | rho | 95% CI | worst | by scenario | paired CI vs adele | Brier skill | logloss skill | verdict |
|---|---|---|---|---|---|---|---|---|
| adele (baseline) | 0.131 | [0.027, 0.255] | 0.120 | .120 .125 .128 .150 | | 0.000 | -0.003 | |
| dry | -0.016 | [-0.155, 0.254] | -0.217 | .130 .122 -.217 -.098 | [-0.294, 0.075] | -0.035 | -0.028 | worse / not better |
| dry_cat_adele | 0.146 | [0.022, 0.288] | 0.008 | .233 .210 .008 .132 | [-0.054, 0.089] | -0.018 | -0.017 | not better |
| dry_grp_adele | 0.189 | [0.087, 0.307] | 0.166 | .219 .206 .166 .166 | [-0.008, 0.125] | -0.047 | -0.062 | not better (closest) |
| dry_cat_adele+adele | 0.160 | [0.051, 0.282] | 0.090 | .201 .208 .090 .140 | [-0.017, 0.075] | 0.000 | +0.005 | not better |
| lad2_adele_dry | 0.172 | [0.077, 0.270] | 0.134 | .209 .188 .134 .157 | [-0.048, 0.111] | +0.046 | +0.064 | not better |
| lad2_adele (earlier, reference) | 0.182 | | | | [-0.061, 0.101] | | | |

dry_cat_adele+lad2_adele: pooled rho was not printed by the harness (the second column comes from another folder). Its extra metrics are below.

Extra metrics (metrics_extra, SOPBench procedure-level rho is on the new-domain split):

| method | procedure-level rho | paired CI vs adele | hard-case recall at 20% (chance 0.2) | selective lift at 50% |
|---|---|---|---|---|
| adele | 0.231 | | 0.281 | 0.033 |
| dry | 0.244 | [-0.373, 0.368] | 0.226 | 0.016 |
| dry_cat_adele | 0.455 | [+0.005, +0.416] | 0.281 | 0.034 |
| dry_grp_adele | 0.433 | [-0.020, 0.390] | 0.287 | 0.039 |
| lad2_adele_dry | 0.466 | [-0.004, 0.461] | 0.308 | 0.038 |
| lad2_adele | 0.411 | [-0.008, 0.370] | 0.333 | 0.041 |
| grp_proc_adele | 0.438 | [-0.040, 0.428] | 0.295 | 0.040 |
| dry_cat_adele+adele | 0.338 | [-0.035, 0.270] | 0.288 | 0.036 |
| dry_cat_adele+lad2_adele | 0.473 | [+0.056, +0.437] | 0.306 | 0.040 |

Label-conditional rho on SOPBench new procedures (hidden allow/refuse state held fixed): adele 0.130, dry 0.084, dry_cat_adele 0.197, dry_grp_adele 0.181, lad2_adele_dry 0.194, lad2_adele 0.190. Full per-scenario values are in out/run/extra_metrics.json.

Only two paired CIs sit above 0, and both are for the procedure-level metric on the SOPBench new-domain split (few procedures per domain, so noisy): dry_cat_adele [+0.005, +0.416] and dry_cat_adele+lad2_adele [+0.056, +0.437]. They are not the protocol's primary metric and lad2_adele alone already has 0.411, so the dry features add about +0.04 to +0.06 there.

## 4. Diagnostics (label = 1 - should_succeed, evaluation only; diagnostics.py, diagnostics.json)
SOPBench refuse AUC (overall / mean within domain):

| score | overall | within domain |
|---|---|---|
| p_refuse | 0.638 | 0.625 |
| refuse_share_sampled | 0.574 | 0.560 |
| n_violated_by_request | 0.539 | 0.551 |
| decision_entropy | 0.529 | 0.517 |
| rubric p_refusal (earlier) | 0.606 | 0.571 |

The dry run predicts the label slightly better than the rubric (0.64 vs 0.61 overall, 0.62 vs 0.57 within domain), but far from useful. 65% of SOPBench cases are refuse cases, yet the model leans to "perform" (mean p_refuse 0.35, sampled refuse share 0.10). The reason is visible in the conditions: it marks 3.5 of 3.7 conditions per case as needs_system_data, because the hidden database decides most conditions.

Violated-by-request per domain (share of cases with at least one violated condition / of those, share truly refuse / share of all refuse cases caught):
bank 0.022 / 1.00 / 0.035. dmv 0.000 / none / 0. healthcare 0.185 / 1.00 / 0.291. hotel 0.041 / 0.875 / 0.056. library 0.045 / 1.00 / 0.071. online_market 0.047 / 1.00 / 0.071. university 0.000 / none / 0. Overall 5.4% of cases flagged, 97.8% of those are truly refuse, recall 8.1% (base refuse rate 65.3%). So the flag is precise but covers almost nothing: the request text alone rarely reveals the violation, which is the expected limit. Only healthcare gives a sizeable share.

tau2: a clean gold-derived label "correct outcome is refusal or transfer" is possible only partly. I defined it as: gold action list is empty, or contains transfer_to_human_agents. Statements were matched to task files by reason_for_call and known_info. Airline: 8 positives of 50, retail: 6 of 114, matched uniquely. Telecom: 2,285 tasks with many near-identical scenario texts, 32 have a gold transfer, but only 85 of our 114 telecom statements match unambiguously and none of those is positive, so telecom is only usable as negatives. Results (small positive counts, treat as indicative): airline plus retail pooled AUC p_refuse 0.691, refuse_share_sampled 0.700, n_violated_by_request 0.595. By domain: airline 0.699 / 0.760 / 0.621, retail 0.618 / 0.529 / 0.538. Adding the 85 matched telecom negatives: p_refuse 0.777, sampled share 0.731, violated 0.651. Caution: that overall number is inflated by easy telecom negatives. Positives are only 14.

Reliability: the 4 sampled decisions at temperature 1 agree with each other in most cases (mean entropy 0.05 to 0.06 nats); entropy is nonzero only for a minority of tasks and has little relation to the label (AUC 0.53). The temperature-0 main call was run once, so no test-retest of the main JSON.

## 5. Verdict
- Does it help: not by the primary metric. The best variant (dry_grp_adele, 0.189) is +0.058 over the baseline with a paired CI [-0.008, +0.125]; not significant and below the earlier lad2_adele (0.182) only by noise. The dry features alone fail to transfer (-0.016).
- Where there is a signal: SOPBench at procedure level (new domain) and in label-conditional rho, where adding the dry columns raises rho from 0.13 to about 0.19, similar to the earlier rubric; the dry-run columns do not add beyond lad2_adele in the ladder (0.172 vs 0.182).
- Why: with no database, the model cannot see the case state; it marks almost every condition needs_system_data, and the refuse label AUC stays near 0.62 to 0.64. Confidence: moderate for "not better", low for any small gain (CIs wide, 4 scenarios, tau2 has few positives).

## 6. Not run
Banking, random-case scheme, a second thinking level, a repeat of the main call (test-retest), medium thinking. The "L2 + ADeLe + dry" ladder was run (lad2_adele_dry), so no second run was left.

## 7. Paths
Folder: research/screen/ideas/dryrun/. Scripts: lib.py (prompts, cached calls), fullrun.py, build_features.py, run_screen.py, diagnostics.py, ladder.py (copy). Cache: cache/calls.jsonl. Features: features/dry_*.csv, features/concat/. Results: out/run/eval.json, eval.txt, extra_metrics.json; diagnostics.json.
