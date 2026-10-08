# FMEA by an LLM as a case-difficulty feature (ideas round, 4 to 5 Oct 2026)

## 1. Idea and source
An LLM plays a quality engineer and runs a failure mode and effects analysis on one case: it lists up to 8 ways an AI agent could fail on this case, with occurrence, severity and detection from 1 to 10. The risk priority number (RPN = O x S x D) and the counts per failure category become features for the difficulty predictor. Method source: IEC 60812:2018 and Stamatis 2003 (FMEA). Using it to predict LLM agent difficulty is new.

## 2. Setup (fixed before the full run, no variant was added or dropped afterwards)
- Model gemini-3.8-flash on Vertex, temperature 0, thinking "low", max 4000 output tokens. Prompt in prompts.py (category definitions, anchors for 1, 5, 10 on each scale). Output JSON with up to 8 modes, each with one of the 8 fixed categories.
- Input: statements.jsonl text. For tau2 the domain policy (airline_policy.md, retail_policy.md, telecom_main_policy.md) is put at the top. Banking was not run (not needed).
- Calls: 1,108 main calls (830 sopbench, 278 tau2) plus 100 retest calls = 1,208 calls in total, 0 failed parses, 0 retries. Every response is cached in cache/fmea.jsonl.
- Features (15): n_modes, sum_rpn, max_rpn, mean_occurrence, max_occurrence, count per category (8), occurrence-weighted counts of skips_required_check and acts_when_should_refuse. Files features/fmea_<bench>.csv.
- Harness runs (benches sopbench and tau2, schemes new_procedures and new_domain only): fmea (ridge on FMEA alone), fmea_cat_adele (one ridge on FMEA + 18 ADeLe), fmea_grp_adele (grouped ridge, "LLM Judge" grid for both families), combo fmea_cat_adele+adele. metrics_extra additionally with lad2_adele, grp_proc_adele and the combo fmea_cat_adele+lad2_adele.

## 3. Results (pooled within-domain rho over the four new-process scenarios)
Brier and log-loss skill are means over the four scenarios. Verdict by EVALUATION_PROTOCOL section 8 (paired CI of the difference must lie above 0).

| method | pooled rho | 95% CI | worst | sop-proc | sop-dom | tau2-proc | tau2-dom | paired diff vs adele | Brier skill | log-loss skill | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| adele (baseline) | 0.131 | [0.027, 0.255] | 0.120 | 0.120 | 0.125 | 0.128 | 0.150 | - | 0.000 | -0.003 | baseline |
| fmea | 0.062 | [-0.051, 0.205] | 0.039 | 0.039 | 0.041 | 0.083 | 0.088 | [-0.209, +0.079] | -0.017 | -0.022 | not better |
| fmea_cat_adele | 0.096 | [0.017, 0.249] | 0.087 | 0.101 | 0.087 | 0.100 | 0.095 | [-0.104, +0.056] | -0.003 | -0.007 | not better |
| fmea_grp_adele | 0.069 | [-0.024, 0.200] | 0.014 | 0.091 | 0.098 | 0.070 | 0.014 | [-0.139, +0.021] | -0.016 | -0.044 | not better (worse point estimate) |
| fmea_cat_adele+adele | 0.111 | [0.021, 0.253] | 0.095 | 0.113 | 0.110 | 0.095 | 0.125 | [-0.052, +0.026] | 0.000 | -0.003 | not better |
| (reference) lad2_adele | 0.182 | | | | | | | [-0.061, +0.101] | | | not better |
| (reference) grp_proc_adele | 0.172 | | | | | | | | | | not better |

Extra metrics (hard@20 = hard-case recall at 20%, chance 0.2. lift50 = selective lift at 50% coverage. Averages over the four scenarios. Procedure-level rho = SOPBench new domain, baseline 0.231):

| method | procedure-level rho (paired CI vs adele) | label-conditional rho, sop-proc / sop-dom | hard@20 | lift50 |
|---|---|---|---|---|
| adele | 0.231 | 0.130 / 0.142 | 0.281 | 0.033 |
| fmea | 0.316 ([-0.142, +0.275]) | 0.070 / 0.055 | 0.248 | 0.032 |
| fmea_cat_adele | 0.297 ([-0.057, +0.189]) | 0.110 / 0.093 | 0.300 | 0.032 |
| fmea_grp_adele | 0.276 ([-0.102, +0.198]) | 0.074 / 0.062 | 0.266 | 0.027 |
| fmea_cat_adele+adele | 0.255 ([-0.050, +0.110]) | 0.121 / 0.115 | 0.296 | 0.033 |
| fmea_cat_adele+lad2_adele | 0.407 ([+0.019, +0.346]) | 0.178 / 0.165 | 0.309 | 0.035 |
| lad2_adele | 0.411 ([-0.008, +0.370]) | 0.190 / 0.173 | 0.333 | 0.041 |
| grp_proc_adele | 0.438 ([-0.040, +0.428]) | 0.193 / 0.169 | 0.295 | 0.040 |

Reading: every FMEA variant is at or below the baseline in the pooled rho, and the point estimate gets worse when FMEA is added to ADeLe (0.131 to 0.096 in one ridge, 0.069 grouped). The averaged combination with the baseline (0.111) is also below it. The one apparent gain is the procedure-level rho of FMEA alone (0.316 against 0.231), but its paired interval contains zero (this metric has only about 30 procedures per cell). The combination with lad2_adele has an interval above zero at the procedure level ([+0.019, +0.346]), but it is no better than lad2_adele alone (0.411), so the gain comes from lad2_adele, not from FMEA.

## 4. Diagnostics
Test-retest (100 random tasks, same prompt, category list in a different random order, Spearman between the two runs): sum_rpn 0.72, max_rpn 0.57, mean_occurrence 0.60, n_modes 0.55, max_occurrence 0.38. The headline occurrence feature (max_occurrence) is the least stable: a single model rating of "how likely" is mostly noise. sum_rpn is acceptable, mostly because it follows the number of modes.

SOPBench, can the FMEA tell must-refuse from must-act? AUC of the maximum occurrence of the category for the refuse label (1 - should_succeed, analysis only):
| score | overall | mean within domain | range within domain |
|---|---|---|---|
| acts_when_should_refuse | 0.540 | 0.526 | 0.44 (university) to 0.60 (hotel) |
| refuses_when_should_act | 0.465 | 0.485 | 0.40 (online_market) to 0.68 (university) |
| skips_required_check | 0.525 | 0.527 | 0.46 to 0.66 |
| acts minus refuses | 0.550 | 0.520 | 0.28 (university) to 0.59 |
The AUCs are close to 0.5 (the rubric's p_refusal had 0.61 overall). The model's own sense of "an agent might wrongly act" carries almost no information about whether the case really must be refused, and the hidden-state dependence cannot be read from the text. This is why the case-level signal stays weak.

Which single FMEA column tracks the fitted beta inside domains (case-count weighted Spearman across domains, sign as in items.csv):
- SOPBench: best is cnt_wrong_calculation, rho +0.094 (next: cnt_incomplete_information_gathering +0.080, n_modes +0.076, sum_rpn +0.075). All below 0.1. Its linear CV R2 from the 18 ADeLe columns is 0.54 (largest single ADeLe correlation QLq 0.63), so on SOPBench the best column is largely a restatement of ADeLe (quantitative reasoning demand).
- tau2: best is max_rpn, rho +0.167 (cnt_wrong_parameters_or_record +0.165, cnt_skips_required_check -0.136). Its CV R2 from ADeLe is -0.08 (mean absolute correlation with an ADeLe column 0.06, largest 0.14 with CEe), so here it is new information relative to ADeLe, but it is weak and did not survive in the fitted models (tau2 scenarios of fmea_cat_adele: 0.100 and 0.095 against 0.128 and 0.150 for adele).
Full tables: diagnostics.json.

## 5. Not run, caveats
- tauk_banking was not run (optional, not needed). Random-cases scheme not run (battery rule).
- One prompt, one model, one seed per task (plus one reordered repeat for 100 tasks). The retest shows the occurrence ratings are noisy, so a mean over several repeats might do better, but that was outside the budget plan. No variants were tuned on test folds.
- tau2 telecom uses telecom_main_policy.md only, not the tech support manual.

## 6. Verdict
- Does not help. No variant reaches the baseline pooled rho (best FMEA variant 0.111 against 0.131), all paired intervals contain zero, and log-loss skill is worse for the standalone and grouped variants.
- Adds no case-level information about the hidden refuse/act state (AUC 0.52 to 0.55). The only positive sign is the procedure-level rho of FMEA alone (0.316), which is not significant. Fairly sure of the negative on pooled rho (CI upper bound for the gain is +0.056 to +0.079), less sure for the procedure level.
- Business-process framing is natural to a process owner, but the numbers an LLM gives for occurrence are too unstable (retest 0.38) to rank cases.

## 7. Files
Folder research/screen/ideas/fmea/: prompts.py, llmrun.py, build_features.py, run_screen.py, diagnostics.py, cache/fmea.jsonl, features/, retest.json, diagnostics.json, out/run/eval.json, out/run/eval.txt, out/run/extra_metrics.json.
