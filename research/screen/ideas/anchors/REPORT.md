# In-context anchors: show the model what hard looks like

## 1. Idea and source
Give the LLM 8 training cases of the same benchmark with their observed success shares, then ask it to estimate the success share of a new case. LLMs can do regression from in-context examples (Vacareanu et al. 2024, COLM, "From words to numbers"). The earlier zero-shot forecast in our screening gave rho about 0.

## 2. Setup
- Model gemini-3.8-flash via the repo llm.py (Vertex), temperature 0, thinking "low". Responses cached by prompt hash in llm_cache.jsonl.
- Seeds: SEEDS=3 for the whole run, so the baseline in this run also uses 3 seeds (baseline pooled rho is 0.145 here, not the 0.131 of 5 seeds). Comparison is paired within this run. Benchmarks sopbench and tau2, schemes new_procedures and new_domain.
- Predictor (alib.py, AnchorPredictor): in fit, smoothed share (successes+0.5)/(trials+1) of each training task over training agents, from training tasks only. Linear map beta = a + b*(-logit(share)) fitted by least squares on the fold-IRT beta of training tasks. LLM percent (clipped to 1..99) goes through the same map, P = sigmoid(theta - beta_hat). Unparsable answers (4 of 4431 target cases) fall back to knn4_beta.
- Anchors per target: 4 nearest training tasks by cosine on bge-base embeddings, plus one per training-difficulty quartile (nearest inside the quartile, no duplicates), 8 in total, always training tasks. Shown in ascending order of share with their domain. For tau2 the policy of the target domain (airline, retail, telecom main policy) is at the top.
- knn4_beta (no LLM): mean fold-IRT beta of the 4 nearest anchors.
- Calls: 3,370 for the main run (3,373 unique prompts, 3 were test calls) plus 300 for anchors_random, 3,673 in total. No errors.
- Harness run: benches sopbench and tau2, new_procedures and new_domain only, inside cpu_lock. Scripts: alib.py, precompute.py (LLM pass), run.py (harness), diag.py, rand_cmp.py. Outputs: out/eval.json, out/eval.txt, out/extra_metrics.json, diag.json, rand_cmp.json.

## 3. Results (pooled within-domain rho over four new-process scenarios, 3 seeds)
| column | rho | 95% CI | worst | sop-proc | sop-dom | tau2-proc | tau2-dom | paired diff vs adele | Brier skill | log-loss skill | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| adele (baseline) | 0.145 | [0.027, 0.255] | 0.119 | 0.119 | 0.125 | 0.186 | 0.150 | | 0.002 | -0.001 | |
| anchors | 0.075 | [-0.130, 0.256] | -0.048 | 0.070 | 0.085 | -0.048 | 0.194 | [-0.249, +0.084] | -0.148 | -0.129 | not better (worse) |
| knn4_beta | 0.127 | [-0.029, 0.131] | 0.016 | 0.016 | 0.017 | 0.109 | 0.364 | [-0.245, +0.073] | -0.051 | -0.072 | not better |
| anchors+adele | 0.130 | [-0.067, 0.270] | 0.089 | 0.089 | 0.096 | 0.141 | 0.194 | [-0.144, +0.072] | 0.007 | 0.021 | not better |
| knn4_beta+adele | 0.128 | [-0.005, 0.146] | 0.055 | 0.071 | 0.055 | 0.152 | 0.234 | [-0.161, +0.027] | 0.007 | 0.008 | not better |

anchors+lad2_adele has no protocol-evaluator row (lad2_adele lives in another folder); it appears only in the extra metrics below.

Extra metrics (metrics_extra):
| column | SOP new-domain procedure-level rho (CI of diff to adele) | hard-case recall at 20% (mean of 4) | selective lift at 50% |
|---|---|---|---|
| adele | 0.231 | 0.286 | 0.035 |
| anchors | 0.343 [-0.136, +0.364] | 0.230 | 0.044 |
| anchors+adele | 0.365 [-0.050, +0.341] | 0.242 | 0.045 |
| anchors+lad2_adele | 0.413 [-0.050, +0.412] | 0.255 | 0.045 |
| knn4_beta | -0.126 [-0.694, -0.013] | 0.213 | 0.029 |
| lad2_adele | 0.411 [-0.008, +0.370] | 0.335 | 0.042 |
Label-conditional rho on SOPBench: anchors 0.130 (proc) and 0.145 (dom), adele 0.130 and 0.142. So holding the allow/refuse state fixed, the LLM is no better than the baseline.

## 4. Diagnostics (diag.json; raw rows diag_rows.csv)
Spearman, LLM percent vs the mean share of its 4 nearest anchors: 0.41 (SOP new-domain), 0.29 (SOP new-proc), 0.28 (tau2 new-domain), 0.27 (tau2 new-proc). It partly follows neighbours but is not a copy.
Does the LLM add beyond neighbours? Within-domain rho of LLM percent vs true share (all-agent share, seed 0 average over seeds): SOP new-domain 0.148 vs neighbours -0.036, SOP new-proc 0.196 vs 0.127, tau2 new-domain 0.254 vs 0.273, tau2 new-proc 0.348 vs 0.203. In the harness (rho against fitted beta, the protocol metric) the order is reversed on tau2 new-domain (knn4_beta 0.364 vs anchors 0.194) and on SOP knn4_beta is near 0 while anchors is 0.07 to 0.09. The two rho versions differ in target (raw share vs fitted beta) so they are not identical numbers, and the gap is unexplained beyond that.
Inside one SOPBench procedure (50 procedures, mean Spearman LLM vs true share): 0.078 (new-domain), 0.134 (new-proc). Neighbours: -0.032 and 0.117. The LLM separates cases of one procedure only weakly.
Calibration of raw percentages (mean predicted vs observed share by decile, SOP new-domain): predictions 16.5 to 77.9 versus observed 40.9 to 61.1. The LLM overspreads: low deciles are far too pessimistic (16.5 vs 40.9), which explains the strongly negative Brier and log-loss skills of the raw `anchors` column (-0.148, -0.129). On tau2 predictions are too low in every decile (e.g. 18 vs 68 in the lowest). The map a + b*(-logit p) is fitted on training shares, not on LLM outputs, so it does not correct this. Combining with adele (logit average) repairs the scores (Brier skill +0.007, log-loss skill +0.021) but not the rho.
anchors_random (SOPBench new-domain, 297 of 300 parsed, 300 calls, same 300 target cases): within-domain rho against full-data IRT b (difficulty direction) similar anchors 0.105, random anchors -0.066, pooled 0.238 vs 0.112. Correlation between the two LLM answers 0.53. Similarity-chosen anchors matter, but the sample is small and no CI was computed.

## 5. Verdict
- No. Anchored LLM estimates do not beat the baseline: pooled rho 0.075 (paired diff to adele [-0.249, +0.084]), worse calibrated, worse on tau2 new-procedures (-0.048). The combination with adele (0.130) is also not better.
- Where it shows a hint: procedure-level rho on SOPBench new-domain (0.343, 0.365 with adele, 0.413 with lad2_adele) but all paired CIs include 0 and lad2_adele alone already gives 0.411. Beyond copying neighbours the LLM adds a little within domain on SOPBench (diag), nothing reliable on the protocol metric.
- Confidence: moderate that there is no gain at the +0.08 level needed for significance (CIs wide, 3 seeds). Not run: 5 seeds, other anchor counts, an LLM-output calibration map.
