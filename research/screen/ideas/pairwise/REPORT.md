# Compare, do not score: pairwise LLM judgment plus Bradley-Terry (5 Oct 2026)

All numbers use the shared harness and the baseline protocol (baseline `adele` reproduced: pooled rho 0.131 [0.027, 0.255]). Benches sopbench and tau2 only, schemes new_procedures and new_domain only (banking and random cases skipped to save calls and battery).

## 1. The idea

Ask an LLM which of two cases from the same domain an agent is more likely to handle wrongly, many times, and fit a Bradley-Terry model to get one difficulty score per case. Sources: Thurstone 1927, Bradley and Terry 1952, adaptive comparative judgement (Pollitt 2012), pairwise ranking prompting (Qin et al. 2024, Findings of NAACL), LLM comparative assessment (Liusie et al. 2024, EACL). The application to agent-task difficulty is new here.

## 2. Setup (fixed before the full run)

- Model: gemini-3.8-flash on Vertex, temperature 0, thinking low. Inputs are statements.jsonl texts only. For tau2 the domain policy (airline_policy.md, retail_policy.md, telecom_main_policy.md) is put once at the top of the prompt. The question is the one in the task description, answer JSON {"harder": "A"|"B", "confidence": 0.5..1}.
- Pairs (pairs.py, seed 20261005): 4,336 primary pairs inside domains, about 7.8 comparisons per case. SOPBench: 1,502 same-procedure pairs and 1,758 cross-procedure pairs of the same domain (small procedures with 5 or fewer cases are fully paired, so same-procedure share is below half for them and cross pairs fill up to degree 8). tau2: 1,076 random same-domain pairs. A/B order random, recorded. Plus 1,000 order-swapped repeats of random primary pairs for reliability and position bias. Total 5,336 pairs, 5,234 API calls (no failed parse, no retry needed), every answer cached in cache.jsonl.
- Fit (fit.py): per domain, P(A harder) = sigmoid(s_A - s_B + gamma), gamma = position bias, L2 penalty 0.1 on the scores, L-BFGS. All cases of a domain are fitted together (texts only, legal transductively). Feature = score z-scored inside the domain. Main variant `bt` uses the confidence as a soft outcome (P(A harder) = conf if the answer is A else 1 - conf). `bt_hard` uses the 0/1 answer. The swapped repeats are not used in the fit.
- Runs (run_screen.py, one harness run): `bt` ridge on the BT score alone, `bt_hard`, `bt_cat_adele` (one ridge on BT + 18 ADeLe), combos `bt+adele`, `bt_cat_adele+adele`. Extra metrics also for `bt+lad2_adele` and `bt_cat_adele+lad2_adele`. No variant was added or dropped after seeing results.
- Data caveat found after the run: tau2 telecom has 114 cases but only 5 distinct statement texts, and SOPBench university has 36 distinct texts for 42 cases, hotel 194 for 195. For identical texts the judge cannot discriminate, so telecom BT scores are mostly noise plus position effects. Airline (50) and retail (114) have all-different texts.

## 3. Results (pooled over the four new-process scenarios)

Order of scenarios: sop-proc, sop-dom, tau2-proc, tau2-dom. Brier and log-loss skill and within-agent AUC are means over the four. Verdict by section 8 of EVALUATION_PROTOCOL (needs the paired interval above zero, criterion 5 not run).

| method | pooled rho | 95% CI | worst | sop-proc | sop-dom | tau2-proc | tau2-dom | paired diff vs baseline 95% CI | Brier skill | log-loss skill | within-agent AUC | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adele (baseline) | +0.131 | [+0.027, +0.255] | +0.120 | +0.120 | +0.125 | +0.128 | +0.150 | - | +0.000 | -0.003 | 0.562 | baseline |
| bt | +0.081 | [-0.018, +0.271] | -0.053 | -0.053 | -0.036 | +0.225 | +0.187 | [-0.125, +0.123] | -0.012 | -0.015 | 0.544 | not better |
| bt_hard | +0.078 | [-0.025, +0.258] | -0.059 | -0.059 | -0.039 | +0.214 | +0.196 | [-0.139, +0.085] | -0.013 | -0.016 | 0.541 | not better |
| bt_cat_adele | +0.126 | [+0.039, +0.268] | +0.079 | +0.106 | +0.079 | +0.204 | +0.114 | [-0.027, +0.072] | -0.009 | -0.014 | 0.555 | not better |
| bt+adele | +0.115 | [+0.045, +0.286] | +0.059 | +0.094 | +0.059 | +0.178 | +0.129 | [-0.027, +0.098] | -0.000 | -0.003 | 0.555 | not better |
| bt_cat_adele+adele | +0.123 | [+0.029, +0.260] | +0.099 | +0.114 | +0.099 | +0.158 | +0.123 | [-0.021, +0.036] | -0.003 | -0.007 | 0.557 | not better |

Extra metrics (extra_metrics.json). Procedure-level rho is SOPBench new domain, paired CI against the baseline. Label-conditional rho is the mean of SOPBench new procedures and new domain.

| method | procedure-level rho [CI] | paired diff vs baseline | label-conditional rho | hard-case recall at 20% | selective lift at 50% |
|---|---|---|---|---|---|
| adele | 0.231 [-0.019, 0.430] | - | 0.136 | 0.281 | 0.033 |
| bt | 0.329 [0.080, 0.525] | [-0.182, +0.359] | 0.022 | 0.231 | 0.028 |
| bt_hard | 0.311 [0.060, 0.514] | [-0.205, +0.340] | 0.018 | 0.217 | 0.030 |
| bt_cat_adele | 0.260 [-0.002, 0.475] | [-0.062, +0.116] | 0.109 | 0.273 | 0.028 |
| bt+adele | 0.245 [0.012, 0.445] | [-0.018, +0.067] | 0.107 | 0.264 | 0.029 |
| bt_cat_adele+adele | 0.235 [-0.001, 0.446] | [-0.053, +0.061] | 0.121 | 0.280 | 0.030 |
| lad2_adele (earlier) | 0.411 [0.177, 0.587] | [-0.008, +0.370] | 0.182 | 0.333 | 0.041 |
| grp_proc_adele (earlier) | 0.438 [0.186, 0.606] | [-0.040, +0.428] | 0.181 | 0.295 | 0.040 |
| bt+lad2_adele | 0.377 [0.142, 0.560] | [-0.033, +0.334] | 0.161 | 0.291 | 0.034 |
| bt_cat_adele+lad2_adele | 0.398 [0.165, 0.589] | [+0.039, +0.326] | 0.177 | 0.293 | 0.037 |

The one paired interval above zero (bt_cat_adele+lad2_adele at procedure level) is almost the lad2_adele result itself (0.411 vs 0.398), so it comes from the rubric ladder, not from the BT score. Protocol-level pooled rho for the lad2 combos was not computed (the protocol evaluator was not run with extra_dirs, as instructed).

## 4. Diagnostics (diagnostics.json)

- Reliability is high. Split-half correlation of BT scores (random halves of the comparisons in each domain, 5 repeats): SOPBench 0.79 (Spearman-Brown 0.885), tau2 0.77 (0.868). Order-swapped repeat: the same case is named harder in 86.8% of 1,000 pairs (SOPBench 89.8%, tau2 77.3%). The scores are stable, so the weak validity below is not noise.
- Position bias is small to moderate. Share of "A is harder" answers: SOPBench 0.530, tau2 0.581. Fitted gamma (soft) 0.06 to 0.15 on SOPBench domains, airline -0.06, retail -0.16, telecom +0.27. Hard-outcome gamma is larger (up to 0.79 university, 2.05 telecom) because telecom is full of identical-text pairs where the judge defaults to A. Mean confidence 0.80.
- Zero-shot check, no training. Spearman of BT score vs the full-data fitted beta inside domains, pooled by pairs: SOPBench 0.028 [-0.165, 0.184] (bootstrap over procedures), per domain bank -0.16, dmv +0.37, healthcare +0.19, hotel -0.16, library +0.38, online_market +0.14, university +0.06. tau2 0.187 [0.004, 0.459] (soft) and 0.196 (hard), per domain airline +0.53, retail +0.45, telecom -0.14. So the judge ranks tau2 airline and retail cases reasonably and fails on SOPBench and telecom. The ridge on top of it (bt, tau2 scenarios 0.225 and 0.187) is the same signal.
- Agreement of the LLM choice with the truly harder case on pairs with |beta difference| > 1: SOPBench 52.7% (n = 2,117), cross-procedure pairs 54.9%, same-procedure pairs 50.1% (chance). tau2 60.1% (n = 679).
- SOPBench label analysis (label used for analysis only). AUC of the BT score for the refuse label inside domain: 0.576 weighted (bank 0.63, dmv 0.50, healthcare 0.56, hotel 0.63, library 0.54, online_market 0.54, university 0.59), close to the rubric's p_refusal (0.57 within domain). Spearman with beta inside each procedure separately (44 procedures with 5 or more cases): weighted mean -0.122, median -0.088, so the BT score does not rank cases of one procedure and is on average slightly inverted. Inside procedure and label cells it is -0.051. Inside domain and label cells it is 0.101.
- Why same-procedure comparisons fail: Fact 1 of the brief says the performed/refused state drives difficulty, and the judge cannot tell it apart from the message alone. The judge's answers on same-procedure pairs agree with the true order exactly at chance.

## 5. Verdict

- Does not help. BT alone is 0.081 [-0.018, 0.271], below the baseline 0.131. With ADeLe the best (bt_cat_adele) is 0.126, paired difference [-0.027, +0.072]. Nothing passes the rule.
- Where it works: tau2 (rho 0.19 to 0.23 alone, bt_cat_adele 0.204 in tau2-proc, better than ADeLe's 0.128 but the paired interval at scenario level is wide), mostly airline and retail where the policy is given. On SOPBench it is about zero or negative, and it fails within procedure, which is exactly the case-level signal the idea was meant to capture.
- How sure: the null on SOPBench is solid (zero-shot test needs no training, bootstrap interval includes zero, reliability 0.88 rules out noise). The tau2 gain is suggestive only (4 domains-by-scenarios, 278 cases, telecom is 5 texts). A tau2-only follow-up with more comparisons on airline and retail would be the cheap next test.

## 6. Not run, and paths

Not run: banking and random-case schemes, a repeat at temperature above 0, adaptive pair selection (non-adaptive design only), criterion 5 confirmation.
Paths (all in research/screen/ideas/pairwise/): pairs.py, pairs.csv, llmpairs.py, cache.jsonl, fit.py, features/bt_*.csv, run_screen.py, out/run/eval.json, out/run/extra_metrics.json, diag.py, diagnostics.json, screen.log.
