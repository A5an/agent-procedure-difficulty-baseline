# LUPI and allow/refuse mixture (learning from how agents failed on training tasks)

## 1. Idea and source
For training procedures we know trajectories and, on SOPBench, the allow/refuse label of each case. A model learns on training tasks to predict these signals from the text, and the predictions become extra inputs for the difficulty ridge (LUPI: Vapnik and Vashist 2009, Neural Networks 22; generalised distillation: Lopez-Paz et al. 2016, ICLR). The second part splits the cases into "must be performed" and "must be refused", fits one difficulty ridge per group and mixes them by a predicted probability of refusal. The mixture is new (the label split is our own).

## 2. Setup
- Harness run: benches sopbench and tau2, schemes new_procedures and new_domain only, 4 new columns plus constant, oracle, adele. One full run, no rerun. No LLM calls. Variants were fixed before the run and all are reported.
- Base features X: 18 ADeLe levels plus 12 rubric columns (rub_proc_<bench>.csv), 30 columns, standardised inside each fold.
- Privileged signals (parse_priv.py, CSVs privileged_sopbench.csv, privileged_tau2.csv, parsed once under cpu_lock, one file at a time).
  - SOPBench, per case averaged over the 28 agents: mean tool calls, mean assistant messages, share of runs that called the target action (user_goal), share with zero tool calls, share dirgraph_satisfied, share action_called_correctly. Flags come from the earlier rescored table pertype/flags.csv.gz, with the saved evaluation as a fallback.
  - tau2, per case averaged over the 15 submissions in responses.jsonl (45 files, 4 trials each, 60 runs per case): mean messages, mean assistant tool calls, share of runs with a transfer tool call, shares of termination reasons. Termination reasons are almost constant (98.9% user_stop). A fixed rule kept only the shares that are nonzero for at least 15 tasks (all four were kept).
- lupi: stage 1 is a ridge from X to each standardised signal (alpha from the repository grid by inner CV grouped by procedure). Stage 2 is a ridge from [X, stage-1 predictions] to the training-fold IRT beta. On training tasks stage 2 uses out-of-fold stage-1 predictions (inner 5-fold, grouped by procedure so that procedure mates do not leak). On test tasks stage 1 is fit on all training tasks. P = sigmoid(theta - beta_hat) with the fold abilities.
- mix2: logistic (L2, C by grouped CV) for P(refuse | X) and two beta ridges, one on training tasks with label perform and one on label refuse. P(success) = pi * sigmoid(theta - beta_refuse) + (1 - pi) * sigmoid(theta - beta_perform). Falls back to a plain ridge on X if a class has fewer than 8 training tasks (never happened).
- lupi_mix2: the two beta ridges use [X, stage-1 predictions]. The logistic still uses X only.
- x_ridge: the repository ridge on the 30 columns of X (reference for this group). Combos lupi+adele, mix2+adele, mix2+lad2_adele are equal-weight logit averages.
- Leakage statement: the allow/refuse label (and every privileged signal) is used only as a training target on training-fold tasks. It is never a feature of a test task. Test-fold signals were read only afterwards, in diag.py, to score the stage-1 predictions.
- tau2 label: the task files give a clean gold-derived label, "the gold action list is empty (the correct outcome is no state-changing action) or contains transfer_to_human_agents". It marks 34 of 278 cases (12.2%, airline 8 of 50 in the full file, retail 6, telecom 32). It is rarer and less pure than the SOPBench label, because an empty action list also covers pure information requests. mix2 therefore uses it on tau2.

## 3. Main table (pooled within-domain rho over the four new-process scenarios)
Scenarios in order: sop-proc, sop-dom, tau2-proc, tau2-dom. CI is the 95% bootstrap interval. "vs base" is the paired 95% interval of the difference to adele. Skills are means over the four scenarios.

| column | pooled rho | 95% CI | worst | by scenario | vs base | Brier skill | log-loss skill | verdict (section 8) |
|---|---|---|---|---|---|---|---|---|
| adele (baseline) | 0.131 | [0.027, 0.255] | 0.120 | 0.120 0.125 0.128 0.150 | | 0.000 | -0.003 | |
| x_ridge | 0.148 | [0.032, 0.261] | 0.104 | 0.165 0.168 0.104 0.156 | [-0.063, +0.076] | 0.015 | 0.009 | not better |
| lupi | 0.151 | [0.039, 0.258] | 0.061 | 0.173 0.209 0.061 0.163 | [-0.063, +0.074] | 0.006 | 0.000 | not better |
| mix2 | 0.140 | [0.013, 0.260] | 0.119 | 0.147 0.149 0.144 0.119 | [-0.069, +0.061] | 0.027 | 0.029 | not better |
| lupi_mix2 | 0.126 | [-0.019, 0.250] | 0.066 | 0.156 0.154 0.066 0.130 | [-0.082, +0.045] | 0.030 | 0.031 | not better |
| lupi+adele | 0.157 | [0.038, 0.271] | 0.123 | 0.173 0.202 0.123 0.130 | [-0.039, +0.069] | 0.012 | 0.009 | not better |
| mix2+adele | 0.141 | [0.018, 0.258] | 0.100 | 0.156 0.159 0.150 0.100 | [-0.047, +0.054] | 0.019 | 0.019 | not better |

mix2+lad2_adele is in the extra-metrics table below (its protocol rho is in out/eval.txt only through the extra report, the pooled combo with an external directory is not formed by the repository evaluator). No variant has an interval above zero. Reference points from the brief: lad2_adele 0.182, grp_proc_adele 0.172.

## 4. Extra metrics (metrics_extra.report)
SOPBench procedure level (new domain, mean predicted vs mean fitted difficulty per procedure, Spearman inside domains) and the selective metrics averaged over four scenarios.

| column | proc-level rho | paired CI vs adele | hard-case recall at 20% | selective lift at 50% |
|---|---|---|---|---|
| adele | 0.231 | | 0.281 | 0.033 |
| x_ridge | 0.360 | [-0.041, +0.318] | 0.304 | 0.039 |
| lupi | 0.516 | [+0.029, +0.531] | 0.262 | 0.036 |
| mix2 | 0.336 | [-0.075, +0.317] | 0.267 | 0.034 |
| lupi_mix2 | 0.373 | [-0.056, +0.347] | 0.249 | 0.037 |
| lupi+adele | 0.453 | [+0.043, +0.399] | 0.282 | 0.037 |
| mix2+adele | 0.353 | [-0.016, +0.283] | 0.277 | 0.035 |
| mix2+lad2_adele | 0.380 | [-0.044, +0.350] | 0.299 | 0.040 |
| lad2_adele | 0.411 | [-0.008, +0.370] | 0.333 | 0.041 |
| grp_proc_adele | 0.438 | [-0.040, +0.428] | 0.295 | 0.040 |

The one interval above zero is lupi at the procedure level (0.516, [+0.029, +0.531]) and lupi+adele (0.453, [+0.043, +0.399]). It is one metric among many on 7 domains and about 100 procedures, and the case-level protocol rho shows no such gain, so we treat it as a lead and not as a result. x_ridge already has 0.360, so part of the jump over adele comes from the rubric columns and not from LUPI. The correct comparison for LUPI is lupi against x_ridge (0.516 vs 0.360, paired interval against x_ridge not computed).

Label-conditional rho (protocol rho inside fold, domain, allow/refuse cells, SOPBench), new procedures / new domain: adele 0.130 / 0.142, x_ridge 0.178 / 0.174, lupi 0.166 / 0.196, mix2 0.173 / 0.173, lupi_mix2 0.178 / 0.152, lad2_adele 0.190 / 0.173. The mixture does not capture the label effect any better than a plain ridge.

## 5. Diagnostics (diag.md has all tables)
Stage-1 predictability on test tasks (SOPBench, Spearman over tasks / R2 on the standardised signal, new procedures):
- mean tool calls 0.63 / 0.41, mean assistant messages 0.60 / 0.36, share zero tool 0.49 / 0.24, share dirgraph 0.38 / 0.13, share action correct 0.29 / 0.04, share called target 0.16 / 0.05.
- So effort signals (length of the trajectory) are predictable from text, but the outcome-like signals (called the target, dirgraph, correct action) are not. Those are the signals that correlate with beta, so the predictable signals carry little new information about beta.
- tau2: mean messages 0.66 / 0.45 over tasks but only 0.06 inside domains (it mainly separates the three domains). Mean tool calls 0.24 / 0.06, transfer share 0.43 / 0.07 and the termination shares are near zero. On new domain several R2 values are negative.

Signal-to-beta correlation inside domains (Spearman, mean over domains):
- SOPBench: share called target +0.66, share dirgraph -0.64, share action correct -0.50, mean tool calls +0.33, mean assistant messages +0.27, share zero tool +0.13. Inside allow/refuse cells the outcome-like signals keep -0.52 (dirgraph) and -0.70 (action correct), so they carry more than the label.
- tau2: mean messages +0.66, mean tool calls +0.44, termination shares +/-0.3 (driven by rare events), transfer share +0.06.

P(refuse) quality (AUC against the label, mean over domains of per-domain AUC):
- SOPBench logistic: 0.593 (new domain), 0.594 (new procedures), pooled 0.581 / 0.605. Rubric p_refusal: 0.571 within domain, pooled 0.606. So the logistic on X is no better than the single rubric column in a meaningful way (+0.02 within domain).
- tau2 (rare label): logistic 0.559 within domain on new domain and 0.355 on new procedures (below chance, noise from 34 positives with grouped folds). Rubric p_refusal 0.676 within domain.

## 6. Verdict
- It does not help at the case level. LUPI gives 0.151 against 0.148 for the plain ridge on the same features, and mixture 0.140. The whole gain over adele (0.131) that shows in x_ridge comes from the 12 rubric columns, not from the privileged signals. No paired interval over adele excludes zero.
- Why: the signals that correlate with beta (called target, dirgraph, action correct) are exactly the ones text cannot predict (R2 near 0.05), and the signals text predicts (trajectory length) correlate weakly with beta. The refusal probability from X stays near AUC 0.59, so the mixture cannot use the label effect (oracle 0.52 to 0.55) that needs reading the individual case.
- Sure about: no case-level gain (intervals all include zero, high confidence). Not sure about: the lupi procedure-level jump on SOPBench new domain (0.516, interval above zero) which needs a replication and a paired comparison with x_ridge before it counts. lupi is worst on tau2 new procedures (0.061), so its case-level behaviour is unstable.

## Files
Scripts: parse_priv.py, lupi_lib.py, run_lupi.py, diag.py (all in this folder). Data: privileged_sopbench.csv, privileged_tau2.csv, X_*.csv. Results: out/eval.json, out/eval.txt, out/extra_metrics.json, run.log, diag.md. LLM calls: 0.
