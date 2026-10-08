# Risk-type split and pilot-run value (agent `risk`, 5 Oct 2026)

No LLM calls. Existing data only. Code: `extract.py` (flags per run), `partA.py`, `partA2.py`, `partB.py`.
Outputs: `flags.csv`, `outcomes.csv`, `partA_tables.md` (all Part A tables), `case_split.csv`, `proc_table_*.csv`, `partB_table.csv`, `partB_diffs.csv`.

## Part A. Failure by risk type (SOPBench, 830 cases x 28 configurations, 23,235 runs)

### Setup
- Every run was re-scored with SOPBench's own evaluator (same call as the baseline repo build). The recomputed success matches the repo's responses.jsonl in 100% of runs. 5 runs with no trajectory are left out.
- Outcome type, in this order: empty answer (no tool calls and an empty last assistant message), correct (success), then for failures:
  - must-refuse case: unsafe action if the action succeeded, or the database differs, or a tool response contradicts the strict ground truth. Otherwise process-only.
  - must-perform case: wrongful refusal if the target action was never called. Wrong execution if it was called but did not succeed, or database or constraint flags are bad. Otherwise process-only (the dirgraph was not satisfied, 98.8% of these).
- Rates are conditional on the label that makes the type possible: unsafe among must-refuse cases (542 cases), wrongful refusal and wrong execution among must-perform cases (288), process-only and failure among all.
- Empty-answer configurations gpt-5, gpt-5-mini, gemini-2.5-pro are flagged and everything is shown with all 28 and with the other 25. My empty-answer measure gives 42.5%, 6.6% and 9.9% of their runs, lower than the 89, 47 and 28% quoted in CONTEXT.md. I could not reproduce those numbers from the trajectories (text content is also empty for most react and act-only runs of other models, so a text-only rule is unreliable). Treat my flag as a lower bound. Excluding the three configurations is the safe analysis.

### Rates (share of runs)
Without the three flagged configurations (25 agents, 20,745 runs):

| label | correct | unsafe | wrongful refusal | wrong execution | process-only | empty |
|---|---|---|---|---|---|---|
| must-refuse (13,546 runs) | 0.502 | 0.456 | 0 | 0 | 0.036 | 0.007 |
| must-perform (7,199 runs) | 0.302 | 0 | 0.134 | 0.160 | 0.401 | 0.003 |
| all | 0.432 | 0.297 | 0.046 | 0.056 | 0.163 | 0.006 |

With all 28: must-refuse unsafe 0.416, correct 0.521, empty 0.029. The flagged three have the lowest unsafe rates (0.02 to 0.09) and that low rate is partly an artefact of doing nothing.
By domain (25 agents), unsafe among all runs: university 0.405, library 0.376, hotel 0.380, online_market 0.299, healthcare 0.266, bank 0.221, dmv 0.174. Full domain and per-configuration tables are in `partA_tables.md`.
Two facts shape everything below. Among must-refuse cases almost every failure is an unsafe action (0.456 of runs). Among must-perform cases the largest failure is process-only (0.40: the action is done but the prerequisite checks were skipped), so for a perform case the strict benchmark success hides that the outcome was right.

### Is each rate a property of the procedure?
Share of case-level variance between procedures (adjusted eta squared, chance level in brackets, 25 agents, case rate = mean over agents):

| rate | cases | procedures | between-procedure share (adj.) | between-domain share (adj.) |
|---|---|---|---|---|
| unsafe | 542 | 66 | 0.25 (chance 0.12 before adjusting) | 0.16 |
| wrongful refusal | 288 | 70 | 0.46 | 0.07 |
| wrong execution | 288 | 70 | 0.80 | 0.09 |
| process-only | 830 | 70 | 0.03 | 0.05 |
| any failure | 830 | 70 | 0.23 | 0.18 |

Two stability checks, 1000 random splits each, Spearman-Brown corrected:

| rate | split agents (procedure ranking, min 3 cases) | split cases (min 6 cases) |
|---|---|---|
| unsafe | 0.93 (48 procedures) | 0.82 (26) |
| wrongful refusal | 0.92 (28) | 0.80 (14) |
| wrong execution | 0.90 (28) | 0.95 (14) |
| process-only | 0.94 (54) | 0.55 (37) |
| any failure | 0.93 (54) | 0.81 (37) |

Reading: the agent split is high for everything because all agents see the same cases, so it only says agents agree on which cases and procedures are hard. The case split is the honest test of "property of the procedure". Unsafe action rate and wrongful refusal rate are moderately stable (0.8). Process-only is not (0.55, and it needs 10 cases per procedure to reach 0.74). Wrong execution looks the most stable, but it is concentrated in procedures with 1 to 4 must-perform cases where nearly all agents fail (hotel check-in, check-out, show_available_rooms, library add_book). This looks like a benchmark artefact (action impossible or database check unsatisfiable) and not a measurable agent risk. Unsafe and wrongful refusal are weakly negatively related across procedures (rho -0.37 within domain) and unsafe tracks overall failure (0.73).

### Documentation vs risk type (out-of-fold predictions, new_domain split)
Per procedure mean predicted difficulty (minus logit of predicted success, averaged over cases and agents), Spearman with the empirical rate inside a domain (pooled by pairs, 25 agents, bootstrap over procedures within domain, 95% interval). Procedures with fewer than 3 relevant cases are dropped for the type rates.

| method | any failure (70) | unsafe (48) | wrongful refusal (28) | wrong execution (28) | process-only (54) |
|---|---|---|---|---|---|
| adele (baseline) | 0.24 [-0.01, 0.46] | 0.43 [0.07, 0.69] | 0.05 [-0.52, 0.57] | -0.10 [-0.62, 0.51] | 0.11 [-0.20, 0.41] |
| lad2_adele (rubric) | 0.40 [0.14, 0.60] | 0.27 [-0.13, 0.51] | 0.36 [-0.22, 0.83] | 0.00 [-0.57, 0.52] | 0.30 [-0.04, 0.60] |
| cat_proc_adele | 0.35 [0.10, 0.55] | 0.34 [-0.03, 0.55] | 0.21 [-0.37, 0.72] | -0.12 [-0.68, 0.48] | 0.26 [-0.07, 0.56] |
| grp_proc_adele | 0.43 [0.17, 0.61] | 0.23 [-0.17, 0.50] | 0.34 [-0.24, 0.80] | -0.02 [-0.60, 0.53] | 0.21 [-0.12, 0.56] |

Paired differences (type minus overall failure) have intervals that all include zero (for example adele unsafe minus failure +0.19 [-0.17, +0.48], lad2_adele wrongful refusal minus failure -0.04 [-0.64, +0.54]). Baseline adele ranks unsafe better than overall failure and the rubric methods rank overall failure better than unsafe, but none of these gaps is significant. Wrong execution is not predicted by any method, consistent with it being an artefact. The share of must-perform cases in a procedure correlates -0.27 with adele's predicted difficulty (the only significant one) and about 0 with the rubric methods.

### Verdict, Part A
- Failure splits very unevenly: 0.46 of must-refuse runs are unsafe actions, 0.13 of must-perform runs are wrongful refusals, and 0.40 of must-perform runs only break process order. The empty-answer configurations flatter the safety numbers.
- Unsafe action and wrongful refusal rates are moderately stable properties of a procedure (case-split 0.8) with a between-procedure share of 0.25 and 0.46. Process-only and wrong execution are not usable (noise and artefact).
- Documentation predicts risk types no better than overall difficulty, and no significant difference between types exists at this sample (28 to 70 procedures). The baseline hints at unsafe (0.43) while rubric features hint at overall failure (0.40 to 0.43), both with wide intervals.

## Part B. Documentation prior in pilot runs saved (SOPBench)

### Setup
- Unit: (procedure, agent) pair, 70 procedures x 25 agents = 1750 pairs (28 agents: 1960). Truth: success rate over all cases of the procedure, automate if it is at least tau (0.5, 0.7, 0.8). Share of pairs that should be automated: 0.48, 0.27, 0.22.
- Pilots: n cases drawn at random without replacement (capped at the procedure size). Decision: automate if the posterior probability that the full-procedure rate is at least tau is 0.5 or more. The posterior accounts for the finite population: unseen cases follow a beta-binomial predictive, so seeing all cases gives the exact truth. An earlier version with a plain Beta(theta) posterior was biased for small procedures and was discarded before reporting.
- Accuracy is computed exactly (hypergeometric draws, dynamic programming), no simulation noise. Intervals: bootstrap over procedures, 1000 draws.
- Priors: flat Beta(1,1). Agent-only: mean is the agent's pooled success on the training domains, strength by beta-binomial marginal likelihood on training domains. Documentation: mean is the agent's mean out-of-fold predicted success on the procedure (adele, lad2_adele), strength fitted the same way. Domain held out as in new_domain, so the success rates and strengths of the held-out domain are never used. The fitted strength was 4 pseudo-cases in every fold, so priors are weak.
- Sequential rule: stop when the probability is above 0.9 or below 0.1, or after 12 pilots (or all cases).
- Many procedures are tiny (median 6 cases, 52 of 70 have under 12), so n=8 often sees the whole procedure and accuracy is inflated. I show a second sample with procedures of at least 6 cases (37 procedures, 925 pairs).

### Decision accuracy vs n (25 agents, all procedures, interval for n=8 is about +/-0.012)

| tau | prior | n=0 | 1 | 3 | 5 | 8 | 12 | pilots to match flat at 8 | sequential: pilots / accuracy |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | flat | 0.507 | 0.787 | 0.886 | 0.934 | 0.957 | 0.976 | 8 | 3.38 / 0.949 |
| 0.5 | agent | 0.671 | 0.796 | 0.889 | 0.936 | 0.963 | 0.979 | 7.3 | 2.90 / 0.940 |
| 0.5 | doc adele | 0.671 | 0.783 | 0.883 | 0.933 | 0.963 | 0.979 | 7.3 | 2.18 / 0.899 |
| 0.5 | doc lad2_adele | 0.680 | 0.797 | 0.891 | 0.935 | 0.964 | 0.980 | 7.2 | 3.11 / 0.952 |
| 0.7 | flat | 0.694 | 0.760 | 0.902 | 0.946 | 0.971 | 0.984 | 8 | 2.62 / 0.957 |
| 0.7 | agent | 0.718 | 0.808 | 0.907 | 0.949 | 0.972 | 0.984 | 7.7 | 2.25 / 0.945 |
| 0.7 | doc adele | 0.715 | 0.803 | 0.906 | 0.947 | 0.972 | 0.984 | 7.8 | 1.89 / 0.924 |
| 0.7 | doc lad2_adele | 0.737 | 0.822 | 0.907 | 0.951 | 0.973 | 0.985 | 7.6 | 2.44 / 0.962 |
| 0.8 | flat | 0.745 | 0.830 | 0.923 | 0.956 | 0.977 | 0.988 | 8 | 2.47 / 0.970 |
| 0.8 | agent | 0.763 | 0.837 | 0.929 | 0.956 | 0.978 | 0.989 | 7.7 | 1.74 / 0.950 |
| 0.8 | doc adele | 0.744 | 0.826 | 0.930 | 0.956 | 0.977 | 0.989 | 7.8 | 1.62 / 0.942 |
| 0.8 | doc lad2_adele | 0.781 | 0.853 | 0.933 | 0.957 | 0.977 | 0.990 | 7.7 | 1.90 / 0.967 |

Procedures with at least 6 cases (flat at n=8: 0.918, 0.945, 0.956 for the three taus): the same pattern. Pilots to match flat-at-8 are 7.2 to 7.8 for every informative prior, sequential pilots for flat are 4.5, 3.4, 3.2 versus 4.15, 3.14, 2.29 for doc lad2_adele, and for the agent-only prior 3.7, 2.9, 2.1. The 28-agent version differs by at most 0.03 in any cell (`partB_table.csv`).

Paired comparisons with bootstrap intervals (`partB_diffs.csv`):
- Any informative prior beats flat at n=0 only at tau 0.5 (+0.17 [0.10, 0.23]) because flat n=0 is a coin flip there. At tau 0.7 and 0.8 the gain at n=0 is 0 to 0.04.
- Documentation vs agent-only prior: adele is no better (differences between -0.02 and +0.02 at every n, intervals around zero). lad2_adele is slightly better at n=0 and 1 (+0.01 to +0.02, intervals mostly excluding zero at tau 0.7 and 0.8) and at the sequential rule it buys +0.015 accuracy for +0.2 pilots, which is a trade, not a saving.
- At n=3 or more no prior differs from flat by more than 0.007.

### Verdict, Part B
- Pilots dominate. Three pilots already give 0.89 to 0.93 accuracy, eight give 0.96 to 0.98 whatever the prior (this is optimistic because small procedures are fully observed).
- A documentation prior saves about 0.2 to 0.8 of a pilot against the flat prior to reach the accuracy flat gets with 8, and the agent-only prior (no documentation) saves nearly the same (0.3 to 0.8). Documentation adds at most 0.02 accuracy at n=0 or 1 over the agent-only prior.
- So the case for documentation in this metric is weak: it beats a flat prior mostly because the agent's overall level is informative, and the extra value of documentation is small and only visible with zero or one pilot.

### Tau2
Does not apply. In tau2 each task is its own "procedure" (airline 50, retail 114, telecom 3 distinct task ids, the policy is shared per domain), so there is no set of procedures to decide on. Grouping by domain gives 3 groups x 15 agents = 45 pairs with a documentation prior that is a single number per domain, too few for any bootstrap or fitted prior strength, and the 4 attempts per task are not independent pilots of a procedure.

## Limitations
- The empty-answer flag is a lower bound (see setup). The unsafe rate of the flagged agents is not trustworthy.
- Rate definitions are mine. "Unsafe" includes database or response mismatches on must-refuse cases, so some unsafe cases are tool-use errors and not necessarily harmful actions. Process-only on must-perform cases is mostly a skipped prerequisite check although the outcome is right.
- 52 of 70 procedures have under 12 cases and 33 have under 6, so every procedure-level number has wide intervals. Rates of unsafe and wrongful refusal use only 48 and 28 procedures with at least 3 cases of the right label.
- The case-split reliability treats the agents as fixed. Agents are not a random sample of deployable agents (many are variants of the same model).
- Part B: the strength of the documentation prior was fitted on the out-of-fold predictions of the training domains. Those predictions come from models that had the held-out domain in training, a small leak in principle that I did not remove. Predictions are for the pooled agent, no agent-specific documentation effect is modelled beyond the IRT agent term. Bootstrap intervals for the pilots-to-match number were summarised only by their median (equal to the point estimate within 0.02), full intervals were not stored.
- Part A3 uses one seed of one split (new_domain, 7 folds). Not tuned on test folds, all variants were fixed before running. One earlier Part B run with a biased posterior was replaced, as stated above.
