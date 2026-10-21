# E3 with step types: does an agent's strength depend on the type of step? (pre-registration, 13 Oct 2026)

The 4 Oct screening (`../pertype/REPORT.md`) tested procedure-level types (action family, rule tree, domain): no gain in ranking,
a small agent x type term in log-loss. The plan said "procedure types in the screening, then step types". This is the step-type run.

Step types per case: the seven counts of the step rubric (check, look-up, ask the customer, write, irreversible action, branch,
refuse or hand over; `../rubric/features/concat/cat_full_<bench>.csv`), log(1 + count), standardised on the benchmark.
Model (confirmatory MIRT, as `mirt_q3` of the screening): logit P(a, t) = theta_a - beta_hat_t + sum_k W[a, k] x_tk, beta_hat from the
baseline (ridge on ADeLe), W ridge-penalised, lambda by 3-fold inner CV over training procedures (`steps_mirt`), and lambda fixed
at 10 (`steps_mirt_l10`) as sensitivity. Control `adele_steps`: ridge difficulty on ADeLe plus the seven counts, one ability per agent.
SOPBench and tau2, new procedures and new domain, baseline folds, repository evaluator.

Metrics: pooled within-domain rho (protocol), within-agent AUC and per-agent log-loss on held-out tasks, paired bootstrap over
procedures (300) against the baseline. The question is answered "yes" only if the agent x step-type term lowers held-out log-loss
with the interval below zero in at least three of the four scenarios. Expectation from the screening: rho unchanged, log-loss gain
small (about 0.01) or none.
