# Pre-registration, agent fewshot_A5 (6 Oct 2026). Written before any run.
Replication of fewshot_plan variant A (two-stage ridge: 9 plan3 features -> gold [agent action count, write count, all actions]
of TRAINING tasks, out-of-fold on training rows, predicted counts used as features). No LLM calls.
Scope: tau2 new_procedures x 5 seeds + new_domain (1 partition), harness protocol, metrics_extra, paired bootstrap CIs vs adele and vs c_lad.
Columns fixed in advance:
 A, A_adele, c_lad, c_lad_A      as in fewshot_plan
 A_dm     A with gold targets demeaned within domain (domain means of training tasks) before stage 1 (check a)
 A_w      only predicted write count (b)
 A_all    only predicted total actions incl. user steps (c)
 A_raw    the 9 raw plan3 features, same final grouped ridge, no stage 1 (d)
 A_len    only predicted agent action count (extra, for completeness)
(e) per-domain rho of each column vs IRT difficulty (case-level, rho averaged over seeds), from betas.
Gold: stage 1 gets gold only via gold_of(case, train_set) assertion; test-task gold is read only after prediction, for diagnostics.
SOPBench (4): gold targets from directed_action_graph of TRAINING cases only: [action nodes (non-operator), internal_check nodes, total nodes
 incl. and/or]; same stage 1 from plan3_sopbench; columns A, A_adele, c_lad (ladder+ADELE+PLAN+CMP), c_lad_A. Pooled 4-scenario number if run completes.
