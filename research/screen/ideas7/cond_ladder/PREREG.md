# Pre-registration, agent cond_ladder, round 7 (6 Oct 2026). Written before the first run. No LLM calls.
Target: all_irt, pooled within-domain case rho over sop-proc, sop-dom, tau2-proc, tau2-dom. tau2 has no conditions: the
variants are plain c_lad there (same code path, checked equal to ideas3/combo/out_main).
Model: copy of the c_lad Ladder (rubric ladder L2, global columns RUB glob + ADELE + PLAN + CMP, one common ridge penalty).
CondLadder.fit(): on the fold's training cases only, (1) COND out-of-fold for training cases (5 inner folds grouped by
procedure, CondModel refit on inner-train cases), (2) CondModel on all training cases for test cases; appended as extra
global columns. Asserts: procedures of every fit set disjoint from the procedures it scores; plus a poison test (test
outcomes overwritten, scores unchanged).
Variants (fixed): v1_cond_score = c_lad + [logit of the l0 case score]; v2_cond_summ = c_lad + [max skip_perf, mean skip_perf,
log prod(1 - skip_perf)] over the case's conditions (required checks; skip rates from the text model).
Diagnostic: d_clad_ref = unchanged c_lad (reproduction check). Comparisons: paired bootstrap over procedures (ablation/score.py
machinery, 300 draws) vs c_lad and vs adele; metrics_extra; SOPBench without bank/cancel_credit_card and bank/pay_bill_with_credit_card.
Single run, no tuning after results.
