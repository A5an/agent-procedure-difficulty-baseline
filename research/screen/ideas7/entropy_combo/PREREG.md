# Pre-registration, agent entropy_combo, round 7 (6 Oct 2026). Written before any evaluation.

No LLM calls, no GPU or MLX. Target: all_irt, pooled within-domain case-level rho over sop-proc, sop-dom, tau2-proc, tau2-dom.

Entropy family ENT (from baseline/data/<bench>/entropy_profiles.jsonl, 28 Sep profiles): 19 columns = mean, sd, variance,
quantiles 10/25/50/75/90, 8 position-binned means, max of a 64-token moving average (max-entropy segment) and its relative
position, log length. Code feats.py. No outcome information used.

Model: rubric ladder L2 (step counts + 4 global rubric columns) with extra families, ridge penalty one per family
(STEP, GLOB, and each extra family), chosen by inner 5-fold CV grouped by procedure on training tasks (log-loss),
coordinate descent, one sweep, start 0.3, grid {0.003, 0.03, 0.3, 3, 30} (famladder.py). Reason: ladder beat grouped ridge on tau2.

Variants (exactly three):
 a  c_lad + ENT. Families: ADELE, PLAN, CMP (len_mean, branch_mean, SOPBench only), ENT. L1 (PLAN) + entropy trace.
 b  all cheap families: ADELE, PLAN, CMP+ (cmpplus_sopbench.csv, replaces CMP) on SOPBench, SCN (scn_tau2.csv) on tau2, ENT.
 c  b + EMU (emu_est, level E) on SOPBench; on tau2 identical to b. Reported separately from L0/L1.
Diagnostic columns (not variants, no decision attached): d_clad_fam = c_lad families with per-family penalties (no ENT),
to separate the effect of the penalty scheme from the effect of ENT; d_a_plain = a with the original common-penalty Ladder
(ablation/run.py style).

Comparisons: adele and original c_lad (ideas3/combo/out_main), paired CIs by bootstrap over procedures (ablation/score.py
machinery, 300 draws), both on the main pooled target. Extra: Brier and log-loss skill, procedure-level rho (SOPBench new
domain), rho_given_label via metrics_extra, SOPBench without bank/cancel_credit_card and bank/pay_bill_with_credit_card.
Decision rule: EVALUATION_PROTOCOL section 8, criteria 1 to 4 (5 open). Single run, no reruns, no tuning after seeing results.
Expectation stated beforehand: ENT alone adds little (Krsteski used it with a strong trace on other targets; here the
traces come from Gemini and contexts of SOPBench share policy text within a procedure), so a gain of the size of the noise
(+-0.02) is the prior.
