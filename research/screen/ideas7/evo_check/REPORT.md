# Clean validation of evolved feature set F1 (agent `evo_check`, 6 Oct 2026)

Saved by the main session from the agent's final message. CHECKS.md written before any run. 489 gemini calls (banking inputs).

Check 1, leak: survives. c_lad + F1 0.353; without most_diff_check 0.349 (+0.091 [+0.029, +0.119]); strict in-fold refit
of the skip model 0.351 (+0.093 [+0.032, +0.120]).
Check 2, fresh benchmark tau-Knowledge banking (never used in search), train on SOPBench + tau2, predict banking IRT:
c_lad 0.381 [0.180, 0.551], c_lad + F1 0.340 (paired -0.041 [-0.156, +0.070]); tau2-only training c_lad 0.439, + F1
0.392 (-0.047 [-0.127, +0.028]). Does not survive. Post hoc: statement length alone 0.53 on banking (difficulty is mostly
length there); F1 features alone 0.484.
Check 3, smart-agent target (Gemini 3.8 + Sonnet 5.5 real failure, SOPBench): search domains +0.075 / +0.041, held-out
domains +0.117 [-0.064, +0.188] / +0.102 [-0.021, +0.219], positive but no held-out CI excludes 0. Public IRT, held-out
SOPBench +0.025 / +0.019 (n.s.).
Verdict: F1 is a search-domain gain, unconfirmed outside the search domains, not reproduced on banking.
Files: CHECKS.md, run_leak.py, score_leak.txt, bank_build.py, bank_feats.py, check2.py, check2.txt, bank_preds.csv, check3.py, check3.txt.
