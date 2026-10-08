# PREREG pre-mortem ensemble FCX (round 10, 8 Oct 2026). Written AFTER seeing the single-sample FC results
(fc_score_main.txt; raw fc_diff within-domain: tau2 0.517, banking 0.596, sopbench 0.057), before any FCX call.

Hypothesis: a pre-mortem rating is a noisy single sample; averaging over prompt phrasings and model families reduces
noise (Li et al. 2512.18880 report 0.34 -> 0.47 from averaging four configurations), so the average ranks better.

Sources (all L1, one call per case each, same inputs as fc/): S1 = Sonnet 5.5, prompt P1 (existing fc/ run);
S2 = Sonnet 5.5, prompt P2 (QA-lead framing: list the steps that must be exactly right, then traps, then p_success);
G1 = gemini-3.8-flash, prompt P1; G2 = gemini-3.8-flash, prompt P2 (temperature 0, thinking low).
FCX = mean over available sources of the within-benchmark z-score of fc_diff = logit(1 - p_success).

Tests (no tuning, report all):
1. Raw within-domain Spearman of FCX vs S1 alone on tau2 and banking, paired bootstrap (rawci.py style).
2. Harness: f2x = c_lad + GAP + SCN + PRIOR + FCX (the five FC features of each source averaged after z-scoring)
   vs f2_all_fc (fc run). One run.
3. Each single source reported too (S2, G1, G2 raw).
