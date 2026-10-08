# PREREG failure-type features from the pre-mortem (round 10, 8 Oct 2026, before any run)

Idea: the pre-mortem lists concrete failure modes with probabilities, but a single p_success weighs them by Sonnet's
own beliefs. Old agents fail more often in some kinds of traps than Sonnet expects (on tau2 the misses were refusals
under customer pressure). Grouping the failure modes into types lets the model learn, on training folds, how much each
type matters.

Features FCT (L1, no new LLM calls): embed every failure-mode sentence of the S1 pre-mortem (fc/fc_raw.json) with
bge-small-en-v1.5 (local CPU), k-means with K = 12 (fixed now, seed 0) over all benchmarks together, feature
fct_k = sum of p over the case's failure modes in cluster k. Clustering is unsupervised and label-free.
Variants (one harness run): f4 = f2 (c_lad + GAP + SCN + PRIOR + FC) + FCT; f5 = ADeLe + FC + FCT.
Primary: f4 > f2 pooled (paired CI). Report cluster contents (top sentences) for interpretation.
