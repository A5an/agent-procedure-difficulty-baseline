# E6, second pass: pass^k with the best model (13 Oct 2026)

Script `passk2.py`, results `passk2.json`. tau2, out-of-fold P from `../ideas10/fc/out` (5 seeds for new procedures, 1 for new
domain), 4,170 agent-task cells per seed, up to 4 attempts per cell. Brier of pass^k against the share of all-success k-subsets.

| k | observed mean | baseline raw, independent | f2 recalibrated, independent | f2 recalibrated, beta-binomial |
|---|---|---|---|---|
| new procedures, 1 | 0.759 | 0.1806 (pred 0.827) | 0.1718 (0.761) | |
| 2 | 0.671 | 0.2180 (0.703) | 0.2136 (0.595) | 0.2058 (0.670) |
| 3 | 0.610 | 0.2379 (0.609) | 0.2431 (0.475) | 0.2217 (0.617) |
| 4 | 0.564 | 0.2490 (0.535) | 0.2648 (0.385) | 0.2298 (0.579) |
| new domain, 1 | 0.759 | 0.1984 (0.841) | 0.1790 (0.747) | |
| 2 | 0.671 | 0.2473 (0.733) | 0.2266 (0.580) | 0.2141 (0.661) |
| 3 | 0.610 | 0.2708 (0.652) | 0.2576 (0.462) | 0.2305 (0.612) |
| 4 | 0.564 | 0.2824 (0.590) | 0.2780 (0.378) | 0.2391 (0.579) |

Main comparison, f2 recalibrated beta-binomial minus baseline raw independent, Brier, task bootstrap:
new procedures k=2 -0.0121 [-0.0173, -0.0069], k=3 -0.0162 [-0.0223, -0.0101], k=4 -0.0192 [-0.0266, -0.0117];
new domain k=2 -0.0332 [-0.0409, -0.0256], k=3 -0.0402 [-0.0495, -0.0316], k=4 -0.0433 [-0.0534, -0.0334].
Beta-binomial against independent, both with f2 recalibrated: k=4 -0.0350 [-0.0459, -0.0228] and -0.0389 [-0.0508, -0.0268].

Findings
- With a P that separates tasks (f2), independent attempts under-predict pass^4 badly (0.38 against 0.56 observed): attempts of one
  agent on one task are strongly correlated (phi about 1.1 to 1.2). The beta-binomial form fixes that (0.58).
- With the baseline P, which is nearly flat, independent attempts looked fine in the first pass because a flat p^k happens to land
  near the observed mean. That result does not carry over to a model that ranks tasks.
- Recalibration moves f2's mean P up by a logit shift of +0.22 and +0.32; the gain from it is smaller than from the dispersion term.
- Answer for the plan: pass^k for a given agent and task = beta-binomial with mu from the best model and phi fitted on training
  cells. Banking is not covered (no trained f2 there).
