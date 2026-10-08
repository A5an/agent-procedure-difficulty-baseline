# Few-shot planning and gold-length stage (agent `fewshot_plan`, round 4, 5 Oct 2026)

Saved by the main session from the agent's final message. tau2 only, seed 0 of new_procedures plus new_domain.
A: two-stage ridge, no LLM calls, from the 9 zero-shot plan features (plan_tools plan3) to gold agent action count, gold
write count and gold total actions, fitted on training tasks only (out-of-fold, grouped by procedure, gold access asserts
training membership). B: retrieval few-shot plans (5 nearest training tasks with gold tool-call lists), stopped at 436 of
about 1,370 calls (Vertex throughput dropped), not evaluated.

| column | new_proc rho [CI] | paired vs adele | new_dom rho [CI] | paired vs adele |
|---|---|---|---|---|
| adele (seed 0) | 0.178 | | 0.150 | |
| A | 0.375 [0.136, 0.493] | [-0.066, +0.527] | 0.467 [0.177, 0.635] | [-0.118, +0.624] |
| A_adele | 0.269 | [-0.045, +0.209] | 0.280 | [-0.114, +0.275] |
| c_lad (reference) | 0.322 | [-0.065, +0.357] | 0.285 | [-0.066, +0.302] |
| c_lad_A | 0.323 | [-0.050, +0.325] | 0.320 | [-0.055, +0.327] |

A is poorly calibrated on new_dom (Brier -0.038). Validation: A does not recover within-domain gold length better than the
zero-shot plan (airline 0.22, retail about 0, telecom 0.09 to 0.18), it removes over-planning on average (3.15 vs gold 2.93,
zero-shot 5.21). Within telecom total actions incl. user steps 0.46 to 0.61.
Verdict: A alone gives the highest tau2 numbers so far but on one seed, CI vs adele includes 0, and the mechanism is
unclear (high rho without tracking gold ordering): treat as a lead to replicate on 5 seeds with a domain-demeaned check.
c_lad_A adds +0.001 / +0.035. B untested, resumable (llm_cache.jsonl).
Files: PREREG.md, fs.py, run.py, eval.py, valA.py, out_A/, validation_A.json, llm_cache.jsonl.

## Variant B finished (6 Oct 2026, main session background run)
All B calls done (cache 967 lines, 490 calls in the final process), run.py main, eval.py out_main. tau2, seed 0.

| column | new_proc | CI vs adele | new_dom | CI vs adele |
|---|---|---|---|---|
| adele (seed 0) | 0.178 | | 0.150 | |
| B | 0.371 | [-0.083, +0.522] | -0.079 | [-0.506, +0.300] |
| B_adele | 0.310 | [-0.006, +0.388] | 0.173 | [-0.053, +0.175] |
| c_lad | 0.322 | [-0.065, +0.357] | 0.285 | [-0.066, +0.302] |
| c_lad_B | 0.344 | [-0.030, +0.432] | 0.278 | [-0.051, +0.282] |
| c_lad_Brep | 0.289 | [-0.015, +0.287] | 0.151 | [-0.080, +0.095] |

Validation (new_domain): B writes vs gold writes airline 0.63, retail 0.82, telecom -0.30 (zero-shot 0.48). With examples
from other domains the plan for telecom gets worse. B helps on new procedures (same-domain examples), fails on a new domain.
c_lad_B is level with c_lad (+0.022 new_proc, -0.007 new_dom). Verdict: no gain worth keeping.
