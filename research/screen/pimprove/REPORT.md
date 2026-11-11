# Better procedure-level prediction from the same probe runs (4 Nov 2026)

Plan: `PREREG.md` (written before any number). No API calls. Code `pimprove.py`, output `pimprove.out`,
`results.json`. 50 TauIndianBank families, leave one family out.

| predictor of the family's real failure rate | g38 rho | flash rho |
|---|---|---|
| B: 6 seeded probes (pass rate) | **0.70 [0.52, 0.83]** | **0.73 [0.54, 0.88]** |
| 6 documentation probes | 0.31 [0.01, 0.56] | 0.53 [0.30, 0.71] |
| all 12 probes | 0.62 | 0.75 |
| R: monitor risk averaged over the probe episodes | 0.28 | 0.34 |
| C: regression on seeded rate, documentation rate, R, the other agent's seeded rate | 0.61 (vs B -0.09 [-0.21, -0.00]) | 0.72 (vs B -0.01 [-0.11, +0.08]) |

Hypothesis (C beats B for both agents): not supported. Adding documentation probes, the probe-episode monitor or the
other agent's probes does not improve on the seeded pass rate; for g38 it makes it slightly worse.

How much room there is (descriptive, added after the table): the rate from 6 seeded probes has split-half reliability
0.76 (g38) and 0.82 (flash), the real rate from 6 tasks 0.79 and 0.84. Corrected for both, the correlation between the
true probe rate and the true real rate is about 0.90 (g38) and 0.88 (flash). The observed 0.70 to 0.73 is held down
by noise in both measurements; with 12, 18 or 30 seeded probes the projected observed rho is 0.74 / 0.76 / 0.78 for
g38, because the 6 real tasks per family stay noisy.

Reading: where past conversations exist, the seeded probes already track true procedure difficulty closely (about 0.9),
and what is left is measurement noise, not a missing signal. The open problem for prediction is the case without past
conversations: documentation-only probes reach 0.31 for g38, and nothing in this table closes that gap.
