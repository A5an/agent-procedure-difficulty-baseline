# Exp 3: probes seeded with past conversations (3 Nov 2026)

Plan: `PREREG.md`, hashes before the first episode: `FREEZE.md`. Spend 9.51 USD of the 15 USD allowed (generation 3.19,
240 episodes 6.32). No episode lost. Code: `gen_seeded.py`, `analyze_seeded.py`, `../run.py seeded`; outputs
`probes.json`, `probes_log.jsonl`, `../runs/seeded_*.jsonl`, `results.json`.

## Result

Same 20 families and the same recorded real outcomes as P0-2. Only the probes changed: the generator now also saw two
other tasks of the family (their customer script and correct tool calls) as "past conversations".

| | gemini-3.8-flash | gemini-2.5-flash |
|---|---|---|
| rho, seeded probes (primary) | **0.70 [0.40, 0.87]** | **0.80 [0.54, 0.93]** |
| rho, documentation-only probes (P0-2) | 0.22 [-0.33, 0.68] | 0.64 [0.35, 0.82] |
| seeded minus documentation-only | +0.48 [+0.02, +0.98] | +0.17 [-0.13, +0.46] |
| rho, graded seeded score (share of gold actions missed) | 0.64 | 0.79 |
| rho, frozen MF on seeded probes | 0.65 | 0.79 |
| mean gap between seeded and real failure rate per family | 0.20 | 0.15 |

Pre-registered support (3.8 flash rho >= 0.40 with the lower bound above 0): **met**.

Families that moved for 3.8 flash: tlreqstat (real 5/6 failures; documentation probes 0/6, seeded 6/6), edgbstmtwin (6/6;
0/6 -> 3/6), sqclose (0/6; 4/6 -> 0/6). Still missed: hpservice (3/6 real, 0/6 seeded) and sq2insur (5/6 real, 1/6
seeded). One new miss in the other direction: tlacctdet (2/6 real, 5/6 seeded).

Exploratory, not pre-registered: a router over {3.8 flash, 2.5 flash, human} on the raw seeded rates against the best
constant policy ("always human" at F = 20, human $5 at 0.99): -1.56 USD per case [-2.98, +0.08]; with documentation-only
probes it was +0.90 [-1.30, +3.48]. Better, but with 20 families the interval still touches zero.

## Reading

Two past conversations per procedure fix most of what the one-line description missed: the customer's behaviour and the
final step that 3.8 flash skips. This is the friendliest form of the idea: the benchmark's families are templated, so
the seeds show almost the exact scenario of the evaluation tasks with other customers. Real past tickets vary more, so
this is an upper bound for "probes from past conversations", not an estimate. The level is still off (seeded probes are
easier than the real tasks by 10 to 15 points on average), so calibration needs some real outcomes of the agent on the
benchmark.
