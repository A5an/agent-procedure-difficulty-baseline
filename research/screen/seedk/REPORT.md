# E1: how many past conversations the probe generator needs. Result (12 Nov 2026)

Plan `PREREG.md` (hashes in `FREEZE_PRE.txt`, analysis script hashed before the full episodes), pilot `PILOT.md`, code
`select_seedk.py`, `gen_seedk.py`, `run_seedk.py`, `analyze_seedk.py`, output `analyze_seedk.out`, numbers
`results_seedk.json`. 70.01 USD in total for E1 and E2 together, the hard limit set by Tima (generation 24.14, agent
gemini-3.8-flash 31.82, agent gemini-2.5-flash 5.35, user simulator flash 4.71, user simulator pro 3.99).

What ran: k=5 and foreign on all 50 families, both agents (299 and 300 probes per agent, 0 errors). k=1 was stopped by
the budget: 28 families generated; gemini-2.5-flash ran all 164 probes, gemini-3.8-flash 150 (14 refused by the cap,
26 complete families). k=0 and k=2 are the earlier runs. Valid-probe rate: k1 0.95, k5 0.94, foreign 0.97 (drops are
"no fitting candidate" and two JSON parse failures).

## Primary: Spearman between probe failure rate and real failure rate, 50 families (bootstrap over families, 2000)

| condition | gemini-3.8-flash | gemini-2.5-flash |
|---|---|---|
| k=0, documentation only | 0.31 [0.01, 0.58] | 0.53 [0.28, 0.71] |
| k=2, two past conversations of this family | 0.70 [0.52, 0.83] | 0.73 [0.54, 0.88] |
| k=5, five past conversations | 0.75 [0.59, 0.86] | 0.80 [0.67, 0.88] |
| foreign, two past conversations of another family | 0.40 [0.15, 0.61] | 0.47 [0.20, 0.69] |

On the 26 / 28 families that have k=1: k0 0.15 / 0.35, k1 0.41 / 0.55, k2 0.67 / 0.72, k5 0.70 / 0.79, foreign
0.20 / 0.40 (g38 / flash). The curve k = 0, 1, 2, 5 is monotone for both agents.

Paired differences (50 families):

| | gemini-3.8-flash | gemini-2.5-flash |
|---|---|---|
| k5 - k0 | +0.45 [+0.15, +0.75] | +0.27 [+0.08, +0.51] |
| k5 - k2 | +0.05 [-0.09, +0.21] | +0.06 [-0.09, +0.25] |
| k2 - foreign | +0.30 [+0.08, +0.53] | +0.27 [-0.02, +0.56] |
| foreign - k0 | +0.10 [-0.14, +0.36] | -0.06 [-0.36, +0.25] |

- H1 supported for both agents: five past conversations rank the families better than documentation alone.
- Most of the gain is reached at k=2; from 2 to 5 the change is small and its interval includes 0.
- H2: past conversations of another family do not help (foreign about the level of documentation probes). For
  gemini-3.8-flash k=2 beats foreign with the interval above 0, for gemini-2.5-flash the interval just touches 0. So the
  seeded gain comes from seeing this procedure's scenario, not from the style of the bank's customers. Because the
  families are templated, same-family seeds show almost the evaluation scenario: the k=2 and k=5 numbers are an upper
  bound for real past tickets, as stated in the pre-registration.

## Secondary

"All 6 probes passed" and real success (families, real successes / cases):

| condition | gemini-3.8-flash | gemini-2.5-flash |
|---|---|---|
| k0 | 17 fam, 74/102 = 0.73 | 15 fam, 80/90 = 0.89 |
| k2 | 18 fam, 98/108 = 0.91 | 24 fam, 134/144 = 0.93 |
| k5 | 19 fam, 103/114 = 0.90 | 22 fam, 126/132 = 0.95 |
| foreign | 18 fam, 86/108 = 0.80 | 16 fam, 76/96 = 0.79 |

Brier of the raw probe success rate as a forecast of each real case: g38 k0 0.241, k1 0.205, k2 0.179, k5 0.163,
foreign 0.227; flash 0.203, 0.191, 0.148, 0.138, 0.212.

Not run: `ready_proc.py` per k (needs the A5an branch environment and torch; the inputs are in `runs/`).

## Limits

- One attempt per probe and agent; per-family probe rates move in steps of 1/6.
- k=1 covers 26-28 families only, so it is compared on that subset.
- The foreign rule (same class, tool-set Jaccard < 1/3) is one choice; a different neighbour could carry more style.
- The 50 families are the same as in the earlier experiments, so k=0 and k=2 are not new measurements.
- The Mac slept 25 minutes during the pilot; three g38 runners were restarted, finished episodes kept.
