# Replication of Exp 5 on 30 new TauIndianBank families (4 Nov 2026)

Plan: `PREREG.md`, hashes before the first episode in `FREEZE_PRE.txt` (probe files appended when they were written,
before their episodes). Spend 40.32 USD of the 45 USD approved (probe generation 11.68, gemini-3.8-flash 22.05,
gemini-2.5-flash 3.22, user simulator 3.36). No episode lost: 180 real tasks, 180 seeded and 179
documentation probes, each run by both agents. Analysis `analyze_rep.py` (the code of `../combo/combo.py`; on the
earlier 20 families it reproduces combo.out exactly); output `analyze_rep.out`, `results_rep.json`.

The new families are easier than the first 20: real failure 0.28 (g38) and 0.27 (flash), against 0.44 and 0.39.

## Pre-registered hypotheses (primary costs: failure 20 USD, human 5 USD at 99%)

| | router | loss per case | vs best constant (always human 5.20) | vs probes only | verdict |
|---|---|---|---|---|---|
| R1 | 30 new families, seeded + calibration + monitor on seeded probes | 2.93 | **-2.27 [-3.29, -1.17]** | -0.76 [-2.13, +0.28] | **supported** |
| R2 | same, against the probes-only router | | | -0.76 [-2.13, +0.28] | not supported |
| R3 | 30 new, monitor on documentation + seeded probes | 2.76 | **-2.44 [-3.44, -1.47]** | **-0.93 [-2.04, -0.03]** | supported |
| R4 | all 50 families, R1 variant | 3.23 | **-1.97 [-2.78, -1.10]** | **-0.78 [-1.62, -0.15]** | supported |

The per-family oracle is 2.18 on the new families, so the R1 router closes 75% of the gap between "always human" and
the oracle, sending 37% of cases to a person.

## Other numbers

Monitor on the 180 real episodes of the new families (AUC; within family x agent): trained on seeded probes 0.846
(0.772), on documentation + seeded probes 0.835 (0.822), on real logs of the other families 0.679 (0.783).

With more families the monitor trained on real logs of other families also works: new 30 -1.99 [-3.13, -0.74], all
50 -1.93 [-2.76, -1.05] (on the first 20 it was borderline, -1.56 [-2.95, 0.00]).

Documentation-only probes still do not beat "always human": new 30, probes only -0.94 [-2.38, +0.69], with any monitor
between -0.75 and -1.22 with intervals through 0; all 50 the same.

Other cost settings for R1 on the new families: human 2 USD at 95% -1.26 [-1.81, -0.72]; failure 50 USD -2.01
[-3.18, -0.81]; failure 10 USD (best constant always flash) -0.29 [-1.03, +0.49].

## Reading

The result of Exp 5 replicates on 30 families nobody had looked at: probes written from two past conversations of the
procedure, used both for the procedure's failure estimate and as training data for a commit-point monitor, give a
router that beats "always human" by about 2.3 USD per case (45%) and gets three quarters of the way to the oracle. On
all 50 families the monitor also adds reliably to the probes-only router. Probes written from the one-line description
alone still do not suffice. The episode monitor itself is not fussy about its training source once there are enough
families: real logs of other procedures work about as well as probe runs inside a family.

## Limits

One benchmark, two agents of one family, one user simulator, seeded probes on templated families (the seeds show the
generator almost the exact scenario), one run per case. The cost settings are assumptions; at a cheap failure price no
router beats the best agent.
