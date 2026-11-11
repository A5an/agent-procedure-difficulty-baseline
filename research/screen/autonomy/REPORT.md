# Exp 1: progressive autonomy (3 Nov 2026)

No API calls. Plan: `README.md` (written before running). Code: `autonomy.py`; outputs `sop.out`, `tib.out`,
`results_sop.json`, `results_tib.json`.

## Setup

For each held-out procedure (SOPBench: 68 procedures, leave-one-domain-out MF fits, agents flash / 3.8 flash / Sonnet;
TauIndianBank: 20 families, MF refitted on the other 19, agents 3.8 flash / 2.5 flash), half of its real cases form
the shadow pool and the other half the evaluation set (10 random splits). Shadow mode: a case is handled by the human
and every agent runs on it in the shadow. After m shadow cases the router picks the option with the lowest posterior
expected loss. Cost settings as in `../mf/route.py`.

## Results (USD per case, F = 20, human $5 at 0.99; the other settings point the same way, see `sop.out`)

SOPBench, realized loss on the evaluation half (oracle on the same half 1.69):

| shadow cases | no probes | 4 probes per agent | all ~20 probes |
|---|---|---|---|
| 0 | 3.64 | 2.49 | 2.42 |
| 1 | 3.43 | 3.15 | 3.15 |
| 2 | 3.31 | 3.07 | 3.09 |
| 4 (pool-capped, mean 2.7) | 3.20 | 3.01 | 3.05 |

As pre-planned, the router weighted each agent's two class rates (perform / refuse) by the class mix of the shadow cases
seen so far. That made shadow cases harmful once probes were in: one or two shadow cases swing the estimated mix and the
decision with it. Diagnosis run after seeing this (5 splits, `autonomy.py` with the mix fixed at the benchmark's 0.35):

| shadow cases | no probes | all probes | all probes, shadow evidence down-weighted to 0.1 |
|---|---|---|---|
| 0 | 3.77 | 2.42 | 2.42 |
| 1 | 3.47 | 2.50 | 2.42 |
| 2 | 3.43 | 2.54 | 2.44 |
| 4 | 3.37 | 2.54 | 2.42 |

## Reading

On SOPBench a few shadow cases are worth far less than the probes: 4 real cases without probes (3.37) do not reach
probes without any real case (2.42), and on top of the probes they add nothing. The reason is that real cases of one
procedure differ a lot from each other (most of the variance in SOPBench difficulty is inside procedures), so 2 to 4 of
them are a noisy estimate of the procedure's rate, while ~20 probes cover the procedure more broadly. The model also
treats each shadow case as stronger evidence than it is (no case-level variation in the model), which is why
down-weighting helps.

On TauIndianBank (6 real cases per family, 3 for the pool, 3 for evaluation) the numbers are too noisy to read
(oracle 2.50, best constant 5.20; probes + shadow 4.85 to 5.44, shadow only 4.43 to 4.64); the probes there are weak
for 3.8 flash (main P0-2 report), and shadow cases help a little without probes.

What this means for the "progressive autonomy" idea: the shadow stage is not where the decision gets made with a
handful of cases; certifying a failure rate needs tens of cases (0 failures in 29 cases bound the rate at 10%, in 59 at
5%), and our benchmarks have about 12 real cases per procedure, too few to measure the curve. The shadow stage is still
useful as training data for the commit-point monitor (Exp 2). Next fix for the model: a case-level random effect, so a
single real case moves the posterior as much as it should and no more.
