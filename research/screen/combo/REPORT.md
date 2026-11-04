# Exp 5: probes for the procedure, a monitor for the episode (3 Nov 2026)

No API calls. Plan: `PREREG.md` (main part written before any combined number; SOPBench addendum written after the
TauIndianBank results and before any SOPBench number). Code: `combo.py`, `extra.py` (exploratory, after combo.out),
`combo_sop.py`; outputs `combo.out`, `extra.out`, `combo_sop.out`, `results*.json`.

## Idea

Until now the router chose per procedure: agent or human. A deployment can also let the agent work and hand the
episode to a person at the moment the agent is about to commit, when a monitor says the episode looks risky. The
monitor needs training data. Two sources: real logs of other procedures (M1), or the probe runs we make before
deployment anyway (M2). With M2 the probes are used twice: their failure rate estimates the procedure, and their
episodes teach the monitor what a failing episode of this procedure looks like.

Decision rule fixed in advance: per procedure pick human, agent alone, or agent + monitor at flag rate 10% to 90%,
by expected loss from the probe estimate p and the monitor's hit and false-alarm rates on its training episodes;
realized loss on the real cases with the monitor's actual flags. Costs as in P0-2, primary F = 20, human 5 USD at
99% quality. 95% paired bootstrap over procedures.

## TauIndianBank (20 families, 240 real episodes of g38 and flash)

Monitor quality on the real episodes (AUC, and inside family x agent cells with both outcomes):

| monitor trained on | AUC | within family |
|---|---|---|
| M1: real episodes of the other 19 families | 0.673 | 0.610 |
| M2: all probe episodes (documentation + seeded, includes the family's own probes) | 0.874 | 0.790 |
| M2 without the family's own probes | 0.717 | 0.665 |
| M2 on documentation probes only | 0.692 | 0.650 |
| M2 on seeded probes only | 0.866 | 0.681 |

Routers, primary costs (best constant: always human, 5.20 USD per case; per-family oracle 2.83):

| router | loss | vs always human | vs probes only |
|---|---|---|---|
| probes only, seeded + calibration | 3.92 | -1.28 [-2.79, +0.39] | |
| **C1**: seeded + calibration + M1 | 3.64 | -1.56 [-2.95, +0.00] | -0.28 [-0.71, +0.11] |
| **C3**: seeded + calibration + M2 | **2.88** | **-2.32 [-3.60, -0.84]** | **-1.04 [-1.97, -0.24]** |
| seeded (no calibration) + M2 | 3.17 | -2.03 [-3.38, -0.48] | -0.48 [-1.17, +0.05] |
| documentation + calibration + M2 | 4.27 | -0.93 [-2.49, +0.85] | -1.19 [-2.61, +0.27] |
| documentation + calibration + M2 on documentation probes only | 5.74 | +0.54 [-1.48, +3.07] | +0.29 [-0.37, +1.22] |
| no probes + M2 | 4.43 | -0.77 [-1.75, +0.15] | |

Pre-registered C1 (monitor from real logs of other families): not supported, the interval reaches 0. C2: not
supported. C3 (exploratory, pre-registered): with the probe-trained monitor the router beats always-human by 2.32 USD
per case (45%) and comes within 0.05 USD of the per-family oracle; it also beats the probes-only router. It sends 36%
of cases to a person and leaves 5% failures. Same direction at F = 10 (-1.30 [-2.56, -0.17] vs always flash) and at
h = 2 (-0.82, interval through 0); at F = 50 no router beats always-human.

## SOPBench (68 procedures, gemini-3.8-flash, 808 real and 1,365 probe episodes)

| monitor | AUC on real | within procedure |
|---|---|---|
| M2: probe episodes | 0.786 | 0.640 |
| M2 without the procedure's own probes | 0.700 | 0.630 |
| M1: real episodes of other domains | 0.610 | 0.674 |

| router, F 20 h 5 (best constant: always g38, 4.64; per-procedure oracle 2.40) | loss | vs best constant | vs probes only |
|---|---|---|---|
| probes only, calibrated | 2.41 | -2.23 [-3.28, -1.27] | |
| **S1**: calibrated probes + M2 | 2.36 | **-2.28 [-3.39, -1.24]** | -0.05 [-0.38, +0.34] |
| no probes + M2 | 2.94 | -1.70 [-2.77, -0.74] | |
| F 10 h 5 (best constant always g38 2.33, oracle per procedure 1.87): calibrated probes + M2 | **1.62** | -0.71 [-1.08, -0.39] | **-0.30 [-0.49, -0.14]** |

S1 supported. S2 not at the primary costs: on SOPBench the probes alone already route at the per-procedure oracle
(2.41 vs 2.40), so there is nothing left at the procedure level. When a failure is cheap (F = 10) the monitor goes
below the per-procedure oracle (1.62 vs 1.87) because it escalates single episodes, which per-procedure routing
cannot do. Without probe rates the probe-trained monitor alone recovers three quarters of the probes' gain.

## Reading

1. Probe runs are worth more than their pass rate. Trained on the episodes of the procedure's own probes, a
   commit-point monitor separates failing from passing real episodes of that procedure (within-procedure AUC 0.64 on
   SOPBench, 0.79 on TauIndianBank with seeded probes), better than a monitor trained on real logs of other
   procedures. Nothing from the deployment itself is needed.
2. On TauIndianBank, where every earlier router lost to "always human", probes plus the probe-trained monitor win
   clearly, reaching the per-family oracle. On SOPBench the probes already reach the per-procedure oracle and the
   monitor adds value only where cheap failures make case-level escalation pay.
3. The condition is the same as before: the probes must look like the real cases. With probes written from the
   one-line documentation only, neither the estimate nor the monitor is good enough on TauIndianBank and the router
   still does not beat always-human.

## Third testbed: our synthetic procedures (`combo_synth.py`, exploratory)

71 procedures of `../synth` round 1; odd cases play the probes, even cases the deployment; agents lite, flash, g38.
The probe-half monitor separates real-half failures at AUC 0.65 (lite), 0.55 (flash), 0.68 (g38). No router beats
"always g38" here (primary: probes only +0.14 [-0.43, +0.70], probes + monitor +0.11 [-0.46, +0.65]): the strong
agents fail 4 to 5% of cases and procedure difficulty does not replicate between halves of 4 cases (see
`../synth/REPORT.md`), so there is little to route. The monitor still improves on the probes-only router in two of
four cost settings (h = 2: -0.23 [-0.40, -0.09]; F = 50: -0.28 [-0.52, -0.06]).

## Literature (`LIT.md`, abstracts only, 17 arXiv ids checked)

Closest: monitors that do not transfer across deployments without recalibration (Dubey 2026, 2608.02464), a guard
trained on simulated trajectories for computer-use security (MirrorGuard, 2601.12822), a pre-deployment pilot used to
decide whether to intervene (Vasudev et al. 2026, 2602.03338), pooled failure-prediction AUC mostly measuring task
difficulty (2610.05572). In that search (3 Nov) no paper used the same probe runs both for a procedure-level estimate
and as training data for an episode-level monitor, or routed with that pair under explicit costs.

## Robustness of the TauIndianBank result (`robust.py`, exploratory)

| variant (seeded + calibration + M2, primary costs) | loss | vs always human |
|---|---|---|
| all 6 + 6 probes per family and agent, gradient boosting monitor | 2.88 | -2.32 [-3.59, -0.83] |
| same, logistic-regression monitor | 3.85 | -1.35 [-2.71, +0.17] |
| 3 + 3 probes (10 random subsets, mean loss per family) | 4.42 | -0.78 [-2.11, +0.67] |

The win is fragile: with half the probes or a linear monitor the interval reaches 0. On 20 families it is a lead to
replicate, not an established result.

## Limits

20 and 68 procedures; many variants were computed (the pre-registered primary C1 is borderline, the strong result is
the pre-registered exploratory C3), so the TauIndianBank result needs a replication on new families. Seeded probes are
an upper bound on templated families. The monitor uses 13 structural features; the operating curve is estimated on
probe episodes, whose base rate differs from the real one (handled by the per-procedure estimate p).
