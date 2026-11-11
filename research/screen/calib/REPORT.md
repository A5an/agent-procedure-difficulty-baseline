# Calibrated success probability from few probes (4 Nov 2026)

Plan `PREREG.md`, code `calib.py`, output `calib.out`. No API calls.

Hypothesis (shrinkage EB or logistic recalibration lowers the Brier score against the raw probe rate on most of the 5
agent-benchmark pairs): not supported. Brier skill against the base rate, past-conversation / code probes:

| | RAW | EB | RECAL | POOL (other agent's probes) |
|---|---|---|---|---|
| TauIndianBank g38 | +0.26 | +0.25 | +0.14 | +0.25 |
| TauIndianBank flash | +0.36 | +0.34 | +0.28 | +0.34 |
| SOPBench flash | +0.11 | +0.12 | -0.01 | |
| SOPBench g38 | +0.25 | +0.25 | +0.21 | |
| SOPBench lite | -0.04 | -0.01 | -0.01 | |

The raw probe rate is already calibrated (ECE 0.03 to 0.07 on TauIndianBank and SOPBench g38); shrinkage changes
little and recalibration hurts. Only with documentation probes does pooling the other agent's probes help a little
(skill +0.02 to +0.07 for g38, +0.12 to +0.17 for flash) and makes delegation safer (success among delegated 80 to 82%
instead of 73% for g38, at a smaller share delegated).

Delegating by a lower bound (EB posterior P(success rate >= t) >= 0.8) trades coverage for safety where probes are
many: SOPBench g38 at t = 0.9 delegates 50% of cases at 98% success (point rule: 58% at 96%); SOPBench flash 7% at
100% (point rule 18% at 94%). With 6 probes per family on TauIndianBank it equals the point rule.

Reading: turning probe outcomes into a trustworthy percentage is not the bottleneck; a raw rate from realistic probes is
already calibrated. The bottleneck is probe realism: documentation probes give almost no skill, and no statistical
treatment of their outcomes fixes that.
