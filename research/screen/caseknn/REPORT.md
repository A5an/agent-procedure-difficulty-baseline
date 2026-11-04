# Case-level prediction before running: nearest probes (4 Nov 2026)

Plan `PREREG.md`, code `caseknn.py`, output `caseknn.out`. 600 real cases (50 families x 6 x 2 agents).

Hypothesis not supported: inside a family x agent cell, weighting the seeded probes by how similar their customer script
is to the case gives AUC 0.556 [0.442, 0.664] (35 cells with both outcomes). Pooled over all cases the plain family
probe rate already separates failing from passing cases at AUC 0.821, and the nearest-probe version adds nothing
(0.821, Brier 0.160 vs 0.157).

Reading: on TauIndianBank the cases of one family are written from one template, so their scripts differ in customers
and ids, not in what makes them hard; text similarity cannot tell them apart. Case-level prediction before running
needs either structurally different cases (real tickets) or features of the case's data (the customer's records),
not the script text. The procedure-level rate already carries most of the case-level signal (AUC 0.82).

## Customer records instead of script text (`caserec.py`, exploratory)

Counts of the customer's accounts, cards, deposits, mandates, requests, loans, insurance and record size, with the
family probe rate, gradient boosting leave one family out: within family x agent AUC 0.513 over 50 cells. Pooled
AUC 0.766, below the probe rate alone (0.821). Within a family, which case fails is not predictable from what we can see
before the run; together with E8 (two identical runs of the same agent disagree on 17 of 150 cases) this points to
run-to-run randomness as the main source of within-procedure variation.
