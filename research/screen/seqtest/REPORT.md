# How many probes a delegation decision needs (4 Nov 2026, exploratory, no API calls)

`seqtest.py`, output `seqtest.out`. SOPBench, gemini-3.8-flash, recorded probes (about 30 per procedure on average
over the procedures that have both probes and real cases). Sequential rule: run probes one at a time and stop as soon
as the posterior (Beta prior from the other domains) says the success rate is above or below the threshold t with
confidence c. 50 random probe orders.

| t / c | rule | probes per procedure | cases delegated | success among delegated |
|---|---|---|---|---|
| 0.9 / 0.9 | sequential | 6.0 | 68% | 94% |
| 0.9 / 0.9 | all probes | 30.5 | 61% | 95% |
| 0.8 / 0.9 | sequential | 4.4 | 84% | 91% |
| 0.8 / 0.9 | all probes | 30.5 | 88% | 91% |

A sequential stop needs 4 to 6 probes per procedure on average for the same delegation quality as all 30: easy
procedures are decided after a few successes, hopeless ones after a few failures, and only borderline ones use many.
This also answers the "6 probes cannot certify 95%" problem: certify only where the posterior allows it, and keep
probing the borderline procedures.
