# Case-level prediction before running: nearest probes (pre-registration, 4 Nov 2026)

No API calls. Procedure-level percentages are near their ceiling; the next question is which individual case will
fail, before the agent runs it. Method: for a real case, weight the family's seeded probes by the TF-IDF cosine
similarity of the customer scripts (reason, known and unknown info, task instructions); predicted failure = weighted
probe failure, shrunk to the family's probe rate (weight 50/50). Baseline: the family's probe rate (same for every case
of the family). Measure: AUC within family x agent cells with both outcomes (50 TauIndianBank families, g38 and flash),
pooled AUC, Brier. Hypothesis: within-family AUC above 0.5 (bootstrap over families, lower bound above 0.5).
