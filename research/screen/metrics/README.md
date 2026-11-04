# Probability metrics instead of rank correlation (4 Nov 2026)

Zangir's point: the router needs a trustworthy percentage of success, not a ranking. Re-scoring of recorded results,
no API calls. Metrics fixed before computing: per real case the predicted success probability of its procedure;
Brier score and log-loss (with skill against the base rate learned on the other procedures), calibration error
(|mean predicted - real| per procedure, weighted, and ECE over 5 bins of procedures), and the decision metric:
delegate the procedures whose predicted success is at least t (0.95, 0.9, 0.8) and report the share of cases delegated
and the real success among them, next to the oracle that knows the true rates. Code `metrics.py`, output `metrics.out`.
