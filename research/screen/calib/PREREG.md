# Calibrated success probability from few probes (pre-registration, 4 Nov 2026)

No API calls. Goal (Tima, Zangir): a trustworthy percentage that the agent handles a procedure. Data and metrics of
`../metrics` (TauIndianBank 50 families x {g38, flash} with past-conversation and documentation probes; SOPBench 68
procedures x {lite, flash, g38} with code-generated probes). Everything leave one group out (family / domain).

Predictors of a procedure's success probability:
- RAW: (k + 0.5) / (n + 1) from the agent's own probes (current best).
- EB: beta-binomial shrinkage; prior Beta fitted by moments to the probe success rates of the other procedures.
- RECAL: logistic recalibration, logit(real rate) = a + b logit(RAW), a and b fitted on the other procedures.
- POOL (TauIndianBank only): EB where the other agent's probes on the same procedure add evidence at a weight w
  fitted on the other procedures (w in 0, 0.25, 0.5, 0.75, 1).
- Delegation by a lower bound: delegate when the EB posterior probability that the procedure's success rate is at
  least t is at least 0.8 (instead of the point estimate at least t).

Metrics: Brier skill against the base rate, log-loss, calibration (ECE over 5 bins), and for t = 0.9 and 0.8 the share
of cases delegated and the real success among them. Hypothesis: EB or RECAL lowers Brier relative to RAW on most of the
5 agent-benchmark pairs; descriptive otherwise.
