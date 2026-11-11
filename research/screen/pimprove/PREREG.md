# Better procedure-level prediction from the same probe runs (pre-registration, 4 Nov 2026)

No API calls. Written before computing any number below. Tima: the goal is the prediction itself (will the agent
handle this procedure), not the router's money. Data: 50 TauIndianBank families (P0-2, Exp 3, replication), per family
and agent 6 real tasks, 6 seeded probes, 6 documentation probes (one family has 5), all with messages.

Target: real failure rate of the family for agent a. Predictors, all computed leave one family out:

- B (baseline): seeded probe failure rate.
- P12: failure rate over all 12 probes (seeded + documentation).
- R: mean risk of the probe episodes of the family under a commit-point monitor (13 features, gradient boosting as
  ../monitor) trained on probe episodes of the other 49 families. Uses how the probes went, not only pass or fail.
- C: linear regression on [seeded rate, documentation rate, R, the other agent's seeded rate], fitted on the other 49
  families, predicting the held-out one.

Measures per agent: Spearman rho with the real rate, mean absolute error, with 95% bootstrap over families (4000), and
the difference to B. Ceiling: split-half reliability of the real rate (odd vs even tasks, Spearman-Brown).
Hypothesis: C has higher rho than B for both agents (interval of the difference above 0). Everything else descriptive.
