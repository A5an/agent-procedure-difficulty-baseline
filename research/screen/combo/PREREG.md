# Exp 5: a combined router on TauIndianBank (pre-registration, 3 Nov 2026)

No API calls: everything is recorded (P0-2 real runs and documentation probes, Exp 3 seeded probes, Exp 2 monitor
features). Written before any combined routing number was computed. What I had seen before: the per-family routers of
P0-2 and Exp 3 (probes only) and the monitor's AUCs inside TauIndianBank (0.84 pooled, 0.70 within task).

## Question

P0-2 and Exp 3 routed whole procedure families (agent or human) from probes and never beat "always human". A real
deployment has a second lever: let the agent work and send the risky episodes to a person at the moment the agent is
about to commit (the "agent + verifier" option of the router task). Does probes-for-the-family plus a monitor-for-the-
episode beat the best constant policy and the probes-only router?

## Data

20 families x 6 real tasks x {gemini-3.8-flash, gemini-2.5-flash} (240 real episodes, outcomes and messages);
per family and agent 6 documentation probes (P0-2, scoring v2) and 6 seeded probes (Exp 3), with messages.

## Signals

- **Family failure estimate p(a, f)** before deployment, from the family's probes. Variants: seeded raw rate,
  documentation raw rate (both (k + 0.5) / (n + 1)), each optionally calibrated by the agent's level shift learned on
  the other 19 families (mean real minus mean probe failure, clipped to [0.01, 0.99]); and "no probes" = the agent's
  mean real failure on the other 19 families.
- **Episode monitor**: the 13 commit-point features of `../monitor/build.py`, gradient boosting with the settings of
  `../monitor/eval.py`. Two training sources: (M1, primary) real episodes of the other 19 families (shadow logs of
  earlier procedures); (M2) probe episodes only (documentation + seeded, all 20 families, both agents), which exist
  before any deployment. Labels of probe episodes as scored in their reports.
- **Operating curve** of the monitor for agent a at flag rate r (r in 0, 0.1, ..., 0.9): threshold = the (1 - r)
  quantile of out-of-fold scores on the training episodes of agent a; TPR(r) and FPR(r) measured there. For M1 the
  out-of-fold scores come from an inner leave-one-family-out over the 19 training families.

## Decision rule (fixed in advance)

Per family f, options: human; agent a alone; agent a with the monitor at flag rate r. Expected loss per case:
human h + F(1 - q); agent alone api_a + F p; agent + monitor api_a + [p TPR + (1 - p) FPR] (h + F(1 - q)) +
p (1 - TPR) F. The router picks the option with the lowest expected loss. Realized loss per real case: human
h + F(1 - q); agent alone api_a + F fail; agent + monitor: if the episode's score is above the threshold,
api_a + h + F(1 - q), else api_a + F fail. Mean over the family's 6 cases, then over families.

API cost per episode as measured in P0-2 (g38 0.0385, flash 0.0060 USD). Costs: (F 20, h 5, q 0.99) primary, plus
(20, 2, 0.95), (50, 5, 0.99), (10, 5, 0.99).

## Hypotheses (primary cost setting, paired bootstrap over families, 4000 draws)

- C1: combined router with calibrated seeded probes and monitor M1 beats the best constant policy (upper bound of
  the difference below 0).
- C2: it beats the probes-only router with the same family estimates (upper bound below 0).
- C3 (exploratory): the same with documentation probes instead of seeded; with monitor M2; with no probes.

## Caveats known in advance

20 families only. Seeded probes are an upper bound (templated families, see Exp 3). The monitor M1 needs real logs of
other procedures of the same deployment. The decision rule is chosen once here; no tuning on the test families.

## Addendum, written after combo.out and extra.out, before any SOPBench number (3 Nov)

Replication on SOPBench with the one agent whose SOPBench episodes we have with full messages (gemini-3.8-flash: 830
real cases and 1,365 code-generated probe cases from `../ideas5/smart_gemini`). Options per procedure: human, g38
alone, g38 with the monitor (flag rates 0.1 to 0.9). Procedure estimate: g38's probe failure rate on the procedure,
raw and calibrated by the level shift on the other domains (leave one domain out). Monitors: M2 trained on g38's probe
episodes (all procedures; operating curve from out-of-fold scores by procedure), M2 without the target procedure's
probes, and M1 trained on real episodes of the other domains. API 0.021 USD per episode (`../mf/route.py`). Same cost
settings and decision rule. Hypotheses: S1, the combined router with calibrated probes and M2 beats the best constant
policy (paired bootstrap over the 68 procedures); S2, it beats the probes-only router.
