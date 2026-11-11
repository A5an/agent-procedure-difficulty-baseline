# MF: one hierarchical model for agent failure per procedure (2 Nov 2026)

No API calls. All inputs are recorded SOPBench outcomes: real runs of four agents on the 830 benchmark cases and the same
agents on the 1,365 LLM-written probe cases (gemini-2.5-flash-lite, gemini-2.5-flash, gemini-3.8-flash, Sonnet 5.5;
`ideas2/emulate_run`, `ideas5/smart_*`). Two broken bank procedures dropped, 68 procedures, 7 domains.
Code: `data.py` (tidy table), `model.py`, `eval.py` (scenarios), `eval2.py` (probe curve, ceiling), `route.py`
(routing), `voi.py` (how many probes). Literature check for this framing: `LIT.md`.

## Model

Unit = (procedure, class), class = perform or refuse case. For agent a on unit u:

    real outcome             : logit P(fail) = mR[a,k] + lR[a,k] * z_u + v_au
    probe outcome            : logit P(fail) = mS[a,k] + lS[a,k] * z_u + kap[a,k] * v_au + eps_u
    z_u ~ N(0, 1)            shared difficulty of the unit for all agents
    v_au ~ N(0, tau[a,k]^2)  agent's own deviation, shared by its probes and its real runs
    eps_u ~ N(0, se[k]^2)    how far the probes of this unit are off, the same for all agents

z and eps are integrated on a grid, v by Gauss-Hermite, so the marginal likelihood is exact up to the grid. Global
parameters are fitted by maximum marginal likelihood on the training procedures (empirical Bayes). For a held-out
procedure the posterior over (z, eps, v) given whatever evidence exists (own probes, other agents' probes, real pilots)
gives P(real failure) per agent. 52 global parameters, 68 procedures, a fit takes about 4 s.

Evaluation: leave one domain out (7 folds). Within-domain pair-weighted Spearman of the procedure failure rate
(benchmark mix 0.35 perform / 0.65 refuse, as in `ideas5/smart_eval`), case log-loss, ECE. Baselines: constant (training
mean per agent and class), raw probe rate per class ((f + 0.5) / (n + 1)), and the raw rate passed through a logistic
calibration fitted on training procedures.

## 1. How high can rho go at all

Real per-procedure failure is measured on about 12 cases per procedure, so the target itself is noisy. Split-half
reliability (halves drawn within each procedure and class, 300 splits, Spearman-Brown):

| agent | split-half | reliability | max rho against this target |
|---|---|---|---|
| gemini-2.5-flash-lite | -0.06 | about 0 | about 0 |
| gemini-2.5-flash | 0.46 | 0.63 | 0.79 |
| gemini-3.8-flash | 0.53 | 0.69 | 0.83 |
| Sonnet 5.5 | 0.50 | 0.67 | 0.82 |

The probes reach 0.69 to 0.75 (below), that is 85 to 90% of what any predictor could show against this target.
For flash-lite the real per-procedure failure rate has no reliable variation (it fails almost everything), so the
earlier "probes do not work for flash-lite" (rho 0.09, `ideas2/emulate_run`) says nothing about probes: there is
nothing to rank.

## 2. Probe-count curve (own probes, 10 random subsets per n)

| agent | n probes per procedure | rho raw | rho model | model - raw, rho [95%] | log-loss raw | log-loss model |
|---|---|---|---|---|---|---|
| flash | 2 | 0.539 | 0.540 | +0.001 [-0.056, +0.053] | 0.559 | 0.538 |
| flash | 4 | 0.596 | 0.620 | +0.024 [-0.028, +0.076] | 0.543 | 0.533 |
| flash | 8 | 0.615 | 0.608 | -0.007 | 0.517 | 0.530 |
| flash | all (~20) | 0.686 | 0.632 | -0.054 [-0.142, +0.048] | 0.480 | 0.522 |
| 3.8 flash | 2 | 0.489 | 0.667 | +0.179 [+0.115, +0.263] | 0.380 | 0.327 |
| 3.8 flash | 4 | 0.673 | 0.753 | +0.080 [+0.017, +0.187] | 0.341 | 0.298 |
| 3.8 flash | 8 | 0.734 | 0.735 | +0.001 | 0.282 | 0.265 |
| 3.8 flash | all | 0.699 | 0.713 | +0.014 [-0.052, +0.103] | 0.239 | 0.241 |
| Sonnet | 2 | 0.576 | 0.602 | +0.027 [-0.040, +0.091] | 0.453 | 0.424 |
| Sonnet | 4 | 0.650 | 0.693 | +0.042 [+0.008, +0.080] | 0.411 | 0.387 |
| Sonnet | 8 | 0.704 | 0.696 | -0.009 | 0.366 | 0.357 |
| Sonnet | all | 0.714 | 0.747 | +0.033 [-0.055, +0.146] | 0.321 | 0.317 |

Reading. The ranking comes from the probes themselves; with enough probes the model adds nothing to it. With 2 to 4
probes the model ranks better for two of three agents and gives better probabilities for all three (log-loss lower by
0.01 to 0.05; ECE 0.02 to 0.07 against 0.07 to 0.12 for the raw rate in the scenario runs, `eval_lodo.out`). With 4
probes the model ranks about as well as the raw rate with all ~20 probes (differences -0.07 to +0.05, intervals across
zero). For flash with all probes the model's probabilities are worse than the raw rate (0.522 vs 0.480): the shared
difficulty pulls flash towards the other agents. Not fixed.

## 3. Routing per procedure (the decision the model is for)

Options per procedure: flash, 3.8 flash, Sonnet, human. Choice = lowest predicted loss (API cost + F x P(fail); human
h + F(1 - q)), realized loss from the real failure rates. Leave-one-domain-out predictions, paired bootstrap over
procedures.

| costs | best constant | oracle per procedure | raw, 4 probes | model, 4 probes | raw, all probes | model - best constant | model 4 - raw 4 | model 4 - raw all |
|---|---|---|---|---|---|---|---|---|
| F 20, human $5 @ 0.99 | 4.43 (3.8 flash) | 2.24 | 2.89 | 2.55 | 2.69 | -1.88 [-2.95, -0.93] | -0.33 [-0.73, -0.02] | -0.14 [-0.62, +0.22] |
| F 20, human $2 @ 0.95 | 3.00 (human) | 1.41 | 2.97 | 1.75 | 1.97 | -1.25 [-1.67, -0.83] | -1.22 [-1.62, -0.81] | -0.22 [-0.75, +0.15] |
| F 50, human $5 @ 0.99 | 5.50 (human) | 2.66 | 5.46 | 3.49 | 4.01 | -2.01 [-2.83, -1.11] | -1.97 [-2.79, -1.02] | -0.52 [-1.21, +0.21] |
| F 10, human $5 @ 0.99 | 2.23 (3.8 flash) | 1.61 | 2.04 | 1.86 | 1.97 | -0.37 [-0.68, -0.10] | -0.18 [-0.38, -0.01] | -0.12 [-0.41, +0.15] |

USD per case. With 4 probes per agent per procedure the model's router takes 60 to 86% of the gap between the best
constant policy and the per-procedure oracle, beats routing on 4 raw probes in all four cost settings, and is not worse
than routing on all ~20 raw probes. Raw rates from few probes are pulled towards 0.5 by the smoothing, look risky and
send most procedures to the human; the model's shrinkage towards what similar procedures did fixes that. The calibrated
raw rate does not fix it (it is worse than raw here).

## 4. Other scenarios (`eval_lodo.out`)

- Probes of the other three agents only, none of the target's own: rho 0.71 (3.8 flash), 0.63 (flash), 0.60 (Sonnet),
  0.26 (flash-lite). Ranking transfers across agents; probabilities do not improve over the constant, because the
  agent's level is not identified without its own evidence.
- All four agents' probes together: better ranking for some agents, worse log-loss. The model treats outcomes of
  different agents on the same probe case as independent given the procedure, which double counts. Next fix: a
  case-level effect shared across agents.
- Text prior (Sonnet pre-mortem per procedure) in the model: no gain.
- Real pilots instead of probes: 1 real case per procedure (no probes) gives rho 0.40 to 0.52, below 4 probes (0.62 to
  0.75). Pilots are taken out of the evaluation cases, so with more pilots the remaining target gets noisier and the
  rows for 2 and 4 pilots are not comparable with each other; read only the 1-pilot row.

## 5. How many probes to buy, and for which agent (`voi.py`, `voi.json`)

Probes arrive one at a time from each agent's recorded list (random order, 5 seeds). Same total budget B per procedure.
"fixed" spreads it evenly over flash, 3.8 flash and Sonnet; "voi" gives each next probe to the agent whose outcome is
expected to lower the routing loss most (one-step expected value of sample information under the model).

| costs | B = 3: voi - fixed | B = 6 | B = 12 | probe spend, 68 procedures, B = 3 / 6 / 12: fixed vs voi |
|---|---|---|---|---|
| F 20, human $5 @ 0.99 | -0.20 [-0.41, -0.03] | -0.09 [-0.23, +0.03] | +0.03 [-0.14, +0.22] | $7.2 / 14.5 / 28.9 vs $4.5 / 8.9 / 20.6 |
| F 20, human $2 @ 0.95 | -0.45 [-0.74, -0.19] | -0.07 [-0.17, +0.02] | -0.05 [-0.17, +0.07] | $7.2 / 14.5 / 28.9 vs $4.4 / 7.8 / 17.1 |
| F 50, human $5 @ 0.99 | -0.63 [-0.94, -0.32] | -0.05 [-0.29, +0.18] | -0.01 [-0.21, +0.19] | $7.2 / 14.5 / 28.9 vs $4.5 / 8.5 / 15.5 |

With a tight budget (one probe per agent on average) choosing where to probe matters and the model-driven choice is
reliably better. From 2 probes per agent on, the losses are the same, but the VOI choice spends 30 to 45% less on
probes because it rarely asks the expensive agent when the cheap ones already settle the decision.

A stopping rule "stop when the expected value of the next probe is below its price" stopped after 2 to 3 probes and was
worse than fixed B = 12 in two of three settings. The rule as written compares a per-case loss reduction with a one-off
probe price; it has to be multiplied by the number of cases the procedure will serve, and then it almost never stops.
Not a finding, a bug in the rule; left out of the conclusions.

## Limits

One benchmark, and the friendliest one (SOPBench cases are themselves LLM-written from the same kind of policy). One
real run per case. The model needs some real history of each agent on other procedures to learn its level; a brand-new
agent needs a few real runs first. Routing options are single agents and a human; the gate option of E8 is not in
this table because it was run on 150 cases only. Costs per probe and per case are rough (from the ledgers).
