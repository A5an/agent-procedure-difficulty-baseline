# E2-C: a second customer model with Claude agents. Result (12 Nov 2026)

Plan `PREREG_E2C.md`, analysis `analyze_e2c.py` (hash in `FREEZE_PRE.txt`, added before the runs finished), output
`analyze_e2c.out`, `results_e2c.json`. Claude subscription, $0. The 300 k=2 seeded probes were run again for each Claude
agent with Claude Sonnet as the customer instead of Claude Haiku: 600 episodes, 0 errors
(`runs/k2_haiku_simsonnet.jsonl`, `runs/k2_sonnet_simsonnet.jsonl`). Agent prompt, simulator prompt, bridge and scoring
are unchanged. The real tasks keep the Haiku customer, as in E1-C.

| | agent Claude Haiku | agent Claude Sonnet |
|---|---|---|
| probes whose outcome differs between the two customers | 0.27 [0.20, 0.34] | 0.29 [0.21, 0.38] |
| Spearman with real failure rate, Haiku customer | 0.65 [0.40, 0.83] | 0.73 [0.53, 0.88] |
| Spearman, Sonnet customer | 0.36 [0.09, 0.62] | 0.40 [0.10, 0.65] |
| Spearman, mean of the two | 0.62 [0.39, 0.80] | 0.66 [0.43, 0.84] |
| paired difference, Sonnet minus Haiku customer | [-0.60, +0.04] | [-0.63, -0.05] |
| "6 of 6" families, Haiku customer | 22, real 116/132 = 0.88 | 25, real 135/150 = 0.90 |
| "6 of 6" families, Sonnet customer | 26, real 120/156 = 0.77 | 26, real 125/156 = 0.80 |
| "6 of 6" under both customers | 15, real 84/90 = 0.93 | 16, real 95/96 = 0.99 |
| families whose "6 of 6" status changes | 18 | 19 |

Both pre-registered criteria for "the customer model matters" are met: the disagreement share has a lower bound of 0.20
and 0.21 (threshold 0.10), and the "6 of 6" set changes by 18 and 19 families (threshold 3).

## Reading

- Changing only the customer model flips the outcome of more than a quarter of the probes. For Gemini agents with the
  Gemini 2.5 Pro customer (`../sim2/REPORT.md`, a third of the episodes) the share was about 1 in 6.
- Probes rank families much better when they use the same customer model as the real tasks (0.65 and 0.73 against 0.36
  and 0.40; for the Sonnet agent the interval of the difference excludes 0). Part of what the probes predict is therefore
  the behaviour of the customer simulator, not only of the agent.
- Requiring "6 of 6" under both customers gives the cleanest set (real success 0.93 and 0.99) but keeps only 15 to 16 of
  50 families.
- Real tasks were run with one customer only, so this does not say which customer is closer to real users.
