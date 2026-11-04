# P0-2 draft: probes and the MF model on a new benchmark (NOT frozen, needs Tima's and Assanali's OK)

Nothing here has been run. This is the plan to discuss, because it costs API money.

## Question

Does the MF model (`REPORT.md`), with its global parameters fitted on SOPBench only, predict the failure rate of an
agent on each procedure of a benchmark it has never seen, from documentation probes alone? And does a router built on
those predictions beat the best constant policy there?

## Benchmark

TauIndianBankBench (arXiv 2610.30161; data https://huggingface.co/datasets/NPCI/tau-indian-banking, MIT per the dataset
page; code https://github.com/npci/TauIndianBankBench). About 1,000 synthetic retail-banking tasks in about 50
"families" in the tau2-bench harness, 1 to 6 gold actions per task, evaluation by actions and database state.
A family is our procedure. Checked on 2 Nov from the arXiv and Hugging Face pages only; to verify before any run:
the code licence, that the harness runs locally, that each family has its own policy text the probe generator can read
without seeing the tasks.

Fallback: tau2-bench airline + retail grouped by write actions (10 + 22 groups, most with 1 to 3 tasks, too small for
per-procedure rho), where public real runs of 15 agents x 4 trials exist and only probes would have to be paid for.

## Design

- Agents: gemini-3.8-flash and gemini-2.5-flash. Both already have parameters in the MF model, so their level carries
  over and only the procedure latents are new. A third agent only if the budget allows.
- Real: R tasks per family from the benchmark, one attempt (two if budget allows, for a reliability check).
- Probes: P LLM-written tasks per family from the family's documentation and tool schema, generated and committed before
  any real run, scored by the same harness (database state, not an LLM judge).
- Same user-simulator model for real and probes; a swap of the user model on a subset is a secondary check
  (UserProxyBench reports up to 15 points from the user model alone).

## Frozen before the first real run

MF parameters (refit on all 68 SOPBench procedures), generator prompt and probe files, analysis script, cost settings
of the router (F = 20, human $5 at 0.99, and the 3 other settings of `route.py`).

## Primary outcomes

1. Within-benchmark Spearman between MF-predicted and real family failure rate, per agent, 95% bootstrap over families.
   Support: rho >= 0.40 with the lower bound above 0 for at least one agent.
2. Router: realized loss of the MF router minus the best constant policy, F = 20, human $5 at 0.99. Support: upper
   bound of the interval below 0.
3. Calibration: log-loss of MF against the constant and the raw probe rate on the real tasks.

Also reported: real split-half reliability (the ceiling), rho against n probes per family, raw rate vs MF.

## Size and cost (to fix after a 10-episode cost pilot)

Target 30 families x (8 real + 8 probes) x 2 agents = 960 episodes. At 0.02 to 0.03 USD per tau2 episode with the user
simulator that is 19 to 29 USD, at the edge of the 30 USD session cap. Smaller fallback: 20 families x (6 + 6) x 2 =
480 episodes, 10 to 15 USD. The per-episode price is a guess until the pilot.

## Risks

Probe validity (AdaT2 reports 63 to 83% of generated tests valid): every probe's gold actions are executed on the
environment before use, failing probes dropped and counted. Run-to-run noise of the real label (the benchmark paper
reports 14 to 26% of tasks change outcome on a rerun). Families with too few tasks for a stable rate.
