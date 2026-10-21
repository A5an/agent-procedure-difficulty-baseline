# Blind test on ThinkingBox-Bench v1.0: result (15 Oct 2026)

Pre-registration: `PREREG.md` (commit c052360, before any agent run). Deviations in the runs: `DEVIATIONS.md`.
Analysis: `evaluate.py`, unchanged since the freeze, run once on the committed `outcomes.csv` (commit 992efd2);
output `results.json` and `results.txt`.

## What was run

405 tasks (the first 81 of each domain in the frozen random order), four agents (gemini-2.5-flash-lite,
gemini-2.5-flash, gemini-2.5-pro, gemini-3.8-flash) with two trials each: 3,240 runs in the ThinkingBox harness, all
slots valid after retries. Success per agent: 0.002, 0.090, 0.162 and 0.652; mean 0.227. Difficulty of a task = 1 -
share of its 8 runs that passed.

## Primary result

Pooled within-domain Spearman rho between score and difficulty, 95% bootstrap intervals over task clusters:

| predictor | rho | pairwise accuracy |
|---|---|---|
| D, one question to Sonnet 5.5, nothing fitted | 0.124 [0.022, 0.224] | 0.555 |
| baseline, 18 ADeLe levels, ridge fitted on SOPBench + tau2 + banking | -0.055 [-0.149, 0.044] | 0.476 |
| text length, same ridge | 0.051 [-0.055, 0.152] | 0.523 |

| difference | point | 95% interval |
|---|---|---|
| D - baseline (primary, point 5 of §8) | +0.179 | [+0.067, +0.297] |
| D - length | +0.073 | [-0.046, +0.204] |
| baseline - length | -0.106 | [-0.202, -0.013] |

**Point 5 of the improvement rule is met**: D ranks the tasks better than the baseline, with the interval above zero.
D is not significantly better than text length, and its rho (0.12) is well below the 0.3 to 0.5 written in the
pre-registration.

By domain (D / baseline / length): auto insurance 0.12 / -0.18 / -0.06; consulting IT and HR 0.09 / -0.18 / 0.00;
neobank support 0.22 / 0.09 / 0.32; retail 0.09 / -0.04 / -0.05; travel 0.10 / 0.03 / 0.05. D is positive in all
five domains; the baseline is negative in three.

Against each agent's own success (pooled rho, D): 3.8 Flash 0.15, 2.5 Flash 0.04, 2.5 Pro -0.04, Flash-Lite -0.03.
The three 2.5 models solve too little for their own ranking to carry much signal; the target is driven mostly by
gemini-3.8-flash.

Calibration (descriptive, as registered): D's mean p_success on the 405 tasks is 0.353 against an observed 0.227; its Brier score
against the observed share is 0.055, worse than the constant mean (0.033). D was not written to be calibrated to these
agents.

## Exploratory, not pre-registered

Reliability of the target: difficulty from trial 1 alone against trial 2 alone gives pooled rho 0.64 (0.78 for the
full 8 runs by Spearman-Brown). The target is far from pure noise, so the low rho of D comes from the predictor on
this benchmark, not from an unmeasurable target. 104 of the 405 tasks were solved by no run and none by all eight.

## Cost

Main run 192.64 USD on Vertex (ledger in `run/ledger.jsonl`), pilot 0.92 USD, ADeLe annotation for the baseline about
37 USD; the 507 Sonnet calls of D ran on the subscription (about 44 USD at API prices). The budget rule allowed 150 USD
projected from the pilot; gemini-3.8-flash cost twice its pilot estimate, and the cap was raised to 220 USD with the
user's approval during the run (DEVIATIONS.md, item 4).

## Reading

On a benchmark never used for any choice, the zero-shot estimate keeps a small positive signal in every domain and
beats the transferred ADeLe baseline, which carries no signal here. The signal is much weaker than on tau2 (0.50 to
0.57) and banking (0.60), and it does not separate from plain text length. ThinkingBox is not used again for any
choice in this project.
