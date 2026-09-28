# Evaluation protocol and the choice of the baseline

Every number here is produced by the code in this repository and stored in `reference_results/`:
`final_evaluation.json` and `.txt` (`src/evaluate.py`), `lobo.json` (`src/lobo.py`),
`target_reliability.json` (`src/check_target.py`), `adele_retest.json` and `adele_validation.json`
(`src/adele_retest.py`, `src/adele_annotate.py validate`).

## 1. What is being predicted

Under the 1PL model, P(success) = sigmoid(theta_agent - beta_task). The ability theta of an agent is
known from its runs on other tasks. For a procedure nobody has run yet, the only unknown is the
difficulty beta of its tasks. A method is therefore judged on how well it recovers beta from the
task text alone.

## 2. Metrics

**Primary: ranking of task difficulty inside a domain.** Spearman rho between predicted and fitted
beta, and pairwise accuracy (pAcc: the share of task pairs of the same domain that are put in the
right order; 0.5 is chance). Both are computed inside each (seed, fold, domain) cell and pooled over
cells with the number of task pairs as weights, so predictions of different fold models, whose
scales differ, are never compared with each other. This is the metric of Krsteski and Meyer (2026).

**Why not pooled AUC,** the metric of Agent psychometrics. AUC over all (agent, task) outcomes
compares attempts of different agents, so it rewards knowing which agent is stronger. On SOPBench
with a new domain, the constant predictor, which gives every task the same difficulty and knows
only agent ability, reaches AUC 0.713. The baseline reaches 0.717 and the oracle 0.878. The pooled
AUC barely separates a method that knows something about tasks from one that knows nothing.

**Secondary.**
- AUC within one agent configuration (per seed, fold and agent; the constant scores exactly 0.5):
  does the method separate this agent's successes from its failures.
- Brier skill and log-loss skill against the constant, 1 - loss / loss(constant): are the
  predicted probabilities usable as numbers. The question asks "how reliably", so a method that
  ranks well but is miscalibrated answers only half of it.

**For comparison with the original paper only.** Pooled AUC gain over the constant.

## 3. Scenarios

| scenario | split | role |
|---|---|---|
| random cases | 5-fold over tasks, 5 seeds | sanity check only: tasks of one procedure land in train and test |
| new procedures | 5-fold over procedures, 5 seeds | **primary** |
| new domain | leave one domain out | **primary** |
| across benchmarks | train on two benchmarks, test on the third (beta z-scored per benchmark) | **primary**, needs no shared agents |

tau-Knowledge banking has one domain and one task per group, so it enters through random cases and
the cross-benchmark test only. The summary number is the mean primary rho over the four
within-benchmark new-process scenarios (SOPBench and tau2, new procedures and new domain).

## 4. Uncertainty

1,000 bootstrap draws over procedures (tasks of one procedure are correlated), on seed 0. The draws
of the two splits of one benchmark share their procedure resample, so the pooled interval and all
differences between methods are paired. Point estimates are means over seeds.

## 5. Can the target and the features be trusted?

| check | SOPBench | tau2 | banking |
|---|---|---|---|
| beta refitted with two other IRT seeds, within-domain rho with the reference fit | 0.99, 1.00 | 0.99, 1.00 | 1.00, 0.99 |
| split-half reliability of beta (agents split in two halves, 3 times) | 0.80 | 0.79 | 0.93 |
| the same for all agents (Spearman-Brown) | 0.89 | 0.88 | 0.96 |
| ADeLe re-annotation by the same judge: levels identical | 92% | 89% | 91% |
| ADeLe re-annotation: levels within one step | 99% | 99% | 100% |

The target is stable: the correlation any predictor can reach with it is bounded near 0.9, so the
low scores below do not come from a noisy target. Against the GPT-4o levels shipped with the ADeLe
battery (100 instances x 18 demands), the Gemini judge agrees within one level 79% of the time
(median Spearman over demands 0.62). Two demands do not replicate: AS (Spearman 0.05) and MCu
(0.00). The variant without them (`adele16`) is evaluated as a robustness check.

## 6. Results

Mean primary rho over the four new-process scenarios; "worst" is the lowest of the four;
the last column is the paired 95% interval of the difference to the baseline.

| method | SOP new proc | SOP new domain | tau2 new proc | tau2 new domain | mean [95% CI] | worst | minus baseline [95% CI] |
|---|---|---|---|---|---|---|---|
| **Agent psychometrics + ADeLe demands (baseline)** | 0.120 | 0.125 | 0.128 | 0.150 | **0.131 [0.027, 0.255]** | 0.120 |  |
| kNN router | 0.010 | 0.026 | 0.306 | 0.154 | 0.124 [-0.001, 0.181] | 0.010 | [-0.185, 0.123] |
| same features, fitted jointly as LLTM | 0.121 | 0.131 | 0.104 | 0.097 | 0.113 [0.020, 0.237] | 0.097 | [-0.044, 0.030] |
| estimated human time (reference) | 0.093 | 0.131 | 0.079 | 0.124 | 0.107 [-0.002, 0.283] | 0.079 | [-0.131, 0.108] |
| embedding and ADeLe models, logit average | 0.144 | 0.079 | 0.168 | 0.011 | 0.101 [0.016, 0.211] | 0.011 | [-0.089, 0.035] |
| baseline without AS, MCu | 0.124 | 0.126 | 0.050 | 0.072 | 0.093 [0.001, 0.237] | 0.050 | [-0.064, 0.010] |
| their grouped ridge, embedding + ADeLe | 0.075 | 0.054 | 0.120 | -0.009 | 0.060 [-0.052, 0.181] | -0.009 | [-0.140, 0.003] |
| kNN router + IRT-Router-style MIRT + embedding | 0.027 | -0.005 | 0.317 | -0.104 | 0.059 [-0.021, 0.167] | -0.104 | [-0.210, 0.077] |
| text length (control) | -0.020 | 0.032 | -0.051 | 0.211 | 0.043 [-0.187, 0.230] | -0.051 | [-0.318, 0.046] |
| IRT-Router-style MIRT | 0.013 | -0.031 | 0.289 | -0.131 | 0.035 [-0.047, 0.178] | -0.131 | [-0.232, 0.047] |
| Agent psychometrics, embedding only | 0.082 | -0.006 | 0.065 | -0.172 | -0.008 [-0.110, 0.171] | -0.172 | [-0.275, 0.086] |

Per scenario (`final_evaluation.txt`): the baseline's interval excludes zero on both SOPBench
scenarios ([0.009, 0.229] and [0.024, 0.238]). tau2 has only three domains of 50 to 114 tasks, and
its telecom tasks form just three groups, so every method's interval there is wide. No method is significantly better than the baseline in any new-process
scenario. The router combination is significantly worse on SOPBench new procedures ([-0.238,
-0.023]), and the grouped ridge of Agent psychometrics on tau2 new domain ([-0.313, -0.024]).

Across benchmarks, rho on the held-out benchmark:

| method | SOPBench | tau2 | banking |
|---|---|---|---|
| **Agent psychometrics + ADeLe demands (baseline)** | 0.097 [-0.065, 0.221] | 0.165 [-0.001, 0.348] | 0.476 [0.284, 0.630] |
| estimated human time (reference) | 0.170 [0.032, 0.272] | 0.200 [0.034, 0.394] | 0.582 [0.441, 0.692] |
| text length (control) | 0.211 [0.049, 0.314] | 0.166 [-0.107, 0.470] | 0.532 [0.352, 0.673] |
| Agent psychometrics, embedding only | 0.084 [-0.091, 0.211] | -0.094 [-0.231, 0.142] | 0.356 [0.156, 0.506] |

## 7. The baseline and why this one

**Baseline: Agent psychometrics (Ge et al., COLM 2026; their code unchanged: 1PL IRT on the
training tasks, then ridge from task features to difficulty) with the 18 ADeLe demand levels (Zhou
et al., Nature 2026; their rubrics, prompt and parser) as the task features.**

Four requirements were set before the comparison:
1. a published method with public code, so anyone can rerun it;
2. it reads only the documentation: no runs of the new procedure, no hidden environment state;
3. it predicts for any agent that has runs on other tasks;
4. among the methods that meet 1 to 3, the highest mean primary score over the four new-process
   scenarios, without collapsing in any of them.

The baseline has the highest mean (0.131) and also the highest worst case (0.120). The kNN router
comes close on the mean (0.124), but all of it is earned on tau2, where tasks of a domain resemble
each other; on both SOPBench scenarios it is near zero (0.010 and 0.026). A procedure with no
similar neighbours in the training data is exactly the case the baseline is for.

A combination of two papers is a fair baseline here because neither part is changed and the way
they are joined is the one Agent psychometrics prescribes for any task features. The baseline
takes the measurement of a task from ADeLe and the model that turns a measurement into a
probability from Agent psychometrics.

Why not the other families:

- **Agent psychometrics with a text embedding** (the paper's main recipe). It hardly ranks tasks of
  a new procedure (mean -0.008) and is unstable: rerunning the upstream code without fixed seeds
  moved its tau2 new-domain score from 0.096 to -0.172, while the baseline moved by at most 0.005
  in any scenario. The embedding recognises which procedure a task belongs to rather than what
  makes it hard.
- **Their grouped ridge over embedding + judge features**, with ADeLe in place of their judge:
  lower on every new-process scenario and significantly worse on tau2 new domain; the embedding part
  overfits to known domains.
- **Their own LLM-judge features.** The rubric is written for code (for example solution hints and
  codebase scope) and judged by Claude Opus 4.6. Applying it to procedures would need a new rubric,
  and a procedure-specific rubric is what the proposed method is meant to be, not a baseline.
- **ADeLe's own predictor.** ADeLe trains an assessor on demand levels to predict one system's
  success, which needs many runs of that system. Here one difficulty estimate has to serve every
  agent, which the IRT model provides.
- **Routers** (kNN, Li 2025; IRT-Router, Song et al., ACL 2025). Best by pooled AUC and on seen
  procedures, and the highest score in one new-process scenario (tau2 new procedures, 0.317, with a
  paired interval that still touches zero). On SOPBench they are near zero or below, and the
  combination is significantly worse than the baseline on new procedures. They transfer what was
  learned on similar tasks, and a new procedure may have none.
- **Joint fitting** (amortised IRT, Truong et al., ICML 2025; LLTM, Fischer 1973). Same features,
  weights fitted together with the abilities. Close and not significantly different, in line with
  Ge et al., who found the two-stage fit better (their Appendix C.4). Kept as a robustness check.
- **Estimated human time** (the idea of METR's time horizon, Kwa et al., NeurIPS 2025). A good
  reference and, together with text length, the strongest across benchmarks, but METR measures real human time and an LLM
  estimate of it is our adaptation, not a published predictor. Reported next to the baseline.
- **Screened in a preliminary round, code not included here:** token entropy of a reasoning trace
  (Krsteski and Meyer's best signal), linear probes on hidden states of Qwen3-8B (Zhu et al., EMNLP
  2025), zero-shot success forecasts by an LLM, process-model complexity metrics from BPM (Cardoso
  2008; Mendling et al. 2008), and a fine-tuned text regressor. Entropy and the LLM forecasts were
  near zero; the complexity metrics and the fine-tuned regressor, which could only be run on
  SOPBench, stayed below the baseline there; the last-layer probe came closest (mean about 0.11)
  but the result depends on a layer chosen after seeing test data (the middle layer is negative on
  tau2) and does not transfer across benchmarks. They need a local 8B model or long fine-tuning,
  so their code is not in this repository and these statements cannot be reproduced from it.

What the numbers do not support:
- that the baseline is the best method: differences to the LLTM variant, estimated human time, the
  kNN router and the logit average are not significant;
- that it is good: rho 0.131 against a ceiling near 0.9, pAcc about 0.54 against 0.50 for chance,
  and Brier skill between -0.021 and 0.014 in the new-process scenarios, so its probabilities are
  barely better than agent ability alone;
- that it wins across benchmarks: there text length and estimated human time do as well or better.

The baseline was selected as the best of eleven methods on the same data, so its score is
slightly optimistic. For testing a new method against it, this is the conservative direction.

## 8. Rule for "a new method improves on the baseline"

Fixed before any new method is tested:
1. Mean primary rho over the four new-process scenarios is higher, and the paired pooled 95%
   interval of the difference lies above zero.
2. No single new-process scenario has a paired interval entirely below zero.
3. Across benchmarks, no held-out benchmark where the new method is significantly worse.
4. Brier and log-loss skill not lower than the baseline's in the new-process scenarios.
5. Confirmed once on a held-out benchmark that was not used for any choice, with the method and all
   its settings frozen before that run.
