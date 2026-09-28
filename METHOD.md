# Method

The question: before an agent is deployed on a procedure, predict the probability that it executes
a task of that procedure correctly, using only the procedure's documentation and the task request,
with no runs of that procedure. This file describes each step of the baseline and where it comes
from. The metrics and the choice of the baseline are in `EVALUATION_PROTOCOL.md`.

## 1. Data: agents, tasks, outcomes

A data point is one attempt of one agent on one task, with a binary outcome.

| benchmark | tasks | task groups | agents | attempts per cell | outcome |
|---|---|---|---|---|---|
| SOPBench (Li et al. 2025, arXiv 2503.08669) | 830 | 70 procedures in 7 domains | 28 configurations (LLM x prompting mode) | 1 | SOPBench's evaluator |
| tau2-bench (Barres et al., ICML 2026, arXiv 2506.07982) | 278 | airline and retail tasks; 3 resolution paths in telecom | 15 leaderboard submissions | up to 4 | reward = 1 |
| tau-Knowledge banking (banking_knowledge domain of tau2-bench) | 97 | single tasks | 12 leaderboard submissions | up to 4 | reward = 1 |

**SOPBench.** A task is a customer request in one service domain (bank, DMV, healthcare, hotel,
library, online market, university). The agent must follow the written procedure for the
requested action: call the required checks, in the order the procedure defines, then act or
refuse. A run passes only if all five checks of the evaluator hold (no tool error, no rule broken,
final database equal to the expected one, the action done if allowed and refused if not, required
checks called in order). `build_sopbench.py` re-scores every released run with SOPBench's own
evaluator (`evaluator_function_directed_graph`, the call their `run_evaluation.py` makes), because
1,939 of the released runs carry no saved flag. All 21,296 saved flags agree with the re-score.
Five runs are absent from the release and are left missing, not scored as failures.

**tau2-bench and tau-Knowledge.** A task is a conversation with a simulated customer. Only the end
state is graded: the database after the conversation must equal the database produced by the gold
actions, plus a per-domain check. Every attempt is kept: a cell is (successes, trials), and the IRT
uses a binomial likelihood (the Agent psychometrics path for multi-attempt data).

**What a predictor may read.** Each task is described by one text (`statements.jsonl`) built only
from what the agent sees:
- SOPBench: the domain role, the policy section of the requested action exactly as rendered in the
  agent's system prompt, the customer's first message, and the tool names. Never used: the
  evaluator tree, the gold action graph, the allow/refuse flag, the initial database.
- tau: the customer scenario given to the user simulator (reason for call, known and unknown
  information, instructions). Never used: evaluation criteria, gold actions, hidden initial state,
  the test purpose, and the telecom task ids, which name the injected faults. Task ids are replaced
  by opaque ids (`id_map.json`).

## 2. Target: task difficulty from item response theory

The one-parameter logistic model (Rasch 1960):

    P(agent a solves task t) = sigmoid(theta_a - beta_t)

theta_a is the ability of agent a, beta_t the difficulty of task t. Parameters are fitted by
stochastic variational inference with hierarchical priors, with the Agent psychometrics trainer
(`swebench_irt/train.py`, unchanged; `fit_irt.py`). The full-data fit gives the rank target of the
evaluation and the oracle. Inside cross-validation the IRT is refitted on the training tasks of
each fold only, so held-out outcomes never enter a prediction.

Source: Ge et al., Agent psychometrics, COLM 2026 (arXiv 2604.00594), Section 2.1.2.

## 3. Task features of the baseline: ADeLe demand levels

ADeLe (Zhou et al., Nature 2026, doi 10.1038/s41586-026-10303-2) defines 18 general demand scales,
each with a rubric of levels 0 to 5+ and anchor examples: 11 cognitive (for example AS attention and
scan, CEc verbal comprehension, QLl logical reasoning, MCr identifying relevant information), 5
knowledge (KNa, KNc, KNf, KNn, KNs) and 2 extraneous (AT atypicality, VO volume). An LLM reads the
task and the rubric and assigns a level. The authors annotate with GPT-4o and validate against
five human experts (Spearman 0.75 to 0.94).

Here: their rubrics, prompt builder and level parser from their repository, unchanged
(`adele_annotate.py`); the 18 levels of each task statement are the feature vector. The judge is
Gemini 3.8 Flash at temperature 0 instead of GPT-4o. Checks of that change:
- on 100 instances of their battery, compared with the GPT-4o levels shipped with it
  (`results/adele_validation.json`): AS and MCu do not replicate, so a variant without them
  (`adele16`) is evaluated as well;
- a second annotation of a sample of tasks (`adele_retest.py`): 89 to 92% of levels identical,
  99% within one level.

The unguessability scale (UG) is not used: it is defined for multiple-choice formats.

## 4. The baseline predictor

Two stages, with the Agent psychometrics code (`FeatureBasedPredictor`, unchanged):

1. Fit the IRT above on the training tasks of the fold; freeze their difficulties beta_t.
2. Ridge regression from the feature vector f_t to beta_t:
   `min_w,b sum_t (beta_t - w'f_t - b)^2 + lambda ||w||^2`, features standardised, lambda chosen by
   5-fold cross-validation inside the training tasks over {0.01, 0.1, ..., 100000}.
3. For a held-out task: beta_hat = w'f + b and P(success) = sigmoid(theta_a - beta_hat), with the
   ability theta_a from stage 1.

Source: Agent psychometrics, Section 3.2.1. They follow Truong et al. (ICML 2025), who fit the
linear model jointly with the abilities; Ge et al. report that the two-stage fit works better.
The joint fit with ADeLe features is the linear logistic test model (Fischer 1973; De Boeck and
Wilson 2004) and is evaluated as `lltm_adele`.

In short, the baseline combines two published pieces without changing either: the measurement of a
task (ADeLe) and the model that turns a measurement into a probability for a given agent (Agent
psychometrics).

## 5. Reference methods

All fitted on the same folds as the baseline (`run_baseline.py`).

| name | what it is | source |
|---|---|---|
| constant | agent ability only; every task equally hard | Agent psychometrics baseline |
| oracle | full-data IRT difficulty | Agent psychometrics |
| length | log(1 + characters) of the statement | Krsteski and Meyer 2026 control |
| emb_ap | their embedding recipe: last-token hidden state of DeepSeek-R1-Distill-Qwen over "task statement + difficulty instruction" (1.5B instead of their 32B, to run on a CPU) | Agent psychometrics, Appendix A.1 |
| ap_combined | their grouped ridge over embedding + ADeLe, one penalty per family | Agent psychometrics, Section 3.2.1 |
| emb_ap+adele | equal-weight average of the two models' logits | fixed combination |
| lltm_adele | ADeLe levels fitted jointly with abilities | Fischer 1973; Truong et al. 2025 |
| knn_router | an agent's success on the 10 most similar training tasks (bge-base-en-v1.5) | Li 2025, arXiv 2505.12601 (reimplementation) |
| amortized_mirt | 4-dimensional IRT with discrimination and difficulty predicted from the embedding | IRT-Router, Song et al., ACL 2025 (reimplementation) |
| knn_router+amortized_mirt+emb_ap | equal-weight logit average of the three | fixed combination |
| human_time | log of the minutes a trained employee would need, estimated by the judge | idea of METR's time horizon, Kwa et al., NeurIPS 2025 (our adaptation, so not a baseline candidate) |

## 6. Splits

| split | how | question it answers |
|---|---|---|
| random_cases | KFold over tasks, 5 folds x 5 seeds | seen procedures (tasks of one procedure share most text); sanity check only |
| new_procedures | GroupKFold over procedures, 5 folds x 5 seeds | a procedure never seen with outcomes |
| new_domain | leave one domain out | a whole domain never seen |
| across benchmarks | train on two benchmarks, test on the third, target z-scored per benchmark | a new benchmark; Krsteski and Meyer's leave-one-benchmark-out |

## 7. Metrics

The primary metric is the ranking of task difficulty inside a domain (Spearman rho and pairwise
accuracy), because pooled AUC over (agent, task) pairs mostly measures differences between agents:
the constant predictor, which knows nothing about tasks, already reaches AUC 0.713 on SOPBench with a
new domain. Details, secondary metrics, uncertainty and the decision rule: `EVALUATION_PROTOCOL.md`.
Source of the argument: Krsteski and Meyer, Predicting task difficulty without rollouts, 2026
(arXiv 2608.05797), Section 4.2.

## 8. Differences from the original papers

- The Agent psychometrics LLM-judge features (a rubric for code tasks, judged by Claude Opus 4.6)
  are replaced by ADeLe's general rubrics; no procedure-specific rubric is used.
- Their decomposition of agent ability into LLM and scaffold is not used: every agent
  configuration is seen in training, only tasks are new.
- The embedding backbone is 1.5B instead of 32B. Their Table 8 reports smaller backbones within
  about 0.01 to 0.03 AUC on SWE-bench Verified.
- ADeLe's own predictor (an assessor trained on the demand levels to predict a system's success) is
  not used; the demand levels enter the shared IRT model instead, so one fit serves all agents.
- Multi-attempt data use the binomial likelihood instead of a majority vote per cell.
