# Exp 2: a failure monitor at the commit point (3 Nov 2026)

No API calls. 26,111 recorded episodes: SOPBench release logs (16,109, 22 function-calling configs), tau2-bench airline
and retail submissions (9,762, 15 submissions x 4 trials), our TauIndianBankBench real runs (240, 2 agents).
Code: `build.py` (features), `eval.py`, `within_task.py`; outputs `episodes.csv`, `eval.out`, `within_task.out`,
`results.json`. Literature check: `LIT.md`.

## What the monitor sees

The episode is cut at the agent's first state-changing call (SOPBench: the case's target action; tau2 and TauIndianBank:
their write tools), or at its end if it never writes. 13 features from the prefix: whether it is about to write, number
of calls, distinct tools, tool errors, calls with empty arguments, repeated calls, user and agent turns, length, whether
the last user turn is a confirmation, whether the agent asked, number of arguments of the write, and the share of the
write's argument values already seen in tool outputs or user turns ("grounding"). Models: logistic regression and
gradient boosting. Label: the episode failed.

## Results (within-agent AUC removes "stronger agents fail less"; within-task AUC compares episodes of the same task)

| setting | AUC | within-agent AUC [95%] | failures caught by escalating the riskiest 20% |
|---|---|---|---|
| inside SOPBench (5-fold by task) | 0.84 | 0.82 [0.80, 0.83] | 34% |
| inside tau2 | 0.64 | 0.64 [0.59, 0.68] | 36% |
| inside TauIndianBank (240 episodes) | 0.84 | 0.84 [0.78, 0.89] | 40% |
| train on tau2 + TauIndianBank, test SOPBench | 0.76 | 0.76 [0.75, 0.78] | 31% |
| train on SOPBench + TauIndianBank, test tau2 | 0.53 | 0.54 [0.49, 0.58] | 20% |
| train on SOPBench + tau2, test TauIndianBank | 0.44 | 0.45 [0.37, 0.53] | 10% |

(gradient boosting; random escalation catches 20% by construction.)

Beyond task difficulty (the trap the literature check flags: pooled AUC mostly measures how hard the task is):

| | within-task AUC of the monitor | pooled AUC: task difficulty from other agents only | difficulty + monitor |
|---|---|---|---|
| SOPBench | 0.83 (769 tasks) | 0.76 | 0.91 |
| tau2 | 0.63 (155 tasks) | 0.78 | 0.73 |
| TauIndianBank | 0.70 (38 tasks) | 0.67 | 0.84 |

## Reading

Inside a benchmark the monitor works: on SOPBench it separates failing from passing episodes of the same task at 0.83
and adds a lot to task difficulty; on TauIndianBank it does about as well on a small sample; on tau2 it is weaker
(0.63 within task, no gain over difficulty, similar to the 0.66 to 0.70 airline-to-retail number in the literature).

Across benchmarks it does not transfer to tau2 or TauIndianBank (0.44 to 0.54). The same feature means different
things: on SOPBench "about to write" raises the risk (most cases there should be refused), on TauIndianBank it lowers
it (failures there are mostly missed writes). So a monitor has to be trained on logs of the deployment it guards,
which fits the shadow-mode stage: the shadow cases are its training data.

## Limits

Structural features only, no text model; one decision point; SOPBench dominates the pooled training set. The
two-model disagreement signal of E8 is not in this table (only 150 E8 cases); there it was 60-80% errors on
disagreement against about 25% on agreement.
