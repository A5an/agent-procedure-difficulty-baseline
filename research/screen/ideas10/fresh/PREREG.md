# PREREG fresh-benchmark test of the zero-shot pre-mortem (round 10, 8 Oct 2026, about 00:55)
Written before computing any task-level outcome or feature on these benchmarks. Data from the scout (scout/REPORT).

Question: does a pre-mortem rating by an LLM, with NOTHING fitted, rank task difficulty for agents on benchmarks never
used in this project? On tau2 it gave within-domain 0.517 and on banking 0.596 (fc/), but those were seen.

Benchmarks and targets (difficulty = 1 - mean over agents of the per-agent outcome; Spearman, bootstrap over tasks):
- TheAgentCompany (TAC): 175 tasks, the 17 agents with >= 170 tasks; outcome = final_score result/total (partial
  credit), task text = workspaces/tasks/<id>/task.md (checkpoints.md is gold and is NOT used). Primary.
- MCPMark v1-0905: 127 tasks, the agents with >= 120 tasks (k2 duplicates dropped: keep kimi-k2-0905 only);
  outcome = mean pass over runs; task text = description.md. Primary.
- DrafterBench: secondary, a stratified sample of 20 cases per task type (240 cases, seed 0), 30 agents, outcome =
  eval score >= 1.0. Within task type and overall.

Predictors (all zero-shot, computed once, reported all):
- FCg: Sonnet 5.5 via claude -p, generic pre-mortem prompt (fresh/fcg.py, same structure as fc/fc.py with the
  customer-service wording replaced by a neutral description of the benchmark environment and grading), score
  fc_diff = logit(1 - p_success). PRIMARY predictor.
- GG: the same generic prompt on gemini-3.8-flash.
- length: characters of the task text. HT: the repo human_time prompt on gemini-3.8-flash (log minutes).
Primary test: Spearman(FCg, difficulty) on TAC and on MCPMark, and paired difference vs length and vs HT.
No prompt changes after the first look at outcomes.

## Round 2 (added about 01:25, after seeing round-1 fresh results: FCg TAC 0.458, MCPMark 0.272, GG 0.252 / 0.347,
## HT 0.444 / -0.059; drafter incomplete)
Motivation from tau2 and banking only (fcx/): the QA-lead prompt P2 worked for both model families, Gemini with P1 was
weak, averaging two Sonnet prompts helped on banking. Not chosen from the fresh results.
New sources on TAC and MCPMark (and drafter if time allows): SP2 = Sonnet, generic P2; GP2 = Gemini 3.8 Flash, generic
P2; OP1 = Opus 5.5 (claude -p --model opus), generic P1.
Tests: (a) ENS3 = mean of within-benchmark z-scores of FCg (Sonnet P1), SP2 and GP2 vs FCg alone, paired bootstrap;
(b) OP1 vs FCg; (c) each source alone. Same targets and evaluation as round 1. Report all.
