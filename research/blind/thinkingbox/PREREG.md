# Blind test on ThinkingBox-Bench v1.0: pre-registration (15 Oct 2026)

This file, the predictor (`forecast.py`), the comparators (`baseline.py`), the task inputs (`build_items.py`) and the
analysis (`evaluate.py`) are committed before any agent has been run on ThinkingBox-Bench by us and before we have
seen any task-level outcome on it. No task-level outcomes of this benchmark are public (the paper, arXiv 2608.19741,
reports aggregates only). The benchmark has not been used for any choice in this project. It is point 5 of the
improvement rule in `EVALUATION_PROTOCOL.md` §8: one confirmation on a held-out benchmark with the method frozen.

## Data

ThinkingBox-Bench v1.0 (Microsoft): 507 tool-agent-user tasks in five domains (retail 98, travel 104, auto insurance
100, neobank support 104, consulting IT/HR 101). Versions, fixed by `fetch.sh`: `thinkingbox` 892964e,
`thinkingbox-data` tag thinkingbox-bench-v1.0 (fcaba4c), Hugging Face dataset revision 5e49f89 with SHA-256 checks.
The release asks that the benchmark be used for evaluation only; we use it only to evaluate a frozen predictor.

What the predictors read (`build_items.py`): per task, the user's first message and the simulated user's
instructions; per domain, the agent's system instructions, the policy knowledge base the agent can search, and the
tool names. Never read: initial state patches, expected tool interactions, rubrics, golden test cases.

## Predictor (frozen)

D: one call per task to Sonnet 5.5 (`claude -p --model sonnet`, resolved to claude-sonnet-5-5, no tools) with the
prompt in `forecast.py`: the tau2 direct-question prompt of round 10 with four wording changes for this benchmark,
listed in the file. Score = logit(1 - p_success), p clipped to [0.02, 0.98]; larger is harder. Nothing is fitted.
All 507 predictions are computed and committed (`predictions_D.csv`, with the SHA-256 of every prompt) before the
first agent run. A task with no parsable answer after six attempts gets the median score.

## Comparators (frozen)

- baseline: 18 ADeLe demand levels per task (`src/adele_annotate.py` unchanged, Gemini 3.8 Flash, temperature 0) and
  the leave-one-benchmark-out ridge of `src/lobo.py`, fitted on SOPBench, tau2 and tau-Knowledge banking.
- length: the same ridge on log characters of the task text.
Both are computed and committed (`predictions_baseline.csv`) before the first agent run.

## Target: our own runs

- Harness: ThinkingBox `tb infer` with its agent `think`, unchanged.
- Agents: gemini-2.5-flash-lite, gemini-2.5-flash, gemini-2.5-pro, gemini-3.8-flash on Vertex AI (OpenAI-compatible
  chat completions), temperature 1.0 and medium reasoning effort as in the harness's default configuration, output
  limit 16,384 tokens per turn (the harness default of 4,096 truncated tool calls in our E8-0 check).
- Two trials per agent and task: 8 runs per task.
- User simulator and rubric judge: gemini-2.5-flash with reasoning off; temperature 0.3 for the user and 0 for the
  judge, as in the harness defaults.
- Task set: the first N tasks of each domain in `order.csv` (a random order per domain, seed 20261015, written now).
  N is set after the pilot from cost alone: all tasks if the projected cost of the main run is at most 150 USD,
  otherwise the largest N with projected cost at most 150 USD. If N would fall below 40, we stop and ask before
  spending more. N is written to `eval_set.json` before the main run.
- Pilot: the last task of each domain in `order.csv` (5 tasks), each agent once. It checks the machinery and measures
  cost per run. Pilot runs are not used in the evaluation. After the pilot we may change only run settings that do not
  touch the predictors or the analysis (endpoints, timeouts, concurrency, token limits); every change is logged in the
  report as a deviation.
- Validity: a run that ends in an infrastructure error (an API error after retries, a harness system error) is rerun
  up to two times and is invalid if it still fails. A run where the agent itself fails (wrong state, a bad tool call,
  an empty reply, the turn limit) is a failure, not an invalid run. A task enters the evaluation with at least 6 valid
  runs of 8.

## Analysis (frozen, `evaluate.py`)

Difficulty of a task = 1 - mean success over its valid runs. Primary metric: Spearman rho between score and
difficulty inside each domain, pooled over the five domains with weights = number of task pairs (the protocol's
within-domain rho). 95% intervals from 2,000 bootstrap draws that resample task clusters inside each domain (retail
template STnnn; single tasks elsewhere), paired across predictors.

Primary test (point 5 of §8): D minus baseline in pooled rho, with the 95% interval above zero.
Secondary, reported in full: D minus length; D's own rho and interval; rho by domain; pairwise accuracy; rho against
each agent's own success; Brier of p_success against the observed success share (D is not calibrated to these agents,
so this is descriptive).

## Expectations written now

From tau2 (D 0.50 and 0.57 in the two new-process scenarios), tau-Knowledge banking (0.60 for the pre-mortem version)
and the fresh benchmarks of round 10 (0.23 to 0.46 for the pre-mortem version): D around 0.3 to 0.5. Baseline: the leave-one-benchmark-out ADeLe ridge reached 0.10, 0.17 and 0.48 on
the three development benchmarks, so 0.1 to 0.3 here. A result outside these ranges is reported as it is.

## After the test

The result is reported whatever it is. ThinkingBox is not used afterwards to choose or tune anything in this
project. A new predictor version would need a different held-out benchmark.
