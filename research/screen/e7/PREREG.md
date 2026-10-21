# E7: does the method work outside business procedures? (pre-registration, 13 Oct 2026)

Written before any E7 forecast was requested. Plan item E7 (approved 13 Oct): transfer to Terminal-Bench 2.0, SWE-bench Verified
and other agent benchmarks with public per-task results. Toolathlon and METR have no per-task agent results we can use; in their
place we take the two benchmarks the baseline paper (Agent psychometrics) itself uses as held-out tests, SWE-bench Pro and GSO.
TheAgentCompany, MCPMark and DrafterBench were tested in round 10 (`../ideas10/fresh`) and are not repeated.

## What transfers

Our trained models use features that exist only for policy-driven tasks (plan with tools, data gap, compiled policy), so they cannot
run on code tasks. The part that can transfer is the zero-shot forecaster: one call to Sonnet 5.5 (`claude -p --model sonnet`, no
tools) that reads the task and returns the probability that a typical agent fully succeeds (the "direct question" D of round 10,
`../ideas10/ablate`). Score = logit(1 - p), p clipped to [0.02, 0.98]. No training on any benchmark.

Prompt: `e7.py` (`PROMPT`, `ENV`). The text shown is the statement-only level of Agent psychometrics: problem statement for SWE-bench
Verified and Pro, task instruction for Terminal-Bench, repository plus API name for GSO (their `problem_info` formatters).
Statements are cut at 15,000 characters.

## Data

Targets: 1PL difficulty `b` from Agent psychometrics (`data/<bench>/irt/1d_1pl/items.csv`). Terminal-Bench: the binomial per-attempt
fit (`irt_binomial`, 139 agents) is primary, the binary fit is reported too. Task texts: Hugging Face datasets-server (SWE-bench
Verified 500, SWE-bench Pro 642 rows of which 641 match the 730 IRT items, GSO 102); Terminal-Bench `tasks.jsonl` (89).

## Analyses

1. Main metric, per benchmark: Spearman between the score and `b` over all tasks, 95% interval by bootstrap over tasks (2,000).
2. Comparators at the same information level (statement only):
   - length of the shown text;
   - the 15 statement-only LLM-judge features of Agent psychometrics (`1_problem_15.csv`) with ridge, 5-fold cross-validation
     inside the benchmark (5 seeds, alphas 0.01 to 1e5): out-of-fold predictions;
   - the same with their statement-only embeddings;
   - SWE-bench Pro and GSO only: their held-out-benchmark judge predictions (`results/table5_statement_only/<bench>/predictions.csv`),
     the setting closest to ours (no training on the target benchmark).
   Paired bootstrap over tasks for the difference between D and each comparator.
3. Combination: ridge on the 15 judge features plus D, same cross-validation.
4. Table 5 format: their own pipeline (`experiment_new_benchmarks.run_all_datasets`, held-out SWE-bench Pro and GSO, statement only),
   with D as a one-column judge feature file, and with the 15 judge features plus D. Reported AUC next to their statement-only rows
   (judge 0.652 and 0.683, combined 0.653 and 0.699, baseline 0.571 and 0.637).

Main hypothesis: on every benchmark D ranks tasks at least as well as the statement-only judge ridge trained inside the benchmark,
and better than length. Prediction from round 10 (TheAgentCompany 0.46, MCPMark 0.25, DrafterBench 0.23): positive on all four,
weakest on GSO, where the text is only a repository and an API name.
