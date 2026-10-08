# Emulated pilot with real agent runs (agent `emulate_run`, 5 Oct 2026)

Saved by the main session from the agent's final message. About 33 USD (episodes 28.3, generation 4.5).
Files: gen_all.py, runner.py, collect.py, analyze.py, extra.py, synth_all/, gen_all_rows.json, gen_all_stats.json,
gen_all_meta.json, runs/<run>/results.json, runs/<run>/usage_*.jsonl, runs/<run>/out/, proc_table.csv,
analysis_synth_lite_real_lite_synth_flash_real_flash.json, extra_diagnostics.json.

## Pre-registered rule
Procedure-level within-domain Spearman >= 0.40 against mean IRT difficulty, bootstrap lower bound > 0.23.

| agent | estimate 0.35 x perform + 0.65 x refuse failure | rule |
|---|---|---|
| gemini-2.5-flash-lite (named first) | 0.390 [0.099, 0.594] | FAIL |
| gemini-2.5-flash | 0.557 [0.322, 0.725] | PASS |

Conditional pass: the rule did not name the agent, flash is the second look. Flash with raw failure instead of the weighted
estimate: 0.412 [0.157, 0.631], misses the lower bound. Lite gets only 4.9% of real perform cases right.

## What ran
SOPBench copied without output/, Vertex route, token caching, usage logging with thinking, retries, timeouts, empty-parts
fix, function call found in any part. Temperature 0, top_p 0.01, max_tokens 4096, one attempt, scripted user,
run_evaluation.py. 8 workers, sharded resumable runner, no episodes lost, every case scored.
Smoke S0 (379 cases, flash-lite): evaluator works on synthetic cases, no refuse case passed by an empty answer.
Generation (gemini-3.8-flash, 231 calls): 231 policy variants of 68 usable procedures, 2,257 of 2,310 valid (98%), 1,365
kept (up to 6 per variant, 49% perform, estimator reweights by class), median 12 cases per procedure.

## Fidelity: synthetic vs real, same agent and harness (all 830 real cases run for both agents)

| agent | estimate | raw failure | per class within domain (perform / refuse) |
|---|---|---|---|
| flash | 0.650 [0.451, 0.812], overall 0.701 | 0.677 [0.484, 0.829] | 0.77 / 0.49 |
| flash-lite | 0.092 [-0.199, 0.356] | 0.155 [-0.135, 0.417] | 0.20 / -0.07 |

Flash: perform success 51% synthetic vs 55% real, refuse 88% vs 83%. Lite at a floor on perform cases.
Real flash vs IRT difficulty 0.525 [0.284, 0.699], real lite 0.464. Synthetic flash 0.557 vs real flash 0.525.
Against the 25-agent public mean failure: flash synthetic 0.554, lite 0.454. Case-resampling noise: flash 0.506
[0.404, 0.604], lite 0.351 [0.192, 0.505].

## Limitations
Two agents examined, only one passes. Same recipe as SOPBench's own cases (friendliest test). Wide intervals (68
procedures). Vertex 4096 route not comparable to public AI Studio 512 runs. Retry at temperature 0.7 not counted.
Harness parsing fixes apply equally to synthetic and real runs.
