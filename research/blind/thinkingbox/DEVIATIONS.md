# Deviations from PREREG.md and run notes (written during the run, before the analysis)

None of these touches the predictors (`forecast.py`, `baseline.py`, their committed predictions) or the analysis
(`evaluate.py`). All of them concern how the target runs were produced.

1. **Pilot launch.** The first pilot launch failed on a missing data path (`TB_DATA` not set) before any episode ran;
   it was started again.
2. **Reasoning off for the user simulator and the judge.** Implemented in the proxy as a thinking budget of 0.
   `reasoning_effort: minimal`, tried first in a smoke test on the harness's own non-benchmark scenario, still spent
   about 120 thinking tokens and left the judge's 128-token answers empty.
3. **gemini-2.5-flash-lite.** In the pilot most of its turns ended with `finish_reason: malformed_function_call`
   (an empty reply). Replaying five pilot requests twice each gave a malformed call in 9 of 10 calls with reasoning on and in 8
   of 10 with reasoning off, so the agent was kept as registered. It solves almost nothing on this benchmark.
4. **Concurrency and a spending cap.** Main run with the four agents in parallel, 6 tasks each; the proxy refuses calls
   above a spending cap (190 USD in total, reset to the remainder at each proxy restart). gemini-3.8-flash cost about
   0.15 USD per run in the main run against 0.076 in the pilot (retries after Vertex failures and longer episodes), so
   the projected total rose to about 207 USD; with the user's approval the cap was raised to 220 USD at 18:30.
5. **Agent turn limit (14:30).** A gemini-2.5-pro episode repeated one tool call hundreds of times; the harness has no
   limit on agent turns by default. The proxy now answers an agent request that already holds 100 tool-call turns
   with a final text reply, which ends the agent's loop. No finished episode before then had more than 36 such turns.
   The limit fired once (one gemini-2.5-pro episode); it counts as an agent failure (PREREG: "the turn limit").
6. **Proxy restarts.** The proxy was restarted at 14:30 (turn limit), 14:45 (masking of the project number in error
   messages), 16:30 (keep-alive sessions, error details in the ledger). Calls in flight failed; the affected runs are
   harness system errors and were run again.
7. **gemini-3.8-flash latency and restarts.** From about 14:15 the global endpoint (the only one that serves this
   model) answered in a median of 25 to 67 s instead of 5 s. Its runner was restarted with 16, then 32, then 10, then
   16 tasks in parallel. With 32 (38 in total) the proxy saw 233 connection errors to Vertex between 15:20 and 16:25
   and 128 runs of this agent ended as system errors; they were run again.
8. **Retry bug (fixed at 17:08).** Rows of crashed runs carry `repetition: None`; `tb infer --previous-results-file`
   drops every previous row after the first row that does not match the current test, so every finished run after the
   first crash was run again. This happened for gemini-2.5-flash-lite (its retry pass re-ran almost all runs) and for
   part of gemini-3.8-flash. Since 17:08 each retry pass gets a resume file from `run/resume.py`, and
   `run/collect.py` builds the outcomes with the same rule: each (trial, task) slot takes the first valid run in the
   agent's output files, oldest file first. The rule depends on file order only, never on the outcome.
9. **What was seen during the run.** The harness's progress lines print the running share of passed runs per agent,
   so aggregate success rates were visible. Task-level outcomes were looked at for the five pilot tasks (not part of the
   evaluation) and for one main-run task (a gemini-2.5-flash failure next to a gemini-3.8-flash success) to check that the failure came from the model and not from the harness.
