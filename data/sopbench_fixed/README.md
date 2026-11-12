# SOPBench-fixed (10-11 Nov 2026)

A repaired version of SOPBench (commit d2622008, `prototype/data/external/SOPBench-d2622008`) and every run made on it.
Code lives in `prototype/screen/sopaudit/` (fixsop.py, build_fixed.py, runfixed.py); reports there: REPORT.md,
REPORT2.md, REPORT3.md.

## What was wrong with SOPBench and what was done

Grader (fixed in `fixsop.fixed_eval`, applied when scoring, the benchmark code is untouched):
1. `try_eval` runs Python `eval()` on tool output: `'2024-11-30'` becomes 1983, `'108'` becomes 108.
2. The same `eval()` on JSON arguments: `true`, `false`, `null` are not Python, the call counts as a tool error.
3. The harness keeps only the value of a `(True, value)` result; the grader then sees no successful call, so every
   PERFORM case whose action returns a value (exchange, renew, get_*, show_*, check-in) is unpassable. Older published
   runs stored the full tuple, newer ones (gpt-5, o4-mini, Claude 3.7, Gemini 2.5) did not: the leaderboard mixes both.
4. The action graph of hotel book_room / modify_reservation demands a loyalty lookup the policy makes optional.
5. The library policy lists `internal_get_interaction_date` but the agent is not given that tool.

Tasks (`cases.json`, one entry per kept case: `policy_append`, `user_prompt`, `user_known`, `notes`):
6. 21 unsolvable cases dropped (`unpass.json`: target tool missing, customer message contradicting the case), plus 3
   open_account cases whose conditions contradict each other.
7. Conditions the grader checks but the policy text never states (built-in dependencies such as "the book exists",
   "the old booking is confirmed", "amount > 0"): 1423 lines appended in 629 cases.
8. 19 cases where the required checks need a login but the customer gives no credentials: credentials added
   (labels verified unchanged with the strict system).
9. 74 customer messages inconsistent with the case values (other name, item, dates, "UNKNOWN_PLACEHOLDER", ambiguous
   time ranges): rewritten. Missed by the check: a password with the period inside the quotes, and a plain ID string
   called "my driver's license number" (4 cases).

## Files

- `cases.json`, `stats.json`, `unpass.json`, `promised_missing.json`, `case_subsets.json` (broken, message or
  credential defects, refuse only by an unwritten condition)
- `published_agents_responses_orig_and_fixed.json`: the 28 published agents of the baseline repo, every case,
  published label and label under the fixed grader (1032 of 23 240 labels change)
- `runs/`: gemini-3.8-flash and gemini-2.5-flash on SOPBench-fixed (806 cases each), gemini-3.8-flash on the original
  tasks (fresh), counterfactual runs (unchanged x3, explicit x3, with credentials x3), our 4 agents on the original
  tasks re-scored (`rescored_our_4_agents_original_bench.jsonl`)
- `verdicts/`: why each failure happened: Gemini judge, 9 independent Opus judges, manual labels for 54 cases

## Results

| agent | original tasks, official | original tasks, fixed grader | SOPBench-fixed |
|---|---|---|---|
| gemini-3.8-flash | 0.871 | 0.943 | 0.976 |
| gemini-2.5-flash | | | 0.857 |
| Claude Sonnet 5.5 (Claude Code headless) | 0.766 | 0.848 | 0.885 |
| Claude Haiku 5.5 (Claude Code headless) | | | 0.883 |

On SOPBench-fixed the agent's own errors (Opus judges): gemini-3.8-flash 13-14 of 806 (1.6-1.7%), gemini-2.5-flash 109
(13.5%), Sonnet 5.5 74 (49 of them one borderline pattern: ownership checked with get_order_details itself), Haiku 5.5
84 (57 of that pattern). They are almost all about checks: acting without a stated check or before that tool's own login condition,
verifying and then not acting, misreading a result (empty list of valid products), wrong arguments.
