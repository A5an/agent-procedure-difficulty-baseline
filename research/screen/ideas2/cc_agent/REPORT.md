# Claude Sonnet 5.5 inside Claude Code as the SOPBench agent (agent `cc_agent`, 5 Oct 2026)

Saved by the main session from the agent's final message. Tables and paired tests: analysis_out.txt, results.json.

## Setup
- Agent: headless `claude -p --model sonnet` (claude-sonnet-5-5), user's subscription, locked to one Bash pattern
  (`python3 <abs>/sop_tool.py ...`), other tools disallowed, --permission-mode dontAsk, --setting-sources "".
- Bridge sop_tool.py: rebuilds the case env from the initial database each call, replays earlier calls, runs the new
  call, logs it, prints the return value. No gold fields, constraints, data paths or database are printed. Calls beyond
  10 refused (SOPBench max_num_actions). Opaque session names s000 to s149.
- Guard: harness's required_tools() verbatim, same error text. Blocked call not executed, counts toward the 10-call cap,
  scored as harness H2 with blocked attempts stripped.
- Cases: exactly harness/sample.csv (150). 4 runs in parallel, no rate or usage limit errors.
- Scoring: session log to tool calls to the unchanged evaluator_function_directed_graph (compile/execu.py approach).
  Re-scoring 160 public o4-mini-high trajectories through the same replay path matches 158 of 160 (2 mismatches in
  library/add_book). 3 cases checked by hand.
- Prompt: SOPBench's own system prompt from task_initializer, the tool schemas as JSON, the bridge command, "the customer
  will not reply, finish with a plain-text reply", the customer message last. exit_conversation replaced by the final reply.

## Main table (150 cases)

| arm | success | perform | refuse | decision acc | unsafe on must-refuse | wrongful refusal on must-perform | tool calls | guard blocks per case | equivalent cost |
|---|---|---|---|---|---|---|---|---|---|
| CC Sonnet native | 0.813 | 0.760 | 0.867 | 0.947 | 0.067 | 0.067 | 3.91 | 0 | $12.01 ($0.080/case) |
| CC Sonnet guard | 0.840 | 0.773 | 0.907 | 0.913 | 0.027 | 0.147 | 4.39 | 0.21 | $13.42 ($0.089/case) |
| public o4-mini-high | 0.727 | 0.640 | 0.813 | 0.827 | 0.173 | 0.187 | 4.31 | | |
| public median (gpt-4.1-mini) | 0.440 | 0.387 | 0.493 | 0.752 | 0.448 | 0.091 | 4.58 | | |
| compiled A | 0.720 | 0.493 | 0.947 | 0.773 | 0.013 | 0.440 | 3.29 | | |
| compiled AC | 0.847 | 0.773 | 0.920 | 0.933 | 0.027 | 0.120 | 4.03 | | |
| Gemini 3.8-flash H0 | 0.813 | 0.773 | 0.853 | 0.933 | 0.107 | 0.040 | 5.16 | | $3.18 |
| Gemini 3.8-flash H2 | 0.833 | 0.813 | 0.853 | 0.900 | 0.080 | 0.133 | 5.53 | 0.25 | $3.42 |
| Gemini 2.5-flash H0 | 0.733 | 0.627 | 0.840 | 0.907 | 0.107 | 0.080 | 5.81 | | $1.12 |

Median agent's unsafe and wrongful rates cover 133 cases (17 missing in its public log).

## Paired bootstrap over cases (difference, 95% interval)

| comparison | success | unsafe | wrongful |
|---|---|---|---|
| native - o4-mini-high | +0.087 [+0.020, +0.153] | -0.053 [-0.100, -0.013] | -0.060 [-0.113, -0.007] |
| guard - o4-mini-high | +0.113 [+0.047, +0.180] | -0.073 [-0.120, -0.033] | -0.020 [-0.080, +0.040] |
| native - public median | +0.373 [+0.287, +0.460] | -0.188 [-0.256, -0.120] | -0.030 [-0.075, +0.008] |
| native - compiled A | +0.093 [+0.013, +0.173] | +0.027 [0.000, +0.060] | -0.187 [-0.253, -0.127] |
| native - compiled AC | -0.033 [-0.080, +0.007] | +0.020 [0.000, +0.047] | -0.027 [-0.060, 0.000] |
| guard - compiled AC | -0.007 [-0.060, +0.047] | 0.000 [-0.020, +0.020] | +0.013 [-0.020, +0.053] |
| native - Gemini 3.8 H0 | 0.000 [-0.047, +0.047] | -0.020 [-0.053, +0.013] | +0.013 [-0.013, +0.040] |
| native - Gemini 2.5 H0 | +0.080 [+0.013, +0.147] | -0.020 [-0.060, +0.020] | -0.007 [-0.047, +0.033] |
| guard - native | +0.027 [-0.020, +0.073] | -0.020 [-0.053, +0.007] | +0.040 [+0.013, +0.073] |

## Problems found
1. First run invalid (out/native_v1): in 40 of 150 episodes the agent chained shell commands, the allowed-tools prefix rule
   denied them and the agent refused. Fixed with one sentence (one python3 call per Bash command). Then 1 denial in 150.
2. Real date leak: Claude Code puts the real date (Oct 2026) in context, SOPBench cases are dated 2024. About 9 episodes
   per arm used the real date (about 4 failures native, 6 to 7 guard). Sensitivity arms with "get the date only from the
   tools": native 0.787, guard 0.833, within noise.
3. Guard cost on perform cases: compiled programs require a login the user cannot always provide, the guard blocks, the
   agent refuses (bank/pay_bill/0, pay_loan/0, set_safety_box/0). Wrongful refusals 0.067 to 0.147.
4. Run-to-run swing about +-0.03 on identical settings. Differences under 0.05 are not readable.

## Verdict
- A current agent in a modern harness beats the best public agent on these 150 cases: 0.813 vs 0.727, unsafe 0.067 vs 0.173.
- Sonnet in Claude Code ties Gemini 3.8-flash in the plain SOPBench harness (0.813 vs 0.813). The compiled program AC (0.847)
  is as good or better.
- The guard adds +0.027 [-0.020, +0.073], not significant, cuts unsafe actions, raises wrongful refusals.

## Limitations
One run per case per arm. Claude Code loop with Bash-based tool calls, not a function-calling API. Prompt differs from
SOPBench only in the bridge text. 150 of 830 cases (50% perform). Scorer reproduces 158 of 160 public records.

## Paths
sop_tool.py, setup_case.py, run.py, score.py, analyze_cc.py, validate_score.py, dump_harness.py, prompt_template*.txt,
out/{native,guard,native_date,guard_date,native_v1}, sessions/<arm>/sNNN/, results.json, analysis_out.txt, harness_cases.json.
