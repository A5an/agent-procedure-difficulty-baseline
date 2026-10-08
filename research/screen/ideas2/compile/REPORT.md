# Compiling SOPBench policies into rule programs (agent `compile`, 5 Oct 2026)

Saved by the main session from the agent's final message and analysis_out.md, extra_out.md, step7_out.md
(the agent could not write files itself). 2,491 gemini-3.8-flash calls, about 12.5 USD.
Code: compile.py, execu.py (executor and evaluator wiring), common.py (prompts), extract.py, analyze.py, extra.py, step7.py.
Results: results_{A,B,AC,BC}.json, procedure_table.csv, unwinnable_perform_cases.json.

## Setup
- Policy text differs by case: 830 cases have 415 distinct texts. One program per text (temperature 0), aggregated over
  the 70 procedures.
- Compile prompt: only the agent's own policy text, its tool list and the action name. No gold fields.
- Scoring: the official evaluator_function_directed_graph wired through the executor. Replaying 1,660 stored agent
  trajectories reproduced their success and dirgraph_satisfied in 1,660 of 1,660 cases.
- Variant A: parameters from the structured user_known dict. Variant B: parameters extracted from the customer message by
  one LLM call per case.
- Condition C (post hoc, optimistic): after seeing the base failures on the same cases, the domain's default example
  database was added to the compile prompt.

## Results (success, same evaluator, 830 cases)

| | Base A | Base B | C+A | C+B | Best agent | Median agent | Always refuse |
|---|---|---|---|---|---|---|---|
| overall | 0.770 | 0.764 | 0.863 | 0.836 | 0.753 | 0.417 | 0.653 |
| perform (288) | 0.431 | 0.406 | 0.691 | 0.615 | 0.559 (o4-mini-high) | 0.299 | 0 |
| refuse (542) | 0.950 | 0.954 | 0.954 | 0.954 | 0.856 (o4-mini-high) | 0.485 | 1 |

Best and median among the 25 agents without the three empty-answer configurations. Best is o4-mini-high (0.753, also
best of 28). Paired bootstrap over cases, compiled minus best agent: A +0.017 [-0.012, +0.047], B +0.011 [-0.019, +0.040],
C+A +0.110 [+0.084, +0.135], C+B +0.083 [+0.059, +0.110]. Compiled minus median agent: A +0.353 [+0.312, +0.392].

By domain (A): bank 0.761 vs best 0.769, dmv 0.918 vs 0.876, healthcare 0.629 vs 0.927, hotel 0.656 vs 0.708, library
0.742 vs 0.606, online_market 0.924 vs 0.895, university 0.810 vs 0.952. With C healthcare rises to 0.976.

Per procedure: A above the 25-agent mean on 61 of 70 (C+A on 63). Mean procedure success compiled 0.726 (A), agents 0.456.
Spearman over procedures of compiled success with agent mean success: 0.49 (25 agents), 0.52 (28), p < 0.001. Part of this
comes from scorer artefacts shared by programs and agents.

Decision accuracy (action called iff must be performed): A 0.804, C+A 0.927. Success among decision-correct cases 0.93 to 0.97.

## Failures
- Base A (191): 147 wrong refusals of performable cases (112 of them because programs guessed field names of tool
  outputs), 22 right decision but required checks not called, 14 wrong performs, 6 tool-response mismatch, 2 crashes.
- C+A (114): 32 scorer artefacts (perform cases nobody can win: target tool missing for cancel_credit_card and
  pay_bill_with_credit_card, or a non-bool action return the evaluator does not count, agents average 0.10 on them),
  43 missing prerequisite call (a login or loyalty look-up that the policy text does not state, agents have the same gap),
  23 wrong decisions (17 wrong refusals, 6 wrong performs), 6 calls to a tool agents do not have (library interaction
  date), 5 tool-response mismatches, 5 crashes on mixed date types.

## Cost
Compile about 0.0036 USD per case amortised, plus 0.00065 per case for variant B extraction. An agent episode is about
0.041 USD per case (E8_PLAN.md token counts at gemini-3.8-flash prices). Compiled is roughly 11 times cheaper.

## Program disagreement (analysis only, uses case databases as inputs)
Two extra samples at temperature 1. The three base programs disagree on whether to call the action in 6.3% of cases.
Per procedure, disagreement vs agent difficulty rho 0.19 (p = 0.11), 0.37 on the 44 procedures with 5 or more cases.
Program failure rate vs agent difficulty rho 0.49.

## Verdict
1. Compiled rules match the best agent without example data and beat it by about 0.11 with it, mostly through correct refusals.
2. Whether a procedure compiles tracks agent difficulty at the procedure level (rho about 0.5), driven by whether the text
   states the steps and the data formats.
3. SOPBench is templated and rule-shaped by construction, so this does not show that real industry documents compile as well.

## Limitations
The compiler model is newer than most agent models. No held-out procedure split for prompt development (C is post hoc).
Single-case procedures make procedure-level numbers noisy. Closest prior work on the same benchmark: Compile, Then Page
(arXiv 2607.11346), which compiles the machine-readable constraints and keeps an LLM executor.
