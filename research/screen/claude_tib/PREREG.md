# E1-C: the seeds curve with Claude agents (pre-registration, 12 Nov 2026)

Written before any episode except one smoke test (rl_sqmand_0084, Haiku, kept as a real-task episode). Tima, 12 Nov:
"давай все добьем остальное ... юзай клод модели через подписку ... как субагенты с 0 контекста ... чтоб задачи пробовали
решить в своей среде ... соннет, хайку". Gemini money is spent (70 USD total reached), so the remaining work runs on the
Claude subscription (account checked: the same one as the 11-12 Nov SOPBench runs), at no USD cost.

## Question

Does the E1 result (`../seedk/REPORT.md`) hold for agents of another model family: does the agreement between probe
failure and real failure grow with the number of same-procedure past conversations seen by the generator, and do
another procedure's conversations fail to help?

## Design

- Agents: Claude Sonnet and Claude Haiku, each episode a fresh `claude -p` session with no other context, in its own
  empty working directory, acting only through `bridge.py` (bank tools via tau2's environment and replay; messages to
  the customer). System prompt = the benchmark's agent instruction and policy (as tau2's LLM agent) plus how to use the
  bridge; the 34 tool schemas and the customer's first message in the prompt.
- Customer: Claude Haiku with tau2's user-simulator system prompt and the task scenario, seeing only text messages; the
  chat starts with tau2's default greeting. This replaces gemini-2.5-flash, so Claude results are compared within Claude
  (real tasks and probes under the same simulator), not with the Gemini numbers.
- Tasks: the same 300 real tasks (50 families x 6) and the same probes: k0, k2 (earlier), k5, foreign (E1), k1 (28
  families). Nothing is generated again.
- Scoring: tau2's `evaluate_simulation` (ACTION + DB), as for every earlier run. An episode that times out (30 min) is
  scored on what it did; an episode with no agent message at all is an error and is rerun.
- Order: pilot (5 seedk pilot families, Haiku, real + k0 + k2 + k5 + foreign) to measure time and subscription limits;
  then the full sets in the order real, k2, k5, foreign, k0, k1, Haiku first, then Sonnet. If the subscription limit
  stops the runs, the analysis uses complete conditions only and says what is missing.

## Outcomes

As in `../seedk/PREREG.md`: per Claude agent, Spearman between probe failure rate and real failure rate over the 50
families for each condition, bootstrap over families (2000, seed 20261112), paired differences k5-k0, k2-foreign,
foreign-k0, k5-k2; "6 of 6" real success; Brier. Analysis by `../seedk/analyze_seedk.py` with the file paths switched to
this folder.

## Harness fix before the pilot (12 Nov, same day)

The first pilot attempt (moved to `bad_pilot/`, not analysed) used a `say` command for messages to the customer;
Haiku often ended its turn with plain text instead, and some Bash calls were denied because the agent wrapped the
command in `cd ... &&`. Fixed before any analysed episode: each agent turn now ends with its message to the customer
(the final reply text is sent to the simulated customer and the session is resumed with the customer's answer, which
is tau2's turn protocol), the bridge program sits in the agent's working directory and is called as
`python3 bank.py call ...`, at most 30 agent turns per episode (else MAX_STEPS), and permission denials are counted
per episode (`denied`). Three test episodes of probe_edgbcarddisamb_00-02 ran on the fixed harness and are kept.
