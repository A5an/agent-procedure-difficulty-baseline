# E8 and E8-0: step routing pilot on SOPBench (pre-registration, 13 Oct 2026)

Written before any E8 episode was run. The design, arms, sample size and primary contrast come from
`../router_pilot/E8_PLAN.md` (4 Oct), which the supervisor approved on 13 Oct ("план хороший, сделайте первые эксперименты").
Deviations from that plan are listed at the end.

## Harness

The SOPBench harness of `../ideas2/harness` (Vertex client `sop/swarm/vclient.py`, SOPBench release unchanged, evaluator
`evaluator_function_directed_graph` unchanged, re-scoring validated there on 120 public episodes with 0 mismatches).
All arms: native prompt (H0), temperature 0, top_p 0.01, max_tokens 4096 (thinking included), 20 turns, 10 actions, scripted
user, same tools. One attempt per case. Retry on exception with temperature 0.7 (logged).

## Router (design a of the plan: two-phase hand-off)

Model A runs the episode. The first time A emits a tool call whose name equals the case's target action, that message is
discarded (not executed, not put in the history, logged), and the same history goes to model B. From then on B answers every
turn until the episode ends. If A never proposes the target, B never runs and the episode equals A alone.
A's earlier tool calls stay in the history as plain function calls (A's thought signatures are dropped; for a Gemini 3 B the
documented dummy signature is used).

## Cases

`../ideas2/harness/sample.csv`: 150 cases, 75 perform and 75 refuse, 46 procedures. It is the plan's `sample150.csv` with 11
cases from two broken bank procedures and unwinnable perform cases replaced from the same domain and stratum (seed 20261005).

## Arms

Pre-registered in the plan (4 Oct):
1. `flash`: gemini-2.5-flash alone
2. `pro`: gemini-2.5-pro alone
3. `pro>flash`: A = gemini-2.5-pro (checks and refusal decisions), B = gemini-2.5-flash (final action)
4. `flash>pro`: A = gemini-2.5-flash, B = gemini-2.5-pro (reverse, negative control)

Exploratory, added 13 Oct before running (closer to the supervisor's router: cheap model for the steps, strong model for the
gate). Reported separately and never used for the primary verdict:
5. `lite`: gemini-2.5-flash-lite alone
6. `g38`: gemini-3.8-flash alone
7. `lite>g38`: A = gemini-2.5-flash-lite, B = gemini-3.8-flash

E8-0 (diagnostic, plan section 3.5): gemini-2.5-pro alone with max_tokens 512 on the 75 perform cases of the sample
(the plan said 30; all 75 are used because the 4096 run already covers them and the 512 run is cheap). Compared with arm 2 on
the same cases: share of empty assistant turns, finish reasons, thinking tokens, perform success, action rate.

## Outcome and analysis

Outcome per case: SOPBench `success` (repository evaluator). Also `dirgraph_satisfied`, `action_called_correctly`, unsafe action on
refuse cases, wrongful refusal on perform cases, cost from the API usage.

Primary contrast (fixed in the plan): `pro>flash` minus the better of `flash` and `pro` (better = higher raw success on the 150).
Exact McNemar on the paired outcomes and a paired bootstrap over the 46 procedures (10,000 resamples) for the 95% interval.
Headline rates are reported unweighted on the 150 and reweighted to the benchmark mix (288 perform : 542 refuse).

Secondary: `pro>flash` vs `pro` (effect of replacing the gate decision), vs `flash`; `flash>pro` vs both singles; share of
episodes with a hand-off and success inside them; how often B repeats A's discarded call.

Predictions written in the plan (4.4, product rule from the public logs): flash 0.681, pro 0.648, pro>flash 0.693 (+0.012 over
the best single), flash>pro 0.670. A null result for the primary contrast is the predicted result.

Support for the router idea needs all of: primary difference at least +7 points with the interval above zero; the gain on perform
cases without a loss on refuse cases; the gain survives at max_tokens 4096 (all arms use it). Reject: interval inside plus or
minus 5 points with the point estimate below 3. Inconclusive: interval wider than 12 points.

For the exploratory arms: success and cost of `lite>g38` against `g38` and `lite` (cost-success frontier); the same paired tests,
labelled exploratory.

Budget: separate ledger `ledger.jsonl` in this folder, hard cap 80 USD for E8 and E8-0 together.

## Deviations from the 4 Oct plan

- Sample: `sample.csv` of the harness study instead of `sample150.csv` (11 broken or unwinnable cases replaced).
- E8-0 uses all 75 perform cases instead of 30.
- max_tokens 4096 in all arms is fixed in advance (the plan left it to E8-0); the harness study already showed +7 points for
  2.5-flash from 512 to 4096.
- Exploratory arms 5 to 7 added.
