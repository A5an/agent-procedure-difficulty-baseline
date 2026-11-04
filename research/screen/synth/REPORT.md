# Synthetic procedures written by Claude models (3 Nov 2026)

Plan: `PREREG.md` (hash in `FREEZE_GEN.txt`, written before the main generation), inputs of the episodes hashed in
`FREEZE.md` before the first episode, deviations in `DEVIATIONS.md`. Literature: `LIT.md` (all 34 arXiv ids checked
against the arXiv API). Gemini spend of the first round 11.88 USD of the 20 USD budget (agents 8.64, user simulator
2.69, policy judge and audit 0.55); generation ran on the Claude subscription (notional list price 15.85 USD).

## What was built

A factory for customer-service procedures where code, not the LLM, decides what is correct (`engine.py`). A Claude
model writes one procedure per brief as a formal spec (`SPEC_FORMAT.md`): tables, get/search/update/create tools,
request parameters, identity claims, ordered decision rules, outcomes with exact write calls, the policy text, and 8
customer cases. The program checks the spec, computes for every case which records the agent can reach from what the
customer knows (fixed point over the tools, a unique match required), which rule fires, the correct write calls and
the database end state, and refuses cases that do not realize their planned information level, behaviour or class, or
whose expected outcome (the generator's own belief) differs from the computed one. Errors go back to the model, at
most 2 repair rounds. An independent model (gemini-3.8-flash) checks that the policy text states every rule; it
caught 3 of 3 deliberately broken policies in a sanity test (`judge_sanity.out`). Each case becomes a tau2 task with
its own environment; grading is by database end state.

24 briefs (domains, structure, 8 planned cases each: 7 information levels, 4 behaviours, 4 perform and 4 refuse), the
same for Claude Opus 5.5, Sonnet 5.5 and Haiku 5.5. Agents: gemini-2.5-flash-lite (lite), gemini-2.5-flash (flash),
gemini-3.8-flash (g38), user simulator gemini-2.5-flash, one attempt per case. 1,701 episodes, none lost.

## Generators

| | Haiku 5.5 | Sonnet 5.5 | Opus 5.5 |
|---|---|---|---|
| briefs accepted | 23 of 24 | 24 | 24 |
| clean on the first try | 13 | 10 | 19 |
| mean rounds | 1.63 | 1.58 | 1.25 |
| cases kept | 183 of 192 | 192 | 192 |
| minutes per brief / output tokens per brief | 2.3 / 34.7k | 1.5 / 15.0k | 2.2 / 15.6k |
| notional list price, all 24 briefs | 0.51 USD | 5.36 USD | 9.99 USD |
| rules stated exactly in the policy (judge) | 100% | 100% | 100% |
| policy 3-gram overlap between its own procedures (lower = more varied) | 0.037 | 0.058 | 0.028 |
| policy 3-gram overlap with the worked example | 0.070 | 0.089 | 0.033 |
| share of rules that decide at least one case | 0.69 | 0.77 | 0.78 |
| failure, lite | 0.27 [0.21, 0.34] | 0.29 [0.22, 0.35] | 0.32 [0.24, 0.39] |
| failure, flash | 0.11 [0.07, 0.16] | 0.06 [0.03, 0.09] | 0.02 [0.01, 0.04] |
| failure, g38 | 0.09 [0.05, 0.14] | 0.03 [0.01, 0.05] | 0.03 [0.01, 0.05] |
| split-half reliability of procedure difficulty | -0.20 [-1.42, 0.32] | 0.17 [-0.68, 0.64] | 0.57 [0.23, 0.76] |
| audit of 20 random failures: agent / simulator / label | 18 / 2 / 0 | 19 / 1 / 0 | 19 / 1 / 0 |
| all failures of flash and g38: agent / simulator / label | 29 / 7 / 2 | 13 / 3 / 0 | 5 / 4 / 0 |

Brackets: 95% bootstrap over procedures. The audit judge agreed with my own reading on 14 of 15 failures (5 per
generator); the one disagreement was a Haiku case whose "irrelevant" detail ("my sister bought the tickets") touched
the policy's account-holder rule, so a careful agent could refuse.

Same brief, different generators: policy 3-gram overlap 0.05 to 0.09, title-word overlap 0.20 to 0.29. The models do
pick similar concepts (for the hotel brief Sonnet and Opus both wrote points-paid room upgrades, and all three made
the terse customer a consultant typing between meetings).

## Factors (logistic regression per agent, all generators pooled, procedure-clustered bootstrap)

Raw failure rates by information level (lite / flash / g38): upfront 0.33 / 0.07 / 0.06, on_request 0.45 / 0.10 /
0.06, lookup 0.25 / 0.04 / 0.04, distracting 0.29 / 0.11 / 0.07, conflicting 0.34 / 0.07 / 0.02, helpful_extra 0.35 /
0.05 / 0.07, insufficient 0.03 / 0.00 / 0.00. By behaviour: changes_mind 0.42 / 0.10 / 0.10, cooperative 0.21 / 0.05 /
0.02, pushy 0.29 / 0.06 / 0.04, terse 0.26 / 0.04 / 0.04.

| effect (log-odds) | lite | flash | g38 |
|---|---|---|---|
| on_request vs upfront | **+0.59 [0.02, 1.19]** | +0.42 [-0.43, 1.19] | +0.11 [-0.60, 0.76] |
| lookup | -0.38 [-1.02, 0.22] | -0.42 [-1.18, 0.31] | -0.22 [-1.03, 0.49] |
| distracting | -0.13 [-0.71, 0.42] | +0.45 [-0.33, 1.21] | +0.13 [-0.75, 0.99] |
| conflicting | +0.08 [-0.47, 0.60] | +0.03 [-0.78, 0.78] | -0.82 [-1.71, 0.28] |
| helpful_extra | +0.15 [-0.40, 0.73] | -0.27 [-1.22, 0.56] | +0.33 [-0.51, 1.11] |
| insufficient | **-1.97 [-2.61, -1.41]** | **-1.39 [-1.78, -1.00]** | **-1.40 [-1.84, -0.88]** |
| changes_mind vs cooperative | **+0.84 [0.35, 1.36]** | +0.65 [-0.08, 1.32] | **+1.33 [0.52, 2.08]** |
| pushy | +0.35 [-0.19, 0.87] | +0.21 [-0.66, 0.94] | +0.24 [-0.64, 0.95] |
| escalation branch | **+0.47 [0.04, 0.91]** | +0.34 [-0.30, 1.00] | +0.24 [-0.59, 0.98] |
| data by Opus vs Haiku | +0.22 [-0.29, 0.74] | **-1.41 [-2.14, -0.75]** | **-1.03 [-1.76, -0.26]** |

(also in the model: terse, class, number of rules and writes, verification, Sonnet; L2 penalty C = 1.)

Hypotheses: H1 (lookup, distracting, conflicting harder than upfront) not supported for any agent. H2 (insufficient
harder) reversed: not acting when the records cannot be found is the easiest case for every agent. H3 (abundant
helpful details do not hurt) consistent with the data, but there is no sign that they help. H4 supported for
changes_mind (lite, g38), not for pushy. H5 (weak agents hurt more by distractors and conflicts) not supported. H6
partly: Opus is clean on the first try most often (19 vs 13 vs 10) and its procedure difficulty is the most
reliable; Haiku has the most simulator and label problems among the strong agents' failures, but Sonnet, not Haiku,
needed repairs most often.

Agreement between generators on which levels are hard is low (Spearman of the 7 level rates between generators 0.00 to
0.83 depending on agent and pair), as expected with this few failures for the strong agents.

Exploratory, monitor: a commit-point monitor (13 features of `../monitor`) separates failures inside the synthetic
data (AUC 0.62 to 0.72, 5-fold by procedure) and does not transfer to the 240 real TauIndianBank episodes (0.33 to
0.46), the same pattern as the earlier cross-benchmark result.

## Reading

The factory works: every accepted spec is executable, the policy states its rules, and almost all failures are real
agent errors (no label error in 60 random failures, 2 in the 63 failures of the strong agents). The problem is
difficulty: the procedures are easy for 2026 agents. flash and g38 fail 2 to 11% (1 to 10% after removing simulator
and label causes), against 39 to 44% on the real TauIndianBank tasks. The information-sufficiency knobs move only the
weak agent, and only "the customer gives the key fact only when asked". What does move every agent is the
conversation: a customer who changes their mind (lite and flash often act before confirming and cannot undo; g38
fails these 10% of the time against 2% for cooperative customers). The other frequent failure kinds were date
arithmetic and malformed tool calls (lite), claiming a lookup is impossible when a tool exists, and acting on the
customer's statement instead of the database.

On the generators: Opus writes the most varied procedures, follows the format best and gives the most consistent
procedure difficulty, at the highest notional price. Haiku is about 20 times cheaper at list prices and its specs pass
the same checks, but its procedures are noisier: more failures of the strong agents that come from the simulated
customer or the spec, and procedure difficulty that does not replicate between halves of the cases. Sonnet is the
fastest and copies the worked example the most.

## Limits

One run per case, one user simulator model, Gemini agents only (so self-preference of a Claude-written benchmark for
Claude agents is not tested), 24 procedures per generator, and the audit is one model's judgement checked on 15 cases
by me. The policy judge passed every main spec, so it adds no contrast between generators.

## Hard round (PREREG_V2.md, hashes in FREEZE_V2.md)

12 new briefs with 6 to 9 rules, 2 or 3 writes, at least 2 rules on records the customer never mentions, a computed
argument, a counting rule in half; no `insufficient` cases; a new behaviour `negotiates` (asks for something refused,
then for something else). Same three generators; all 36 briefs accepted, 288 cases; flash and g38 on all of them
(lite not run). Spend of the round 7.51 USD (total 19.39 of 20).

| | Haiku 5.5 | Sonnet 5.5 | Opus 5.5 |
|---|---|---|---|
| clean on the first try | 3 of 12 | 6 | 3 |
| mean rounds / minutes per brief | 2.00 / 4.1 | 1.67 / 2.2 | 1.92 / 4.0 |
| rules exact in the policy (judge), first / after fix | 100% / 100% | 100% / 100% | 99% / 100% |
| failure, flash (round 1 in brackets) | 0.19 (0.11) | 0.08 (0.06) | 0.10 (0.02) |
| failure, g38 (round 1) | 0.07 (0.09) | 0.04 (0.03) | 0.07 (0.03) |
| g38 failures caused by the agent, per the audit | 1 of 96 | 2 of 96 | 3 of 96 |

HV1 (g38 fails more in the hard round) not supported: differences -0.020 [-0.081, +0.044] (Haiku), +0.016 [-0.021,
+0.057] (Sonnet), +0.047 [-0.005, +0.109] (Opus). For flash the Opus data got harder (+0.083 [+0.010, +0.172]), the
others not significantly. HV2: for g38, negotiates and changes_mind fail 6 points more often than cooperative and terse
(+0.062 [0.000, +0.125]), borderline; for flash +0.035 [-0.056, +0.125]. In the factor model changes_mind is again the
one behaviour that raises failure for both agents (+0.81 [0.10, 1.41] flash, +0.94 [0.14, 1.58] g38). HV3 not
supported (Sonnet was clean on the first try most often). HV4 not supported: with this few failures procedure difficulty
for g38 does not replicate between halves of the cases in either round.

The audit of all 54 strong-agent failures changes the picture for g38: 11 of its 18 failures come from the simulated
customer, 1 from the label, 6 from the agent. The customer either asked for something else after a refusal (11 cases
over both agents, although `user_guidelines_v2.md` told it not to) or wrote its confirmation and `###STOP###` in the
same message, which ends the episode before the agent can act (7 cases, none in round 1). Counting only agent-caused
failures, g38 fails 2.8% of round-1 cases and 2.1% of hard-round cases; flash 5.5% and 10.1%.

## Overall reading

1. The factory gives clean, executable, varied data: across both rounds 3 label errors in 117 audited failures of the
   strong agents and none in 60 random failures.
2. It does not give data that is hard for a 2026 agent. Making the procedure itself bigger (more rules, hidden checks,
   arithmetic, counting) moved flash a little and g38 not at all; g38 fails about 2 to 3% of cases for its own
   reasons, against 44% on the real TauIndianBank tasks. The information knobs that Tima asked about barely matter for
   strong agents: not enough data is the easiest case (the agent does not act), abundant helpful data neither helps nor
   hurts, distractors and conflicting beliefs do not raise failure. What does: a customer who changes their mind.
3. The weakest link for measuring strong agents is the simulated customer (gemini-2.5-flash, thinking off): about a
   third of the strong agents' failures are its errors. A stronger simulator and a guard against a stop token in the
   same message as a confirmation are needed before any further round.
4. Together with P0-2 this points at where real difficulty lives: in what the policy document does not say (benchmark
   conventions, task scripts) and in long multi-step scripted conversations, not in the structure of a fully documented
   procedure. Our generator, by design and by the judge, documents every rule.

Generators across both rounds: Opus writes the most varied procedures and copies the worked example least; on simple
briefs it is clean on the first try most often, on hard briefs it needs as many repairs as Haiku. Haiku is far cheaper
at list prices and its specs pass the same checks, but its datasets are noisier (more simulator and label problems,
most strong-agent failures in round 1). Sonnet is the fastest and the most consistent at passing the checks on hard
briefs.
