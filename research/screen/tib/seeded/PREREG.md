# Exp 3: probes seeded with past conversations (pre-registration, 3 Nov 2026)

Written before any seeded probe was generated. Approved by Tima on 3 Nov ("давай все 3, но только на 3 можно максимум 15
баксов"): budget 15 USD for this experiment, enforced by the proxy (`../cap_usd.txt` = 34.0 USD on a ledger that stood
at 19.69).

## Question

In the first P0-2 run the probes were written from a family's one-line purpose, and for gemini-3.8-flash they did not
rank the families (rho 0.16) although the real target is reliable (0.80). The failures sat in details of the real
scenarios (customer does not volunteer IDs, a final follow-up request, a "corrected" date window). A company that
automates a procedure has past conversations handled by people. Do probes written with two such past conversations as
examples rank the families for both agents?

## Design

- Same 20 families, same 6 real tasks per family and the same recorded real outcomes (`../runs/real_*.jsonl`).
- Seeds: 2 other benchmark tasks of the same family per family (`seeds.json`, drawn with seed "20261103-seeded" from the
  family's tasks outside the 6 evaluation tasks). They play the role of past conversations: the generator sees their
  customer script and gold actions.
- Generator as before (gemini-3.8-flash, policy, instructions, tool schemas, 3 candidate customers never used by any task
  of the 20 families) plus the two seed scripts, with the instruction to write a new conversation for the chosen customer
  in the same style, not a copy. max_tokens 32,768. Validity check unchanged (gold actions run cleanly). 6 probes per
  family, up to 10 attempts.
- Scoring from the start with the benchmark's per-tool compare_args (`../compare_convention.json`).
- Both agents, one attempt per probe, same harness and user simulator as the first run.

## Outcomes

Primary: Spearman between the seeded-probe failure rate and the real failure rate over the 20 families, per agent,
bootstrap over families. Support: for gemini-3.8-flash rho >= 0.40 with the lower bound above 0. Secondary: paired
difference to the documentation-only probes (v2 scoring), the graded score (share of gold actions missed), the frozen
MF on seeded probes.

## Stated up front

The benchmark's families are templated, so two seed tasks show the generator almost the exact scenario of the
evaluation tasks with other customers. This is the friendliest version of "past conversations" and an upper bound for
the idea; real past tickets vary more. If the budget runs out, the analysis uses what finished and says so.
