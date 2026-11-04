# Hard round of synthetic procedures (pre-registration, 3 Nov 2026)

Written after the first round (PREREG.md) finished and before any hard-round generation. Same 20 USD Gemini budget and
ledger; about 8 USD left (11.88 spent). Reason for the round: on the first round the strong agents passed almost
everything (gemini-2.5-flash 89 to 98%, gemini-3.8-flash 91 to 97%), so the data cannot train or test a predictor of
their failures. Only the weak agent (flash-lite) failed often (27 to 32%).

## What changes

- `briefs_v2.py` (seed 20261107): 12 new domains. Structure: 6 to 9 rules, 2 or 3 writes in the main outcome, at least
  2 rules that check a record other than the main one (the agent must look it up unprompted), a computed argument in a
  write (amount or date from at least two values), a counting rule (db_count) in half of the briefs, escalation and
  identity verification in half each.
- Cases (8 per brief): no `insufficient` (the agents almost never failed it), the six other levels plus two of the
  harder ones (on_request, lookup, distracting, conflicting); behaviours: `changes_mind` x2, new `negotiates` x2 (asks
  for something the rules refuse, then for something else after the refusal; graded on the final request), `pushy` x2,
  `terse`, `cooperative`.
- User simulator guidelines `user_guidelines_v2.md`: one added rule, do not propose a different request after a
  refusal unless the scenario says so. Reason: in the first round 14 of 63 failures of the strong agents were the
  simulated customer renegotiating after a refusal (`audit_strong.jsonl`).
- Everything else as in the first round: same prompt format plus the new brief lines, same three generators via
  `claude -p`, same code checks (plus the new structure checks and the check that a negotiating customer's first request
  is refused by a rule), same policy judge with one fix round, same harness and grading.

## Runs and stop rule

gemini-3.8-flash on all accepted cases first, then gemini-2.5-flash in a fixed random order of procedures (seed
20261108), until the ledger reaches 20 USD. flash-lite is not run.

## Hypotheses

- HV1: gemini-3.8-flash fails more often on the hard round than on the first round, for each generator (bootstrap over
  procedures, interval of the difference above 0).
- HV2: in the hard round, `negotiates` and `changes_mind` fail more often than `cooperative`/`terse` together.
- HV3: generators keep the first-round order of round-0 validity (Opus first).
- HV4: split-half reliability of procedure difficulty for gemini-3.8-flash is higher in the hard round than in the
  first round (descriptive; with 12 procedures per generator the interval will be wide).

Analyses as in PREREG.md (G1 to G7 where they apply, factor model on the hard round). Deviations go to DEVIATIONS.md.
