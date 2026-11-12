# E1: how many past conversations the probe generator needs (pre-registration, 12 Nov 2026)

Written before any call of this experiment. Plan from the handoff of 11 Nov (Asanali, `docs/handoff-2026-11-11-temirlan.md`
in A5an/agent-router-research-context), E1. Approved by Tima on 12 Nov: E1 and E2 only, pilots first, 20 USD for both
pilots together on this folder's ledger (`ledger.jsonl`, cap in `cap_usd.txt`, proxy `proxy.py` on port 8095). The full
run needs a separate approval after the pilot.

## Question

Seeded probes (the generator sees 2 past conversations of the procedure) ranked the 50 TauIndianBank families with rho
about 0.70, documentation probes (0 conversations) did not (0.16 for gemini-3.8-flash). How does the agreement between
probe failure rate and real failure rate grow with the number k of past conversations, and how much of the k=2 gain comes
from knowing this procedure's scenario rather than the style of the bank's customers?

## Design

- Families and real outcomes: the 50 families of P0-2 (`../tib`) and of the replication (`../combo_rep`), their 6
  evaluation tasks each, and the recorded real episodes of both agents. No real task is run again.
- Conditions (`select_seedk.py` -> `selection_seedk.json`, fixed seed strings, nested):
  - k=0: existing documentation probes (`../tib/probes_v2.json` scored v2, `../combo_rep/probes_doc.json`);
  - k=1 (new): the first of the two existing seeds;
  - k=2: existing seeded probes (`../tib/seeded/probes.json`, `../combo_rep/probes_seeded.json`);
  - k=5 (new): the two existing seeds plus three tasks of the family outside its evaluation tasks and outside the seeds,
    drawn with seed "20261112-k5-<family>". hpgold has only two such tasks, so it gets k=4 (recorded);
  - foreign (new, control): the two existing seeds of another family, drawn with seed "20261112-foreign-<family>" among
    the families of the same class (write expected or not) whose set of gold tool names has Jaccard < 1/3 with this
    family's. The generator sees how this bank's customers talk and the format of a correct trace, not this scenario.
- Generator, prompt, candidate filter, gold-action validity check and per-tool compare_args convention as in
  `../tib/seeded/gen_seeded.py` (gemini-3.8-flash, temperature 0.7, max_tokens 32768, 6 valid probes per family, up to 10
  attempts, candidate customers exclude every customer of the family set's tasks as before). Only the examples block
  changes: the count words, and for foreign the header says the examples are of another procedure and the probe must be
  for the procedure described. For k=2 the template reproduces the original text exactly (checked).
- Episodes: both agents (gemini-3.8-flash, gemini-2.5-flash), one attempt per probe, user simulator gemini-2.5-flash,
  harness and settings of `../combo_rep/run_rep.py` (`run_seedk.py`).

## Pilot

5 families drawn with seed "20261112-pilot": hpemi, edgbcarddisamb, sqmand, hpgold, edgbfdcap. All three new conditions,
both agents (90 probes, 180 episodes at most). Purpose: check the pipeline (valid-probe rate, attempts, errors) and the
real price per condition. No rho is computed on the pilot. The pilot families stay in the full run with the same
probes; a change of the design after the pilot (only for a bug) is written in the report.

## Outcomes (full run)

Primary: per agent and condition, Spearman between the probe failure rate and the real failure rate over the 50
families; bootstrap over families (2000 resamples, seed 20261112); paired differences on the same resamples.

- H1 (curve): rho(k=5) - rho(k=0) has a lower bound above 0 for both agents. Monotonicity of k = 0, 1, 2, 5 is reported
  descriptively.
- H2 (leak control): if rho(k=2) - rho(foreign) has a lower bound above 0, the seeded gain needs this scenario (the
  templated families make k=2 an upper bound, and the report says so). If rho(foreign) - rho(k=0) has a lower bound
  above 0 and rho(k=2) - rho(foreign) includes 0, the style of past conversations is enough.

Secondary: "probes passed out of 6 -> real success" per condition; `ready_proc.py` (A5an branch
research/2026-11-11-diagnostic-memory) for each k at targets 0.85 and 0.90; Brier; valid-probe share, generator
attempts and USD per family (the price of generation is a result).

## Stated up front

The families are templated, so same-family seeds show nearly the evaluation scenario with other customers; k=5 shows it
five times. Probes are run once per agent, so per-family probe rates move in steps of 1/6. If the cap stops the full
run, the analysis uses what finished and says so.
