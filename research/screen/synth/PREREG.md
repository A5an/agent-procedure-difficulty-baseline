# Synthetic procedures written by Claude models (pre-registration, 3 Nov 2026)

Written after the harness smoke test and a 2-brief generation pilot (pilot outputs in `gens_pilot/`, not used), before
the main generation and before any agent episode on generated data. Budget: 20 USD of Gemini on this folder's ledger
(`ledger.jsonl`, cap in `cap_usd.txt`), approved by Tima on 3 Nov. Generation runs on the Claude subscription (no
Gemini cost). Literature check: `LIT.md`.

## Question

Our benchmarks so far are friendly (SOPBench: the policy states every rule) or templated (TauIndianBank families).
Can we build our own procedures where we control what makes a case hard, and does it matter which Claude model writes
them? Two parts:

1. **Factors.** How much does each controlled property of a case change the failure rate of Gemini agents: how much
   information the customer gives and how (upfront, only when asked, only findable by search, not enough at all,
   abundant helpful details, abundant distractors with near-duplicate records, beliefs that contradict the database),
   customer behaviour (cooperative, terse, pushy, changes mind) and procedure structure (number of rules and writes,
   escalation branch, identity check).
2. **Generators.** Datasets from Claude Opus 5.5, Sonnet 5.5 and Haiku 5.5 built from the same briefs: how do they
   differ in validity, fidelity of the policy text, diversity, difficulty, discrimination between agents, label errors,
   and do they give the same factor effects?

## Design

- **Briefs** (`briefs.py`, seed 20261103): 24 domains, one procedure each. Structure per brief: rules 2 to 6, write
  calls in the main outcome 1 to 3, escalation yes/no, identity verification yes/no (balanced, shuffled). 8 cases per
  brief: all 7 information levels plus one extra non-insufficient level (rotating), 2 cases per behaviour, 4 perform
  and 4 refuse. 192 planned cases per generator.
- **Generation** (`gen.py`): `claude -p` with `--model opus|sonnet|haiku`, no tools, no CLAUDE.md, default effort, same
  prompt (SPEC_FORMAT.md + worked example in another domain + the brief). The program (`engine.py`) checks every spec:
  structure, expressions, the brief's structure, every case realizes its planned level, behaviour and class, the
  generator's own expected outcome equals the outcome computed from its rules, the correct calls execute and change
  the database, no ids in the customer texts. On errors the model gets them back with its JSON, at most 2 repair rounds.
  A procedure is accepted when its procedure-level checks pass and at least 6 of its 8 cases pass; failing cases are
  dropped. 3 workers per model.
- **Policy check** (`judge.py --fix`): gemini-2.5-flash (another family) reads the formal rules and the policy text and
  rates every rule exact / unclear / missing / different, plus contradictions. Specs with any problem go back to their
  generator once to rewrite the policy text only, then are judged again. Procedures still flagged stay in the dataset;
  analyses are repeated without them.
- **Correct outcome by code.** Which records the agent can reach from what the customer knows (fixed point over get /
  search tools, unique match required), which rule fires, the write calls and the database end state are computed from
  the spec. The generator never labels a case.
- **Episodes** (`run.py`): tau2 text harness, agents gemini-2.5-flash-lite ("lite"), gemini-2.5-flash ("flash"),
  gemini-3.8-flash ("g38"), temperature 0, one attempt per case; user simulator gemini-2.5-flash with thinking off and
  domain-neutral guidelines (`user_guidelines.md`); agent instruction `agent_instruction.txt`; grading by database end
  state only. Empty agent replies are re-asked up to 2 times by the proxy (smoke test: flash-lite returned one).
- **Order and stop rule.** After FREEZE.md: lite on all accepted cases, then flash on all, then g38 in a random order of
  procedures interleaved across generators (seed 20261104). Spending stops at 19 USD on this ledger (1 USD reserve for the
  label audit). If g38 does not finish, its analyses use the completed procedures, which are balanced by construction.
  Episodes that crash are re-run once; still crashing counts as missing, reported.

## Measures and analyses

Generators (descriptive, 95% bootstrap intervals over procedures):

- G1 validity: share of briefs accepted, accepted at round 0, mean rounds, share of planned cases kept.
- G2 policy fidelity: share of rules judged exact before and after the fix round.
- G3 diversity: number of entities, tools, bindings, rules used per decision; lexical overlap of policy texts within a
  generator (mean pairwise Jaccard of word 3-grams, lower is more diverse) and overlap with the worked example.
- G4 difficulty: failure rate per agent.
- G5 discrimination: spread of procedure failure rates; split-half reliability of procedure difficulty (odd vs even
  cases within procedure, pooled over agents, Spearman-Brown); whether failure orders lite > flash > g38 and by how much.
- G6 label quality: 20 randomly drawn failed episodes per generator (pooled over agents, seed 20261105) read by
  gemini-2.5-flash with the spec, the correct calls and the conversation; cause = agent error / spec or label error /
  user simulator error / unclear. Share not caused by the agent = label noise. I read 5 per generator myself to check the
  judge.
- G7 agreement: correlation between generators of the per-level failure rates and the sign of each level's effect.

Factors (all generators pooled, then per generator), per agent: failure ~ information level + behaviour + class +
structure, logistic regression with procedure-clustered bootstrap (2000 resamples over procedures); level effects
against `upfront`, behaviour against `cooperative`.

Hypotheses (directional, checked per agent on the pooled data, support = 95% interval excludes 0):

- H1: `lookup`, `distracting` and `conflicting` cases fail more often than `upfront` cases.
- H2: `insufficient` cases (correct action: do not act) fail more often than `upfront` refuse cases.
- H3: `helpful_extra` does not fail more often than `upfront` (Tima's question whether abundant useful information
  helps; also reported in the other direction).
- H4: `pushy` and `changes_mind` fail more often than `cooperative`.
- H5: weaker agents are hurt more: the lite minus g38 gap is larger on `distracting` and `conflicting` than on `upfront`.
- H6 (generators): Opus and Sonnet accept more briefs at round 0 than Haiku; Haiku's dataset has lower label quality
  (G6) and lower split-half reliability (G5).

Exploratory: a failure monitor (features of `../monitor/build.py`) trained on synthetic episodes of each generator and
tested on our 240 real TauIndianBank episodes (AUC), against the 0.44 to 0.54 of the cross-benchmark transfer in
`../monitor/REPORT.md`; a mixed Opus+Sonnet dataset (12 procedures each) against each single generator on G5.

## Not in this round

Probes written from the policy texts, and variants of the policy with a rule left out (the "the document is
incomplete" curve). Both come after we see whether the datasets are valid and discriminative.

## Deviations

Logged in REPORT.md with the reason.
