# Brief for round 10 (night of 7 to 8 Oct 2026): push toward rho >= 0.6 without leakage

## Research question (unchanged)
From the documentation of a business procedure (policy or SOP text, tool schemas, the customer's request or scenario)
predict whether an LLM agent will handle it, without running agents on that case. Target: item difficulty `b` from a
1PL IRT fit over many agents (repo data/<bench>/irt/1d_1pl/items.csv, HIGHER b = HARDER).

Main metric: pooled within-domain case-level Spearman rho over four new-process scenarios (SOPBench and tau2, new
procedures and new domain), paired bootstrap CI over procedures. Baseline (Agent psychometrics + 18 ADeLe levels) 0.131.
c_lad 0.258. c_lad + GAP (data-gap ledger) 0.311 (+0.053 [+0.007, +0.089] over c_lad, post hoc, tau2 only).
SOPBench procedure level (new domain): baseline 0.231, c_lad 0.486, CMP 0.543, policy-rule surprisal under Qwen (PRI,
not pre-registered) 0.689.

## Ceilings measured tonight (full-data b, within-domain, pooled by pairs; ideas10/ceil.py)
- SOPBench: procedure-mean oracle 0.231; label-only oracle (should_succeed) 0.520; procedure x label cell-mean oracle 0.614.
  86% of within-domain variance of b is within procedure; 47% remains even within procedure x label cells.
- tau2: procedure-mean oracle 0.852 (airline and retail have one case per procedure, telecom 0.677); identical-text oracle
  0.895 (telecom has 5 distinct texts for 114 cases, the rest is hidden device state).
- So pooled case-level 0.6 at L0/L1 requires SOPBench near its procedure x label ceiling, which needs the hidden record.

## Information levels
L0 documentation only; L1 a few cheap LLM calls per case on paper; L2 reading the hidden case record by code (separate
track, never mixed into L0/L1 numbers); E agents run only on synthetic cases generated from documentation.
Leakage = anything derived from outcomes or gold on test cases: agent results, IRT b, should_succeed, constraints,
directed_action_graph, gold actions, evaluation criteria, user_instruction fields written by the generator with the
label in mind, cross-case group structure tricks ("one of k cases is the perform case"). Benchmark-generator artifacts
are not leakage only if a real agent sees the same cue in the request (e.g. a visibly random password).

## Rules
- HANDBOOK is a blind test set: never open prototype/data/external/handbook or anything about it.
- Never print or write API keys or the GCP project id. Do not modify the public repo.
- Heavy CPU work inside `with cpu_lock("<name>")` (prototype/screen/ideas/cpulock.py). One MPS job at a time.
  Never kill processes you did not start.
- Legal sources only, no CAPTCHA bypass, no paywall circumvention. Do not execute code from downloaded repositories.
- Check every citation (arXiv id, venue, year, the number quoted) against the source; mark "unverified" otherwise.
  Never write "we are the first".
- Work only in prototype/screen/ideas10/<your_name>/. Writing files with the Write tool may be refused for subagents:
  use Bash heredocs or python to write files, and ALWAYS return the full report as text in your final message.
