# Brief for round 8 (6 Oct 2026): a swarm that maps every signal we could pull out of a task description

## The research question
From the documentation of a business procedure (policy or SOP text, tool or API schemas, the customer's request or
scenario) predict whether an LLM or an LLM agent will handle it, WITHOUT running the agent on that case and without
expensive simulations. Cheap LLM calls on the documentation are allowed. Main metric: pooled within-domain Spearman rho
between predicted and IRT case difficulty over four new-process scenarios (SOPBench and tau2, new procedures and new
domain). Baseline (Agent psychometrics + 18 ADeLe demand levels) 0.131. Best confirmed model c_lad 0.258. At the
procedure level documentation methods reach 0.5 to 0.7.

Information levels: L0 documentation only; L1 a few cheap LLM calls per case that write something on paper (no tools,
no environment); L2 reading the hidden case record by code (forbidden by the protocol, separate track); E emulation
(agents run only on synthetic cases generated from documentation). White-box signals from a LOCAL open model (hidden
states, logprobs, attention) count as L0 or L1 depending on whether they read the case.

## What this round is for
The user wants a swarm of agents that list ALL the extra data that can be pulled out of a task description, including
signals that come from how LLMs work inside (the user did not know, for example, that token entropy exists). Each agent
covers one angle and goes from the most obvious ideas to the least obvious. This is an idea-mapping round, not a
protocol run.

Universality is a hard requirement. A good signal is computable for ANY procedure from the three inputs every business
process has: (1) the policy or SOP text, (2) the tool or API schemas, (3) the customer request or scenario, plus
optionally (4) what a cheap or local model does with them. For each idea say how it is computed on SOPBench, on tau2,
on tau-Knowledge banking and on a brand-new company process. Ideas that need a benchmark-specific field (SOPBench rule
trees, tau2 user-simulator instructions only) are allowed but must be marked "not universal".

## Facts that must shape the ideas
1. SOPBench case-level difficulty is mostly a hidden perform/refuse state decided by the customer's database record.
   A label-only oracle reaches rho 0.52. Five attempts to read the label from the message failed (AUC 0.49 to 0.55).
   Procedure-level features cap SOPBench case-level rho near 0.2. A signal helps at the case level only if it VARIES
   BETWEEN CASES of the same procedure. Say for every idea whether it varies by case or only by procedure.
2. tau2 has no such cap: the gold number of actions and writes alone reaches 0.40 to 0.47 (oracle, not allowed).
   Plans written by an LLM reach about 0.29. Telecom has only 3 procedures, never trust telecom-only gains.
3. Agent failures are mostly skipped checks (77% of must-perform failures skip or delay a check, 78% of must-refuse
   failures act despite a failed condition). Step reliability by type: authentication 88%, datetime 42%, relation 46%.
4. Strong and weak agents find different things hard (case rho 0.41 between top 5 and bottom 5 agents).

## Already tried (do not re-propose as new; you may propose a clearly different variant and say how it differs)
Details in prototype/screen/ideas*/<name>/REPORT.md. Text length, embeddings (bge, fine-tuned encoder 0.255 mostly tau2),
18 ADeLe levels and an improved ADeLe (adele_plus), a procedure rubric with a formula ladder (lad2), LLM plan with real
tools (PLAN: calls, writes, reads, rules, facts, refuse, transfer, ambiguity), plan as code with AST features, compiled
policy program length and branches (CMP, CMP+), compile disagreement as ambiguity (failed), written-condition skip model
(conditions), customer-scenario behaviour features on tau2 (SCN: number of requests, pressure, forbidden requests,
conditional instructions, calculations), token-entropy profile of a think-aloud trace (Krsteski and Meyer style, ENT,
hurt c_lad), small-model first-action logprob margin (failed) and policy-rule surprisal under Qwen3-8B-Base (PRI, 0.689 at
the procedure level only), NLI between message and gating clauses (about 0), personas, synthetic-case classifiers for
perform/refuse (failed), facts from the message plus running the compiled rule (failed), tau2 tool-to-ID graph (+0.016),
two planners Gemini plus Sonnet (n.s.), emulation on synthetic cases (E, works at procedure level 0.58 to 0.79),
pairwise comparisons over c_lad (hurt), evolutionary feature search (F1, did not hold on a fresh benchmark).
Councils of 6 Oct already listed: condition packing density, guard distance and order, simultaneous load, interference
among similar conditions, polarity and exceptions, user assertion and action readiness, verification chain depth,
partial evaluation of the compiled program, decision-table metrics, data-flow metrics, Cardoso and Mendling process
metrics, tool-schema complexity, numeric boundary margin, clause retrieval margin, position of gating checks (lost in the
middle), wrong-tool confusion, argument provenance, cross-model disagreement on checks, prior surprisal of rules.
Treat those as known; your value is in what is NOT on this list, or in a concrete, universal, cheap recipe for one of
them that nobody wrote down yet.

## Data you may look at (read only)
Public repo (never modify): <repo>
(data/<bench>/statements.jsonl, tasks.csv, features/, irt/). SOPBench raw: prototype/data/external/SOPBench-d2622008.
tau2 policies and tasks: prototype/data/external/tau2-domains/. tau-Knowledge banking: prototype/data/external/tau2-banking.
Shared harness and how to run Python: prototype/screen/SCREEN_BRIEF.md and prototype/screen/ideas/IDEAS_BRIEF.md.
HANDBOOK is a blind test set: never open prototype/data/external/handbook or anything about it.

## Rules
- Idea mapping first. Use WebSearch and WebFetch for literature. Check every citation you give (arXiv id, venue, year, the
  number you quote) against the source. If you could not check it, write "unverified". Never write "we are the first".
- Optional quick screen: for at most 2 of your ideas that are computable by PURE CODE (no LLM calls, no GPU or MPS, no
  local model inference), you may compute the feature on SOPBench and tau2 cases and report its within-domain Spearman with
  the full-data IRT difficulty (repo data/<bench>/irt/1d_1pl/items.csv) per domain and on average, labelled "exploratory
  screen, not the protocol". Wrap CPU work in `with cpu_lock("<your name>")` from prototype/screen/ideas/cpulock.py.
  At most 20 minutes of CPU. Never kill processes you did not start (no pkill).
- No agent runs, no Vertex or Claude calls, no MPS in this round.
- Work only in prototype/screen/ideas8/<your_name>/. Writing files may be refused for subagents, so return the FULL
  catalog as text in your final message.

## What to return (final message, English, plain, no em dashes)
A catalog of 10 to 15 ideas sorted from most obvious (tier 1) to least obvious (tier 5). For each idea:
- id and short name
- tier 1 to 5
- what it measures and which failure mechanism of the agent it targets
- inputs it reads: policy, tools, request, model (and which model: black-box API, local open model, none)
- recipe: how exactly to compute it, concrete enough that another agent can code it in an hour
- level (L0, L1, L2, E) and cost per procedure or per case
- varies by case or only by procedure
- universality: SOPBench / tau2 / banking / new company process, yes or no each, and why
- status: new, or a variant of an already tried idea (say which and how it differs)
- evidence: a checked citation with the number it reported, or "own reasoning"
- expected effect: case level vs procedure level, rough size, and the risk that it fails
Then: your top 3 with one paragraph each on why, and the results of your optional quick screen if you ran one.
