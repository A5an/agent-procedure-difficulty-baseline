# Brief for round 3 (5 Oct 2026): predict agent success cheaply, without running agents

Read this file, then research/screen/ideas2/CONTEXT.md (project, data, rules) and
research/screen/ideas/IDEAS_BRIEF.md (shared harness, metrics_extra, cpulock,
how to run Python and the LLM). Everything in those files applies unless this file says otherwise.

## The research question, as the user insists
From the documentation of a business procedure (policy text, customer request, tool list) predict whether an LLM or an LLM
agent will handle it, WITHOUT spending money on expensive simulations (no agent episodes, no pilot runs, no emulated
environments with agent loops). Cheap LLM calls on the documentation (one or a few per procedure or per case) are allowed.

Every method must state its information level:
- L0: documentation only (policy text, customer message, tool list). One-off LLM calls per procedure or per case allowed.
- L1: L0 plus one cheap LLM call per case that writes a plan or answer on paper, without tools and without an environment.
- L2: L0 plus read-only access to the case record through code (a compiled rule program reads the database before any
  agent runs). Cheap in deployment, but it is NOT documentation-only and the protocol forbids it as a feature, so it is
  always reported as a separate track and never mixed into L0 numbers.
- Reference only: agent pilot runs (ideas round 2, pooled rho 0.45 with a gold grader, 0.27 with a real judge).

## What we learned today that should shape your design
1. On SOPBench most within-domain case difficulty is the hidden perform/refuse state (rho of a label-only oracle 0.52).
   Procedure-level documentation features cap case-level rho near 0.2. At the procedure level the baseline is 0.231 and
   policy-reading methods reach 0.41 to 0.52. tau2 has no such cap.
2. Agent failures on SOPBench are mostly skipped checks: 77% of must-perform failures skip or delay a required check,
   78% of must-refuse failures act despite a failed condition (60% never made the check). Step reliability by type:
   authentication 88%, datetime and availability 42%, relation 46% (prototype/screen/ideas2/steps/, files
   step_outcomes.csv.gz, steps.csv, step_type_mapping.csv, reliability_table.csv, case_outcomes.csv.gz, REPORT.md).
   Step types alone did not transfer to new procedures (no better than step count).
3. An LLM compiled each SOPBench policy text into a Python program (prototype/screen/ideas2/compile/, programs_t0.json,
   results_A.json with per-case "unit" and the tools the program called, common.py, execu.py which wires the official
   evaluator). Program failure rate per procedure correlates with agent difficulty (rho 0.49). Its perform/refuse decision
   used as an L2 feature raised SOPBench case-level rho from 0.12 to 0.24 to 0.27 (precheck.py, precheck_ci.json there).
4. Strong and weak agents disagree on what is hard: case-level rho 0.41 between the top 5 and bottom 5 public agents,
   0.16 at the procedure level, against 0.60 and 0.63 between random groups of 5 (prototype/screen/ideas2/rankstab/).
5. Fresh outcomes of modern agents on 150 SOPBench cases (prototype/screen/ideas2/harness/sample.csv): Gemini 2.5-flash and
   3.8-flash in three harnesses (harness/episodes, results.json) and Claude Sonnet 5.5 in Claude Code
   (ideas2/cc_agent/sessions, results.json). Modern agents reach 0.81 to 0.85 and still fail mostly by skipping checks.

## Evaluation
- Use the shared harness and the protocol metric (pooled within-domain rho over the four new-process scenarios) whenever
  your method produces case-level predictions, plus metrics_extra (procedure-level rho, rho_given_label). Paired CI against
  the baseline adele. If your method only covers one benchmark, report that benchmark's two scenarios and say so.
- Fix your 2 to 4 variants before the first full run. Do not tune on test folds. Gold actions, evaluation criteria,
  action_should_succeed, constraints and directed_action_graph are never features (rule-tree counts only if labelled).
- LLM: gemini-3.8-flash via Vertex (src/llm.py of the repo, see IDEAS_BRIEF) unless your task names gemini-2.5-flash-lite.
  For flash-lite copy the client and change the model name. Max 8 parallel calls, cache every response. Stay inside your
  call budget. Another round-2 agent (emulate_run) may still be calling Vertex, so keep parallelism at 8 or below.

## Output
Your folder: research/screen/ideas3/<your_name>/. Keep code, caches and result
JSON there. Writing REPORT.md may be refused for subagents, so return the FULL report as text in your final message:
idea in two sentences and its information level, setup and variants, LLM calls, main table (pooled rho, CI, four
scenarios, paired CI vs baseline, procedure-level rho, rho_given_label), diagnostics, honest 3-line verdict, limitations, paths.
