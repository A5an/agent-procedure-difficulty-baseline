# Context for the second ideas round (5 Oct 2026)

Read this whole file before you start. Your own task is in the message that launched you.

## The project in a few lines
Course topic of the supervisor (MBZUAI, profile: stochastic routing and scheduling, RL, combinatorial optimisation,
LLM planning): "AI-Enabled Business Process Automation: Feasibility Assessment and Solution Design". Students build a
framework that selects the most suitable automation approach for a business process (rule-based workflow, RPA, ML,
LLM, LLM agent) by feasibility, risk, data requirements, cost and business value, validate it with prototypes, and
apply it to process documentation of an industry partner "without requiring access to sensitive operational data".
Paper due end of December 2026.

Accepted research gap: predict from the documentation of a new business procedure how reliably LLM agents will
execute it (before any run). Data: public per-task outcomes of many agents on SOPBench (830 cases x 28 agent
configurations, 7 domains, about 70 procedures), tau2-bench (278 tasks x 15 agents x 4 attempts), tau-Knowledge banking
(97 x 12 x 4). Blind test sets for the end: HANDBOOK.md (65 tasks, never open it) and ThinkingBox-Bench.

## Where we are stuck
Baseline: Agent psychometrics (Ge et al. 2026, arXiv 2604.00594) two-stage IRT with the 18 ADeLe demand levels
(Zhou et al. 2026, Nature) as task features, protocol of Krsteski and Meyer (arXiv 2608.05797). Metric: Spearman rank
correlation between predicted and real task difficulty inside a domain, averaged over four held-out settings
(SOPBench and tau2, new procedures and new domain). Baseline 0.131 [0.027, 0.255].

Two screening rounds (about 60 variants: LLM rubrics of step types, dry-run of the case against the policy, pairwise
comparisons with Bradley-Terry, in-context anchors, FMEA, privileged information from trajectories, stacking,
borrowing other benchmarks, fine-tuned encoders, kNN routers, MIRT, LLTM) all land between 0.06 and 0.19. None is
significantly better than the baseline. The minimum detectable difference of the test is about 0.05 (same features)
to 0.18 (different information).

Why: on SOPBench most of the within-domain difficulty of a case is the hidden case state. 35% of cases must be
performed (mean IRT difficulty +1.64), 65% must be refused because some policy condition fails in the account
database (mean -0.62). An oracle knowing only this label reaches rho 0.52 to 0.55. An oracle knowing each procedure's
exact mean difficulty reaches only 0.18 to 0.23. Variance of difficulty: domain 14%, procedure 26%,
procedure x perform/refuse 60%. Documentation alone cannot see the database. In tau2 the task text is a scenario for a
user simulator, the domain policy is shared per domain, telecom has 114 tasks but only 5 distinct texts.

The only large positive result: a pilot run. If one other agent runs the same case first (chosen adaptively, the agent
with success chance near 0.5), its outcome raises rho from 0.13 to 0.45 (five pilots 0.70). With a cheap pilot
(gpt-4.1-mini) and a real LLM judge (Gemini Flash) instead of the gold grader: 0.27, gain +0.134 [0.033, 0.250]. The judge
errs 10% on SOPBench, 33% on tau2 (cannot see the final database state). After one pilot the documentation prior adds
nothing at the case level.

At the procedure level (mean predicted vs mean real difficulty per procedure, SOPBench, about 70 procedures) methods that
read the policy reach 0.41 to 0.52 against 0.23 for the baseline, but intervals are wide.

Known benchmark artefacts: in public SOPBench runs gpt-5, gpt-5-mini and gemini-2.5-pro return mostly empty answers
(89%, 47%, 28% of cases), and an empty answer is scored as a correct refusal. tau2 runs mix two user simulators.

The supervisor's own idea: a router that splits a procedure into steps (look-up, check, irreversible action), estimates
each model's success per step type, and sends each step to the cheapest reliable model or to a person. Our public data
records whole-task success, not step success. A planned pilot E8 (two-phase hand-off between two Gemini models on 150
SOPBench cases, about $29) waits for approval.

## Data and code (read-only unless your task says otherwise)
- Public baseline repo (NEVER modify): <repo>
  data/<bench>/responses.jsonl, tasks.csv (case_id, domain, procedure, procedure_id), statements.jsonl (text a method may
  see), features/*.csv, irt/1d_1pl/items.csv (difficulty b), src/llm.py (Gemini client), EVALUATION_PROTOCOL.md
- Shared harness: research/screen/harness.py and the rules in
  research/screen/SCREEN_BRIEF.md and .../screen/ideas/IDEAS_BRIEF.md
- Results of the last round: research/screen/ideas/collected.json and
  */out/**/extra_metrics.json (includes rho_given_label, procedure-level rho)
- SOPBench release: <ZR_DATA>/SOPBench-d2622008
  data/<domain>_tasks.json (per case: initial_database, constraints, user_known, user_prompt, action_should_succeed,
  directed_action_graph, user_goal), env/ (domain classes with the tools, evaluator.py, task.py),
  output/<domain>/ast_<agent>-dep_full-fmt_structured-tool_full-shuffle_False.json (trajectories, each record has task,
  interactions, evaluations[0] with dirgraph_satisfied, action_called_correctly, constraint_not_violated,
  database_match, success)
- tau2 domains: <ZR_DATA>/tau2-domains

## Rules
1. Work and write only in your own folder research/screen/ideas2/<your_name>/.
2. HANDBOOK.md is blind. Do not open anything under prototype/data/external/handbook.
3. Python: PYTHONHASHSEED=0 OMP_NUM_THREADS=4 nice -n 10 <repo>/.venv/bin/python
   Wrap heavy CPU work in `with cpu_lock("<name>"):` from research/screen/ideas/cpulock.py
   (sys.path.insert(0, ".../prototype/screen/ideas")). Never hold the lock around LLM calls.
4. LLM calls only if your task allows them: repo src/llm.py with
   `LLM_BACKEND=vertex GOOGLE_CLOUD_PROJECT=$(gcloud config get-value project 2>/dev/null)`, model gemini-3.8-flash,
   at most 8 parallel calls, cache every response in a jsonl keyed by prompt hash. Never print or write keys or the project id.
5. Be honest. Report what you could not do. Do not tune on test folds. Fix your variants before the first full run.
6. Stay within your budget of time and calls. If something hangs for 20 minutes, kill it and report.
7. Reports in English, plain words, no em dashes, no semicolons.
