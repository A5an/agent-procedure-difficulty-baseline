# Literature check for directions D4 to D7 (5 Oct 2026)

Scope: 30 papers, 2023 to October 2026. For every paper I fetched the abstract (arXiv API, Semantic Scholar batch, ACL Anthology or AAAI page). I did not read full texts, so "does not do X" means "the abstract does not say it does X". Venue status is "peer reviewed" only where I saw a venue page or a venue field. Everything else is "preprint".
Verdicts: closes = the work already answers our question, narrows = answers part of it, supports = gives a method or evidence we can build on, unrelated = looks close but asks something else.

## D4. Controlled factorial perturbation of procedure documents, effect on agent success

| Work | Year | Venue, status | What it does | Verdict | Link |
|---|---|---|---|---|---|
| CC-Gen, "Analyzing and Internalizing Complex Policy Documents for LLM Agents" | 2025 | ACL 2026 long paper, peer reviewed | Generator of agent benchmarks with controllable policy complexity on four levels (task, workflow with nested if-else, environment, query), shows success falls as complexity rises | narrows (closest, but levels are bundled, not a factorial design, and the goal is internalisation by training) | arXiv 2510.11588, aclanthology.org/2026.acl-long.767 |
| MANTRA, SMT-validated compliance benchmarks | 2026 | preprint | Synthesises manuals plus machine-checkable compliance checks with a tunable complexity knob, 285 tasks, 50+ page manuals | narrows (a complexity knob exists, abstract does not report per-factor causal effects) | arXiv 2605.06334 |
| ToolLoad, "Beyond Accuracy: Cognitive Load Framework" | 2026 | AAAI 2026, peer reviewed | Parametric control of intrinsic load (tool dependency graph) and extraneous load (ambiguous presentation), maps performance cliffs | narrows (same causal design logic, but on tool graphs and BFCL, not on policy text) | arXiv 2601.20412, ojs.aaai.org/index.php/AAAI/article/view/40650 |
| Policy Loopholes in Agent Evaluation (Cao) | 2026 | preprint | Audits two tau2 domains, taxonomy of policy silence, ambiguity and contradiction, shows affected tasks give unstable scores | supports (observational audit, not a controlled perturbation, but it shows policy text properties change success) | arXiv 2609.14400 |
| PolicyBank | 2026 | preprint | Extends a tool-calling benchmark with controlled policy gaps and lets the agent refine its policy reading from feedback | narrows (controlled manipulation of one factor, gaps, plus a repair method) | arXiv 2604.15505 |
| COVENANT | 2026 | preprint | Compiles natural-language workflow instructions into a control-flow graph executed by a controller, success 50.0 to 83.3 percent on a benchmark | supports (an intervention that rewrites the procedure for agents, no factor decomposition) | arXiv 2607.25400 |
| FlowBench | 2024 | preprint (arXiv) | Same workflow knowledge given as text, code or flowchart, compares agent success across formats | narrows (format factor only, one factor at a time) | arXiv 2406.14884 |
| SOP-Maze | 2025 | Findings of ACL 2026, peer reviewed | 397 real business SOP instances, wide-option vs deep-branch classes, error taxonomy (route blindness, conversational fragility, calculation) | supports (descriptive, no perturbation) | arXiv 2510.08942, aclanthology.org/2026.findings-acl.715 |
| ManyIFEval, "When Instructions Multiply" | 2025 | Findings of EMNLP 2025, peer reviewed | Success vs number of instructions, regression predicts performance on unseen instruction combinations from about 500 samples | supports (count factor and a predictive model, but flat instructions, not procedures) | arXiv 2509.21051 |
| ComplexBench | 2024 | NeurIPS 2024, peer reviewed | Hierarchical taxonomy of constraint types and composition types (and, chain, selection, nested) for complex instructions | supports (composition and nesting as variables, single-turn text, no agents) | arXiv 2407.03978 |

Nobody in the 10 works crosses several document factors (conditions, nesting, exceptions, cross references, distractor text, prose vs table) in a factorial design and fits their causal effects on agent success. CC-Gen and MANTRA are the nearest and both bundle factors.

What remains open for us:
- A full factorial design with each factor varied alone and in pairs on the same procedure, with a fitted effect size per factor and interaction (CC-Gen levels are bundled).
- Prose vs decision table vs distractor text vs cross references as separate factors. FlowBench has format only and COVENANT has an end-to-end rewrite only.
- Reuse of the same factors as features for the predictor of difficulty. This is the bridge to our accepted gap and none of the works does it.

## D5. Synthetic task generation to train a difficulty or success predictor, test on real benchmarks

| Work | Year | Venue, status | What it does | Verdict | Link |
|---|---|---|---|---|---|
| Predicting Task Difficulty Without Rollouts (Krsteski and Meyer) | 2026 | preprint | Ex ante difficulty prediction on 17 agentic benchmarks, leave-one-benchmark-out Spearman 0.225 | supports (our protocol, uses real tasks only, no synthetic training) | arXiv 2608.05797 |
| Agent psychometrics (Ge et al.) | 2026 | preprint | IRT with task features, transfers to unseen benchmarks and unseen agent-scaffold pairs | supports (real data only, no synthetic generator) | arXiv 2604.00594 |
| TASTE | 2026 | preprint | Generates new tau2-style tasks from tool sequences with iterative difficulty evolution, builds tau^c-Bench | unrelated to the predictor question (generates harder tasks, does not train a difficulty model on them and test on real ones), useful as a generator | arXiv 2605.28556 |
| What Has Been Lost with Synthetic Evaluation? | 2025 | preprint (arXiv) | LLM-generated versions of CondaQA and DROP are valid but easier than human-authored ones, rankings differ | supports (warning: synthetic items shift difficulty, so a predictor trained on them can be biased) | arXiv 2505.22830 |

I found no work that generates synthetic procedures, trains a success or difficulty predictor on them and tests it on real agent benchmarks. Search of automatic item generation also gave only education papers (item difficulty of reading tests from LLM features), which I did not list.

What remains open for us:
- The whole direction looks open for agents. It needs a generator that controls the features of D4 and a real-data test, which we have (SOPBench, tau2).
- Main risk from the evidence: synthetic items are easier and not rank-preserving, so the predictor may not transfer. A check of this on one held-out real benchmark would already be a result.
- A generated procedure has no hidden database state, which we found drives 60 percent of SOPBench difficulty. Synthetic data must generate the state too, or it only covers the documentation part.

## D6. Sample-efficient or adaptive evaluation of agents

| Work | Year | Venue, status | What it does | Verdict | Link |
|---|---|---|---|---|---|
| tinyBenchmarks (Polo et al.) | 2024 | ICML 2024, peer reviewed | 100 curated items estimate full-benchmark LLM score, IRT-based estimators | supports (method for ranking and score, not for go/no-go on one new task family) | arXiv 2402.14992 |
| Fluid Language Model Benchmarking | 2025 | COLM 2025 spotlight, peer reviewed | Item response model picks items adaptively per model ability, like computerized adaptive testing | supports | arXiv 2509.11106 |
| Efficient Benchmarking of AI Agents | 2026 | preprint | 8 benchmarks, 33 scaffolds, tasks with historical pass rate 30 to 70 percent keep rank fidelity with 44 to 70 percent fewer tasks | supports (agents, but about ranking agents, difficulty comes from history, not from documentation) | arXiv 2603.23749 |
| PTA-IRT, Efficient SWE Agent Benchmarking | 2026 | preprint | Uses trajectories as privileged information for subset selection and ability estimation under low budgets | supports (close to our pilot-run finding, needs historical trajectories) | arXiv 2609.01603 |
| Efficient Benchmarking in Production | 2026 | preprint | Recurring evaluation of one production agent, multidimensional 2PL adaptive testing reaches 1 pp MAE with 38.5 percent of a run | supports | arXiv 2609.21267 |
| Don't Pass@k: Bayesian framework | 2025 | ICLR 2026, peer reviewed | Dirichlet posterior over success probability with credible intervals, supports informative priors, stable ranking with few trials | narrows (right statistics for few runs, prior is generic, not from task features, single model not a task family) | arXiv 2510.04265 |
| Trust or Escalate (cascaded selective evaluation) | 2024 | ICLR 2025, peer reviewed per S2 venue field | Cheap judge first, escalates to strong judge only when unsure, with a provable human-agreement guarantee | supports (matches our cheap pilot plus judge idea, about judging pairs not agent outcomes) | arXiv 2407.18370 |
| Agent Evaluation Reliability (Bayesian variance decomposition) | 2026 | preprint | Splits leaderboard variance into signal and noise over 22 benchmarks, reliability of rankings depends on the claim | supports (tells how many tasks and scaffolds, not how few runs for a decision) | arXiv 2610.00651 |

Not found in the abstracts: a decision procedure "is this agent good enough for this task family" that starts from a prior built from procedure documentation and stops after few runs. Best-arm identification papers I met (cost-aware best-LLM identification, bandit evaluation) were not fetched and are not counted.

What remains open for us:
- Prior from documentation features plus a Bayesian stopping rule per task family. The Don't Pass@k posterior is the ready statistical piece, the prior is what we would add.
- Our finding that one cheap pilot run adds more than documentation (rho 0.13 to 0.27 with judge) fits PTA-IRT and the 30 to 70 percent filter. The new point would be a pilot chosen from the predicted 0.5 region plus a judge with a known error rate (10 and 33 percent).
- Judge error enters the stopping rule. Trust or Escalate gives guarantees for pairwise judging only.

## D7. Step-level or compositional reliability, failure attribution, routers

| Work | Year | Venue, status | What it does | Verdict | Link |
|---|---|---|---|---|---|
| Mittal, "How Fast Do Agents Rot?" | 2026 | preprint | 9 models, 10,664 trajectories, success follows a geometric law with one per-step reliability, decay driven by step count not context | supports (steps are homogeneous, no step types) | arXiv 2609.01660 |
| Ord, "Is there a half-life for the success rates of AI agents?" | 2025 | preprint | Constant failure rate per human-minute explains success vs task length, each agent has a half-life | supports (homogeneous per-minute rate) | arXiv 2505.05115 |
| TraceToChain, Markov Chain Reliability for LLM Agents | 2026 | preprint | Fits agent traces to an absorbing Markov chain with an automatic cluster taxonomy of steps, credible intervals | narrows (closest to step-type reliability from traces, but it is one agent's chain with unsupervised state clusters and no router) | arXiv 2604.24579 |
| Who&When, failure attribution | 2025 | ICML 2025, peer reviewed | Dataset of failure logs from 127 multi-agent systems, best method finds the failing agent 53.5 percent and the failing step 14.2 percent | supports (step-level blame is hard, labels are for failed runs only) | arXiv 2505.00212 |
| R2-Reasoner | 2025 | ACM proceedings (10.1145/3774904.3793038), appears peer reviewed, arXiv v2 | Decomposes a query into subtasks and a learned allocator sends each to one of 9 models | narrows (routes by learned policy, not by estimated step-type reliability, reasoning benchmarks) | arXiv 2506.05901 |
| BoPO, Budget-Aware Agentic Routing | 2026 | preprint | Per-step choice between a cheap and an expensive model, RL with boundary-relative reward and a difficulty taxonomy | narrows (step routing in agent tasks, reliability is implicit in the policy) | arXiv 2602.21227 |
| Uno-Orchestra | 2026 | preprint | Learns selective decomposition and a (model, primitive) pair per subtask from RL trajectories | narrows (same, implicit) | arXiv 2605.05007 |
| RSI-Router | 2026 | preprint | Mines subtask types from trajectories, evaluates model assignments per subtask type, evolves model-specific skills, about half the cost | narrows strongly (closest to the supervisor router idea, subtask types come from trajectories and routing is evaluated on outcomes, not an explicit reliability table and not on procedures with irreversible actions) | arXiv 2609.34712 |

None of the abstracts of the three seed routers says it estimates per-step-type success probabilities from traces and uses them in a product formula. RSI-Router mines subtask types from trajectories and tests routes empirically, which is the nearest.

What remains open for us:
- Explicit step-type reliability table (look-up, check, irreversible action) times model, with the product law, tested as a predictor of whole-task success. Mittal and Ord assume one rate. TraceToChain has clusters but no model comparison.
- Our public data have only task-level success, so this needs step-level data (pilot E8 or SOPBench trajectories, which do record dirgraph_satisfied and action_called_correctly). Whether those flags give step-type labels is untested here.
- Routing to a person for irreversible steps with a risk term is not in any of the four routers I read.

## Caveats
- Abstract-level reading only, so a method buried in a full text (for example a reliability table inside RSI-Router) could be missed.
- Not fetched and not counted: failure attribution follow-ups (DeFA 2610.01256, StepFinder, ASCon, others), JourneyBench 2601.00596, SOP-Bench 2506.08119, cost-aware best-LLM identification 2609.30360. They exist and could change D7 or D4 verdicts.
- Venue for CC-Gen (ACL 2026 long), SOP-Maze (Findings ACL 2026), Fluid (COLM 2025), Who&When (ICML 2025), tinyBenchmarks (ICML 2024), ManyIFEval (Findings EMNLP 2025), ToolLoad (AAAI 2026) checked on venue pages. Trust or Escalate and R2-Reasoner come from Semantic Scholar and ACM records, not full proceedings pages.
