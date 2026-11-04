# Literature check: synthetic SOP / customer-service agent benchmarks

Date of search: 2026-11-03. Scope: shallow check (about 25 arXiv/web searches, about 30 abstract fetches).

## How to read this file

- "Abstract only" means I read the arXiv abstract page text (via the arXiv API record of the same page) and did not read the body. Nothing below comes from a paper body.
- Venue was checked in two ways: the arXiv "comments" field, and Crossref lookup by title. "Preprint" means neither showed a venue. A paper listed as preprint here may have a venue I did not find.
- Where a number appears, it is quoted from the abstract (or the BFCL blog page where stated).
- Peer-reviewed = venue found. Preprint = no venue found.

Search sources: arXiv API queries (title and abstract keyword queries), WebSearch for discovery (self-bias in generated benchmarks, BFCL, clarification benchmarks), Crossref title lookup for venues, PMLR page and Berkeley BFCL v3 blog page for BFCL.

---

## Q1. Who generates synthetic tool-use / customer-service benchmarks or environments with controllable difficulty?

### Closest to our setting (policy + tools + DB + tasks)

**Levi and Kadar 2025. IntellAgent: A Multi-Agent Framework for Evaluating Conversational AI Systems.** Preprint. https://arxiv.org/abs/2501.11067 (abstract only)
Automates creation of synthetic benchmarks for policy-bound conversational agents. It builds a graph model of policies (relationships, likelihoods, complexities of policy interactions), generates events, and runs user-agent simulations, with scenarios at varying complexity levels.
differs from us: difficulty comes from combining policies, not from controlling information sufficiency or customer behaviour; the abstract does not mention gold labels computed by code from a formal spec, and does not compare generator models.

**Li et al. 2025. SOPBench: Evaluating Language Agents at Following Standard Operating Procedures and Constraints.** EMNLP 2026 main (Crossref DOI 10.18653/2026.emnlp-main.936; arXiv comment shows only a code link). https://arxiv.org/abs/2503.08669 (abstract only)
Seven customer-service domains, 167 tools, each service SOP written as code and turned into a directed graph of executable functions. The original code is the oracle verifier. Over 900 verified test cases, 18 models evaluated, pass rates of 30% to 50% for most strong models.
differs from us: the SOP code is human or semi-human written per domain, not generated from a brief by different generator models; the abstract reports no controlled information-sufficiency factors and no user-behaviour factors; no generator comparison.

**Yao et al. 2024. tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.** Preprint (no venue found on arXiv page or Crossref). https://arxiv.org/abs/2406.12045 (abstract only)
LLM-simulated user, domain tools, policy text, grading by comparing end database state to an annotated goal state, and the pass^k reliability metric.
differs from us: tasks are hand-built for two domains; no controlled factors; we reuse the grading idea.

**Barres et al. 2025. tau2-bench: Evaluating Conversational Agents in a Dual-Control Environment.** Preprint (no venue found). https://arxiv.org/abs/2506.07982 (abstract only)
Adds a telecom dual-control domain and a compositional task generator that builds verifiable tasks from atomic components with controlled complexity. Also couples the user simulator to environment state and separates reasoning errors from communication/coordination errors.
differs from us: the generator composes atomic hand-written components in one domain; complexity is about compositional task steps, not information sufficiency; the procedure itself is not LLM-authored.

**Prabhakar et al. 2025. APIGen-MT: Agentic Pipeline for Multi-Turn Data Generation via Simulated Agent-Human Interplay.** NeurIPS 2025 (Crossref: Advances in Neural Information Processing Systems 38). https://arxiv.org/abs/2504.03601 (abstract only)
Two phases: LLM-committee-reviewed task blueprints with ground-truth actions, then simulated human-agent interplay to make trajectories. Trains the xLAM-2-fc-r models, which beat GPT-4o and Claude 3.5 on tau-bench and BFCL per the abstract. Releases 5K trajectories.
differs from us: purpose is training data for agents, built on existing tau-bench-style domains; no factor-controlled variants and no comparison of generator models.

**Liu et al. 2024. ToolACE: Winning the Points of LLM Function Calling.** Preprint (no venue found). https://arxiv.org/abs/2409.00920 (abstract only)
Self-evolution synthesis of a 26,507-API pool, multi-agent dialogue generation, dual-layer verification (rules plus model).
differs from us: function-calling training data, no policy/DB end-state environment, no controlled information factors.

**Lu et al. 2024. ToolSandbox (Apple): A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities.** Findings of NAACL 2025 (Crossref DOI 10.18653/v1/2025.findings-naacl.65). https://arxiv.org/abs/2408.04682 (abstract only)
Stateful tools, implicit state dependencies, a user simulator, milestone-based evaluation. Names categories such as State Dependency, Canonicalization and Insufficient Information as hard for top models.
differs from us: hand-built scenarios; "Insufficient Information" is one category label, not a graded factor with a causal estimate.

**Patil et al. 2025. The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of LLMs.** ICML 2025 (PMLR 267). https://proceedings.mlr.press/v267/patil25a.html
The PMLR abstract says BFCL evaluates abstention and reasoning in a stateful multi-step agentic setting. The BFCL v3 multi-turn blog (https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html) lists augmented multi-turn categories: Missing Parameters (200 entries, "recognize when essential information is missing"), Missing Functions (200, no available function can fulfil the request, function supplied in a later turn), Long-Context (200) and Composite (200). The blog is not a peer-reviewed page.
differs from us: categories are fixed splits of one base set, not a factorial design; no policy text, no DB end-state grading of customer-service SOPs, no estimate of the effect of each factor on failure.

### Information-gap and clarification benchmarks

**Qian et al. 2024. Tell Me More! Towards Implicit User Intention Understanding of Language Model Driven Agents (IN3).** ACL 2024 (Crossref DOI 10.18653/v1/2024.acl-long.61). https://arxiv.org/abs/2402.09205 (abstract only)
IN3 benchmark of vague user tasks; trains Mistral-Interact to detect vagueness and ask. 
differs from us: no policy/DB/tool execution grading in the abstract; vagueness is annotated, not manipulated as a controlled factor.

**Wang et al. 2024. Learning to Ask: When LLM Agents Meet Unclear Instruction (NoisyToolBench).** EMNLP 2025 (arXiv comment and Crossref DOI 10.18653/v1/2025.emnlp-main.1104). https://arxiv.org/abs/2409.00557 (abstract only)
Benchmark built from real imperfect instructions; finds LLMs tend to invent missing arguments; proposes Ask-when-Needed and an automated ToolEvaluator.
differs from us: single missing-argument noise type, no customer-behaviour or procedure-structure factors, no DB end-state.

**Ross et al. 2025. When2Call: When (not) to Call Tools.** NAACL 2025 (arXiv comment). https://arxiv.org/abs/2504.18851 (abstract only)
Evaluates when to call a tool, when to ask a follow-up, and when to admit the question cannot be answered with the tools; releases a training set and a preference-optimization recipe.
differs from us: single-turn decision labels, not a multi-turn policy environment with end-state grading.

**Qian et al. 2025. UserBench: An Interactive Gym Environment for User-Centric Agents.** EMNLP 2026 main (Crossref DOI 10.18653/2026.emnlp-main.772). https://arxiv.org/abs/2507.22034 (abstract only)
Simulated users start with underspecified goals and reveal preferences gradually; agents must clarify. Reports that models fully align with all user intents only 20% of the time on average and uncover fewer than 30% of preferences.
differs from us: preference elicitation in travel-like tasks, not policy compliance with DB writes.

**Pu et al. 2026. LHAW: Controllable Underspecification for Long-Horizon Tasks.** Preprint (no venue found). https://arxiv.org/abs/2602.10525 (abstract only)
Closest on controllable information removal. Turns well-specified tasks into underspecified variants by removing information along four dimensions (Goals, Constraints, Inputs, Context) at configurable severity, then validates variants with real agent trials and labels each as outcome-critical, divergent or benign by terminal-state divergence. 285 variants from TheAgentCompany, SWE-Bench Pro, MCP-Atlas.
differs from us: removes information from existing long-horizon tasks (coding/workplace), not customer-service SOPs; no distractor, DB-contradiction or customer-behaviour factors in the abstract; no generator-model comparison.

**Bao et al. 2026. Drift-Bench: Diagnosing Cooperative Breakdowns in LLM Agents under Input Faults via Multi-Turn Interaction.** Preprint. https://arxiv.org/abs/2602.02455 (abstract only)
Taxonomy of input faults (implicit intent, missing parameters, false presuppositions, ambiguity), persona-driven user simulator, executes in grounded environments, links clarification to task success.
differs from us: fault types and personas overlap with our factors, but the procedures are not LLM-authored with code-computed gold, and no generator comparison.

**Chen et al. 2026. AppWorld-UL: Benchmarking Diverse Agent-User Interactions for Tool-Use.** ICML 2026 (arXiv comment). https://arxiv.org/abs/2607.20536 (abstract only)
516 tasks over 9 AppWorld apps, modified to add ambiguities and constraints that force clarification, confirmation, or reporting infeasibility. User simulator has designed knowledge boundaries. Best model (Claude Opus 4.7) gets 48.6%.
differs from us: tasks are edited versions of AppWorld tasks (consumer apps), not SOP/policy procedures; no generator-model comparison.

**Suri et al. 2025. Structured Uncertainty guided Clarification for LLM Agents (SAGE-Agent, ClarifyBench).** Preprint. https://arxiv.org/abs/2511.08798 (abstract only)
Clarification method plus ClarifyBench, a multi-turn tool-calling disambiguation benchmark. 
differs from us: method paper over ambiguous parameters; not a procedure-level synthetic benchmark.

**Cao, Driouich, Thomas 2026. Beyond Task Completion: Revealing Corrupt Success in LLM Agents through Procedure-Aware Evaluation.** Preprint. https://arxiv.org/abs/2603.03116 (abstract only)
Re-grades tau-bench runs for procedural integrity; reports 27% to 78% of reported successes are "corrupt" (violations hidden behind a correct outcome) and flags tau-bench design flaws (task scope gaps, contradictory reward signals, simulator artifacts). Relevant to our gold-label-error comparison.
differs from us: audits an existing benchmark; does not generate one.

**Chakraborty et al. 2026. ToolRACER: A Robust Agentic Conversation Emulation Resource for Agent Training and Evaluation.** Preprint. https://arxiv.org/abs/2610.09163 (abstract only)
Synthetic pipeline with user, assistant and tool emulation models; 55 personas, 5.6K validated conversations, about 66% containing failure-prone scenarios, with adversarial user behaviour injected; trains small models and tests on tau2-bench, BFCLv3 and ACEBench.
differs from us: customer behaviour is varied (closest to our behaviour factor), but tools are emulated by an LLM instead of code-executed, and there is no information-sufficiency factor, no code-computed gold, no generator comparison in the abstract.

### 2025-2026 environment-synthesis papers (RL / training focus)

All are preprints with no venue found, abstract only.

- **Song et al. 2026. EnvScaler: Scaling Tool-Interactive Environments for LLM Agent via Programmatic Synthesis.** https://arxiv.org/abs/2601.05808. SkelBuilder mines topics, models logic and checks quality; ScenGenerator makes scenarios and rule-based validation functions. 191 environments, about 7K scenarios, used for SFT and RL on Qwen3. differs from us: goal is RL training gain; no controlled factors; no generator comparison.
- **Dong et al. 2026. Agent-World.** https://arxiv.org/abs/2604.18292. Discovers databases and tool ecosystems, synthesizes verifiable tasks "with controllable difficulty", trains with multi-environment RL. differs from us: difficulty control is not tied to named information-sufficiency or behaviour factors in the abstract.
- **Tian et al. 2026. ASTRA.** https://arxiv.org/abs/2601.21558. Tool-call-graph trajectories plus code-executable rule-verifiable environments for RL. differs from us: training pipeline, no factor design.
- **Xu et al. 2026. EnvFactory.** https://arxiv.org/abs/2605.18703. 85 verified executable environments, 2,575 trajectories, gains on BFCLv3, MCP-Atlas, tau2-Bench, VitaBench. differs from us: training focus.
- **Gao et al. 2026. EigenData (From Self-Evolving Synthetic Data to Verifiable-Reward RL).** https://arxiv.org/abs/2601.22607. Hierarchical multi-agent engine that makes tool-grounded dialogues with executable per-instance checkers; RL on tau2-bench (73.0% pass^1 Airline, 98.3% Telecom). differs from us: aims at training; no generator comparison.
- Also seen in search, not read: EnvCraft (2609.05576), CuES (2512.01311). Not summarized.

---

## Q2. Which manipulate information sufficiency, and does any measure a causal effect on failure?

What the sources above show, per factor on our list:

| Our factor | Closest work | What it did |
|---|---|---|
| Missing info that must be asked for | BFCL v3 Missing Parameters; NoisyToolBench; IN3; UserBench; Drift-Bench; AppWorld-UL; LHAW | fixed splits or edited tasks; LHAW has configurable severity |
| Info only retrievable via lookup | none found as a controlled factor | in our search we did not find a paper that varies "stated upfront vs asked vs search-tool lookup" for the same case |
| Insufficient, correct action is not to act | When2Call (admit cannot answer); BFCL Missing Functions / irrelevance; ToolSandbox Insufficient Information; AppWorld-UL (infeasible instructions) | label categories, mostly single-turn or small environments |
| Helpful extra details / distracting details and near-duplicate records | not found | |
| User claims that contradict the DB | Drift-Bench lists "false presuppositions" (abstract only); Corrupt-success paper flags user misinformation issues in tau-bench (abstract only) | not framed as a controlled factor against DB records |
| Customer behaviour (terse, pushy, changes mind) | ToolRACER (personas, adversarial behaviour); Drift-Bench (persona-driven simulator); tau2 (reliable simulator) | |

Causal-effect evidence: LHAW is closest. It validates each underspecified variant against the original by running agents and measuring terminal-state divergence, so it estimates the effect of removing information on outcomes, per variant. Drift-Bench reports performance drops under faults and variation by persona and fault type (abstract only). I did not find a paper that fits a model of failure on a crossed design of sufficiency x behaviour x procedure structure. I did not read the bodies, so a factorial analysis inside one of these papers cannot be ruled out.

---

## Q3. Generator models, self-preference, benchmark agreement

**Does a stronger generator give a better synthetic benchmark?** Mixed evidence.

- **Koren, Bar-Haim, Goldsteen 2026. Benchmarking the Benchmarks: Evaluating Benchmarks for Conversational Agents.** Preprint. https://arxiv.org/abs/2608.06329 (abstract only). Closest to our plan. Reference-free LLM-judge framework scoring benchmark consistency, complexity and policy coverage for task-oriented conversational-agent benchmarks; validated by agreement with human annotations, by "evaluating benchmarks generated by LLMs of varying capabilities", and by controlled quality-degrading perturbations. differs from us: it scores benchmark quality with judges rather than measuring agent difficulty/discrimination, gold-label errors, or usefulness as predictor training data; I did not read how the generator comparison turned out.
- **Kim et al. 2024. Evaluating Language Models as Synthetic Data Generators (AgoraBench).** ACL 2025 main (arXiv comment). https://arxiv.org/abs/2412.03679 (abstract only). Six LMs generate 1.26M training instances; 99 student models trained. Finds LMs have distinct strengths (GPT-4o better at new problems, Claude-3.5-Sonnet better at improving existing ones) and that data-generation ability does not necessarily correlate with problem-solving ability. differs from us: training data for students, not evaluation sets; not agents.
- **Gill, Ravichander, Marasovic 2025. What Has Been Lost with Synthetic Evaluation?** Findings of EMNLP 2025 (Crossref DOI 10.18653/v1/2025.findings-emnlp.526). https://arxiv.org/abs/2505.22830 (abstract only). LLM-generated versions of CondaQA and DROP are often valid but less challenging for LLMs than the human-authored originals. differs from us: reading comprehension, one generator family in the abstract, no agents.
- **Yuan et al. 2025. LLM-Powered Benchmark Factory (BenchMaker).** ACL 2026 long (Crossref DOI 10.18653/v1/2026.acl-long.1661). https://arxiv.org/abs/2502.01683 (abstract only). Four-dimension, ten-criterion framework to assess benchmark generators; analyzes weaknesses of direct prompting and adds fixes; reports 0.967 Pearson correlation against MMLU-Pro across 12 LLMs. differs from us: single-turn QA-style tasks, not agent environments.

**Generator bias (benchmark by family X favours family X)?** Yes in two peer-reviewed studies, but not for agents.

- **Yuan et al. 2025. Silencer: From Discovery to Mitigation of Self-Bias in LLM-as-Benchmark-Generator.** NeurIPS 2025 (Crossref: Advances in Neural Information Processing Systems 38). https://arxiv.org/abs/2505.20738 (abstract only). Defines self-bias as inflated performance on self-generated benchmarks, attributed to question domain, language style and wrong labels. Combining heterogeneous generators suppresses it to near zero and raises Pearson correlation with a human-annotated benchmark from 0.655 to 0.833 on average.
- **Xu et al. 2025. When LLMs Benchmark Themselves: Deconstructing Self-Bias in Automated Evaluation.** EMNLP 2026 main (Crossref DOI 10.18653/2026.emnlp-main.629). https://arxiv.org/abs/2509.26600 (abstract only). In machine translation, LLM-generated benchmarks favour their creator; sources are the test set and the evaluator; each model's stylistic habits give homogeneous outputs; "strong enough to cause each model to rank itself first"; also seen on a Chatbot Arena task.
- Related but about judges, not generators: Panickssery et al. 2024, LLM Evaluators Recognize and Favor Their Own Generations, NeurIPS 2024 (Crossref DOI 10.52202/079017-2197), https://arxiv.org/abs/2404.13076 (not re-read here beyond title). Chen et al. 2024, https://arxiv.org/abs/2410.20833, preprint, abstract only: no self-preference found in RAG passage tasks, so the effect is not universal.

differs from us (all of the above): none test generator bias for agent environments where gold labels come from code (so wrong labels, one of Silencer's three sources, may be removed by design) while the user simulator and agents still carry style effects. Our test uses Claude generators vs Gemini agents, so same-family favouring would need a Claude agent in the panel to be visible; this is a design note, not a finding.

**Benchmark agreement.** Perlitz et al. 2024, Do These LLM Benchmarks Agree? Fixing Benchmark Evaluation with BenchBench. Preprint ("Under Review" in arXiv comment). https://arxiv.org/abs/2407.13696 (abstract only). Shows that methodological choices change benchmark-agreement-testing results across 40+ benchmarks and proposes best practices. Relevant to how we should compute rank agreement between our synthetic sets and tau2/SOPBench. differs from us: general LLM benchmarks, not agents.

---

## Q4. Synthetic procedures as training data for predicting agent failure

In our search (arXiv API queries combining predict/failure/agent/benchmark/difficulty terms, plus WebSearch) I found no paper that trains a failure or difficulty predictor on synthetic procedures. Nearest neighbours, all using real benchmark data:

- **Shi et al. 2026. EarlyEval: Cheaper Agent Evaluation via Early Outcome Prediction.** Preprint. https://arxiv.org/abs/2609.02783 (abstract only). Episode-level: LightGBM success/failure classifiers over behavioural, textual and reference-solution features halt a run early; 89% to 97% prediction accuracy on SWE-bench Verified, TerminalBench, Toolathlon. differs from us: predicts during execution on real tasks, not before deployment from the procedure.
- **Al-Haque and Johnson 2026. What Makes Software Issue Resolution Tasks Difficult for Agents?** Preprint. https://arxiv.org/abs/2608.18280 (abstract only). Task difficulty is predictable from static features (AUC = 0.863) on CoderForge-Preview; suggests difficulty-controlled benchmark construction. differs from us: coding tasks, real data, no synthetic generation.
- **Hong et al. 2026. CIRCE Divines LLM Task Complexity from Constraint Structure.** Preprint. https://arxiv.org/abs/2610.41089 (abstract only). Predicts LLM correctness from a constraint graph of the problem before the attempt (AUROC 0.691 to 0.727 on reasoning benchmarks). differs from us: puzzle/reasoning problems, not agent procedures.
- Training-side use of synthetic tool data (APIGen-MT, ToolACE, EnvScaler, EnvFactory, EigenData, ToolRACER) trains agents, not failure predictors.

---

## What looks new in our plan

All statements are of the form "in our search (sources, queries, date 2026-11-03) we did not find X; closest are Y". Sources: arXiv API keyword queries, WebSearch, Crossref, venue pages. This was a shallow check; absence here does not mean absence in the literature.

1. In our search we did not find a benchmark generator for policy-bound customer-service procedures where an LLM writes a formal spec and code computes the gold outcome per case; closest are SOPBench (code-defined SOPs, human/semi-human written), tau2-bench (compositional generator over atomic hand-written components), and EnvScaler (programmatic environment synthesis with rule-based validators, aimed at RL training).
2. In our search we did not find a design that crosses information-sufficiency levels (upfront, only when asked, lookup via search tool, genuinely insufficient, helpful extras, distractors/near-duplicates, claims contradicting the DB) on the same procedure; closest are LHAW (severity-graded information removal, long-horizon tasks), Drift-Bench (input-fault taxonomy), AppWorld-UL and BFCL v3 (missing parameters/functions splits).
3. In our search we did not find the "lookup-only" and "claim contradicts DB" levels treated as controlled factors; closest are BFCL Missing Parameters (ask) and Drift-Bench false presuppositions (abstract only).
4. In our search we did not find estimates of the effect of these factors on agent failure inside a policy+DB end-state environment, adjusted for customer behaviour and procedure structure; closest are LHAW's outcome-based classification of variants and Drift-Bench's reported drops by fault type and persona.
5. In our search we did not find a comparison of several generator models (same briefs) for agent environments on validity, factor fidelity, diversity, difficulty/discrimination, gold-label errors and downstream training use; closest are Benchmarking the Benchmarks (benchmarks generated by LLMs of varying capability, judged by an LLM framework, conversational agents), AgoraBench (generators for training data, not evaluation), Silencer and the EMNLP 2026 self-bias paper (generator self-bias, but QA/translation, not agents).
6. In our search we did not find a test of whether agent benchmarks written by family X favour agents of family X; closest are Silencer and Xu et al. (positive findings for LLM-generated QA and translation benchmarks). Gold labels from code remove the wrong-label source of bias, which would be a different setting.
7. In our search we did not find synthetic procedures used as training data for a pre-deployment failure predictor; closest are EarlyEval (in-run outcome prediction on real tasks), the static-feature difficulty study for issue resolution, and CIRCE (pre-attempt prediction from constraint structure on reasoning problems).

## Caveats

- All findings are from abstracts; I read no paper bodies. Details such as whether IntellAgent or Drift-Bench vary factors we list could differ in the body.
- Venues come from Crossref titles and arXiv comments; Crossref entries for 2026 conferences appear with non-standard DOI prefixes (10.18653/2026...), I did not open the ACL Anthology pages.
- tau-bench, IntellAgent, ToolACE, tau2-bench and the 2026 environment papers show no venue in my checks; they may be published elsewhere.
- Semantic Scholar returned 429, so no citation counts were collected.
