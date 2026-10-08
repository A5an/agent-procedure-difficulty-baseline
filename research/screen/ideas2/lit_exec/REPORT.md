# Literature check for three directions (lit_exec, 5 Oct 2026)

Method. Every paper in the tables had its abstract fetched by me from the arXiv API, Semantic Scholar, CEUR or the PDF itself. Venue status comes from Semantic Scholar metadata (S2) or the arXiv journal_ref, so "peer reviewed" means S2 lists a venue and I did not check the track (main or workshop). "Preprint" means no venue found. Abstract-level reading only, except where stated: full text grep for ToolGuard (baseline result) and SOPBench and SOP-Bench (rule baseline question). Semantic Scholar rate limited me, so venue strings are as returned and not cross-checked.

Verdict scale. closes = answers our question, narrows = answers part of it or the same question on a different object, supports = shows the premise holds, unrelated = same words, different question.

## D1. Compile a policy or SOP into executable code, compare with the end-to-end agent

| Work | Year | Venue, status | What it does | Verdict | Link |
|---|---|---|---|---|---|
| ToolGuard, Towards Enforcing Company Policy Adherence in Agentic Workflows (Zwerdling et al.) | 2025 | EMNLP per S2, peer reviewed (track not checked) | Compiles policy documents into guard code per tool, checked before each agent action, on tau-bench Airlines. Full text: more than 20 points over the plain agent baseline, guard TPR 0.82 for GPT-4.1 generator | narrows (alongside, not instead of the agent, one domain) | arXiv 2507.16459 |
| Compile, Then Page (executable SOP programs) | 2026 | preprint | Compiles machine-readable SOPBench constraints into pseudo-code run by an LLM with a stack machine. Three arms on six models, Bank 70.4 to 86.4 to 92.8, weak models harmed | narrows (closest on SOPBench itself, but the compiler input is the machine-readable constraints and the executor is still an LLM) | arXiv 2607.11346 |
| COVENANT, natural-language workflow compilation | 2026 | preprint | Parses workflow instructions into an AST and control-flow graph, a controller checks each LLM step. 120 cases from three benchmarks, success 50.0 to 83.3 | narrows | arXiv 2607.25400 |
| Artic, artifact-driven compilation of NL workflows | 2026 | preprint | Compiler for NL workflows, 488 instances in 11 domains, +28 points resolve rate over the text workflow | narrows | arXiv 2608.21341 |
| Compiled AI, deterministic code generation | 2026 | preprint | LLM writes code once, then workflow runs with zero model calls. BFCL n=400 gives 96% completion, 57x fewer tokens at 1000 transactions, plus DocILE invoices | narrows (the only one with a no-LLM-at-runtime arm, but tasks are function calling and invoices, no policy compliance) | arXiv 2604.05150 |
| SOP-Agent (pseudocode SOP as decision graph) | 2025 | preprint | Agent follows a decision graph written from the SOP, beats general agent frameworks on several domains | supports | arXiv 2501.09316 |
| SOPBench (Li et al.) | 2025 | preprint (S2 shows no venue) | 167 tools, 7 domains, SOP code is the oracle verifier. Grep of full text finds no rule-program baseline | supports (our data source, no code-vs-agent comparison) | arXiv 2503.08669 |
| SOP-Bench (Nandi et al., Amazon) | 2025 | KDD 2026 volume per S2, peer reviewed | 2000+ tasks, 12 domains, tests FC and ReAct agents only. Text says rule-based systems need manual formalisation, no baseline run | supports | arXiv 2506.08119 |
| GPT-3 for Decision Logic Modeling (Goossens et al.) | 2023 | RuleML+RR 2023 via CEUR, peer reviewed workshop | 72 experiments, six problem descriptions, GPT-3 produces DMN decision tables from text. Variable extraction good, tables need improvement. No agent comparison | narrows (text to DMN only) | CEUR Vol-3485 paper3896 |
| DMN-Guided Prompting | 2025 | preprint | Decision logic as a DMN model guides the prompt, beats chain of thought in a course case study | supports (DMN as structure for LLM, not as code) | arXiv 2505.11701 |
| ProAgent, Agentic Process Automation (Ye et al.) | 2023 | preprint | Introduces APA, LLM agents build and run workflows, compares with RPA as concept | unrelated to our measurement (defines the paradigm, no policy-to-code comparison) | arXiv 2311.10751 |
| FlowBench | 2024 | EMNLP per S2, peer reviewed | 51 scenarios, workflow knowledge in several formats fed to agents | supports (format of the SOP changes agent accuracy) | arXiv 2406.14884 |

Also read at abstract level and not tabled: WorfBench (ICLR per S2, workflow generation quality, unrelated), WorkflowLLM (preprint, unrelated), Near-Miss 2603.29665 (preprint, uses ToolGuard guards to detect latent failures in trajectories), Beyond IVR / JourneyBench 2601.00596 (EACL per S2, graph-built scenarios for policy agents), Cedar autoformalization 2606.26649 (preprint, security policy as code on MedAgentBench).

What remains open for us in D1.
- Everyone compares compiled-vs-prompt on one benchmark and reports the gain. No work I found predicts, from the document alone, which procedures will gain from compilation or which need an agent. That "when does compiling pay" question is open.
- No run of a pure rule program without an LLM on SOPBench or SOP-Bench exists in what I found. On SOPBench the source code is the oracle, so such a baseline is trivial unless the code is regenerated from the text, which is exactly the ToolGuard step. We could measure the code-from-text success per procedure.
- This direction is crowded in 2026 (five compile papers in four months). Novelty must come from the selection or prediction angle, not from the compile-vs-agent comparison itself.

## D2. LLM-emulated environments from documentation to estimate real agent success

| Work | Year | Venue, status | What it does | Verdict | Link |
|---|---|---|---|---|---|
| ToolEmu (Ruan et al.) | 2023 | ICLR 2024, peer reviewed | LM emulates tool execution, 36 tools, 144 cases. Human check: 68.8% of found failures are valid in the real world | narrows (validates failure validity, not success rate vs real) | arXiv 2309.15817 |
| StableToolBench | 2024 | ACL per S2, peer reviewed | Virtual API server with GPT-4 simulators and a cache replaces live APIs, shows evaluation stability | supports (simulated APIs usable for ranking, not tied to real outcomes) | arXiv 2403.07714 |
| EnvSimBench | 2026 | preprint | Benchmark of how well LLMs simulate environment feedback, 400 samples. Finds a state-change cliff when several states update | supports against us (simulators fail exactly on state updates) | arXiv 2605.07247 |
| Lost in Simulation (LLM-simulated users) | 2026 | ACL per S2, peer reviewed | Human study on tau-bench retail. Success varies up to 9 points across simulator LLMs, miscalibrated: under on hard tasks, over on medium | supports (sim-to-real gap is real and difficulty dependent) | arXiv 2601.17087 |
| UserProxyBench | 2026 | preprint | Scores user-simulator fidelity over tau-bench family. Changing only the user proxy shifts reward by 15.2 points | supports | arXiv 2609.38043 |
| SynAE | 2026 | preprint | Metrics for how well synthetic tool-calling benchmarks match real traces, including Ranking Divergence (Spearman between agent rankings on real and synthetic) | narrows (closest: same sim-vs-real ranking question, but synthetic data derived from real traces, not documentation, and it ranks agents not tasks) | arXiv 2605.22564 |
| OccuBench | 2026 | preprint | LLM Language Environment Simulators built from documents for 100 professional scenarios, 15 models | narrows (document-grounded simulated envs, but no comparison with real environment outcomes in the abstract) | arXiv 2604.10866 |

Also read and not tabled: AutoControl Arena 2603.07427 (code-grounded state plus LLM narrative to cut simulator hallucination, preprint), EnvScaler 2601.05808 (programmatic environment synthesis for training), AgentMercury 2608.20634 (business environments for RL), E-Bench 2607.23722 (fully synthetic enterprise tool benchmark graded by DB diff). All are about training or benchmark construction, not predicting real success.

What remains open for us in D2.
- No paper found validates "success in a documentation-generated environment predicts success in the real or reference environment" at the task level. SynAE is the nearest metric and could be borrowed.
- We have a ground truth other works lack: SOPBench ships real executable environments plus 28 agents, so we can test an LLM-built emulator from the SOP text against it. Risk: EnvSimBench says simulators break on state changes, which is where our difficulty lives (the hidden database state).
- Cost is high (emulating 830 cases times many agents), and the user-simulator literature suggests the gap is largest on hard tasks, the tasks we care about.

## D3. Empirical framework choosing between rules, RPA, ML, LLM, agent, human, validated by running alternatives

| Work | Year | Venue, status | What it does | Verdict | Link |
|---|---|---|---|---|---|
| Are LLM Agents the New RPA? | 2025 | preprint | UiPath RPA vs Anthropic computer-use agent on three tasks (data entry, monitoring, extraction), measures speed, reliability, dev effort. RPA faster and more reliable, agent faster to build | narrows (runs two alternatives on the same processes, but no selection framework and three tasks) | arXiv 2509.04198 |
| Holaj and Alpers, Decision Support System for RPA and IPA in SMEs | 2026 | SCITEPRESS conference (PDF read), peer reviewed | 32 criteria, questionnaire and scoring model. Validated by 8 expert interviews and 2 usability interviews, no runs of alternatives | supports (confirms that selection frameworks are validated by opinion) | scitepress.org/Papers/2026/151273/151273.pdf |
| AgentArch (ServiceNow) | 2025 | preprint | 18 agent configurations (orchestration, ReAct vs function calling, memory) on enterprise tasks, best 35.3% and 70.8% | narrows (compares agent designs, not rules vs ML vs LLM vs human) | arXiv 2509.10769 |
| CentaurBench | 2026 | preprint | Same models as automator vs as assistant to a weaker worker, seven tasks. Rankings only modestly correlated | narrows (automate vs augment choice measured, no rules or RPA) | arXiv 2608.18554 |
| Beyond Generalist LLMs (BPMN to agentic workflow) | 2026 | preprint | Specialist workflow beats generalist coding agents by 9 to 20 points tool-use exactness, over 95% less token cost | supports (deterministic BPMN beats general agent, task is generating the workflow) | arXiv 2607.14456 |
| How Do AI Agents Do Human Work? | 2025 | preprint | Compares 48 human workers with four agent frameworks on 16 tasks | supports (human vs agent on same tasks, skill-level, no selection rule) | arXiv 2510.22780 |

Earlier-search items Wellmann 2020 (DOI 10.1007/978-3-030-58779-6_14) and Farinha 2023 (DOI 10.1177/02683962231165066) were located in Crossref only, abstracts not fetched, so left out of the table. Kaltenpoth et al., BPM 2025 (DOI 10.1007/978-3-032-02867-9_19) integrates LLM agents with process rules and reportedly cuts failures from 16% to 1%. I saw this figure only in a search snippet, the publisher abstract is closed, so treat it as unverified.

What remains open for us in D3.
- I found no 2025 to 2026 framework that (a) scores a process from its documentation and (b) validates the recommendation by running rules, ML, LLM and agent on the same process set. The empirical comparisons that exist are two-way (RPA vs agent) or compare agent designs.
- Our SOPBench and tau2 per-task outcomes give a rare dataset for the agent-vs-human-refusal part, but rules and ML arms would have to be built.
- The hybrid view (rules for the stable core, agent for the edge) is common in practitioner text and in the BPM 2025 paper. A framework that predicts the split per step matches the supervisor's router idea.

## Overall
D1 is the most crowded and mostly closed as "compile beats prompt", so only a prediction or selection angle survives. D2 is the most open, with the key risk that simulators fail on state updates. D3 is open but needs us to build the non-LLM arms. Nothing here uses the blind set.
