# Swarm agent `lit`, round 10 (8 Oct 2026)

Saved by the main session from the agent's final message.

The literature has little evidence that text-only methods reach rho 0.6 on LLM-agent difficulty, and every result I found at or above 0.6 comes from a setting we lack. I created `research/screen/ideas10/lit/` but wrote no files, so this message is the whole report. No agent runs or installs were done.

Verification tags: **V** means I fetched the paper page or full text and checked the quoted figure there. **U** means unverified: search snippet, secondary summary or memory only. I never claim priority for anything.

## What the evidence says first

- **Agent difficulty from text is hard in the literature.** Task-text prediction of IRT difficulty for agents reaches 0.40 in-distribution and 0.225 leave-one-benchmark-out (Krsteski and Meyer, 2608.05797). On the within-benchmark z-scored target, ridge on bge embeddings gets 0.26 to 0.30 (Messier, 2607.25891). Our case-level 0.31 is already at or above that range.
- **The 0.6+ results all have a feature we lack.** They are in education, with human difficulty, an answer key, and many simulated or real respondents. Examples are Acquaye et al. (0.75 to 0.82) and Hoyl (Pearson 0.78). The one tiny-sample exception is below.
- **The missing piece is a leak-free key.** Simulated respondents must be scored right or wrong. For us a gold action or `should_succeed` is leakage, so the simulation route needs a substitute key such as strong-model consensus or the compiled-rule decision. I have not tested whether that holds up.
- **Do not copy the AUC numbers from the agent-prediction papers.** Krsteski shows a constant difficulty predictor still gets response AUC 0.715, so those AUCs are not comparable to our rho.

## Thread 1: difficulty from text for LLMs and agents

| Paper (id) | Method and signal | Data | Reported number | Tag |
|---|---|---|---|---|
| Ge et al., Agent psychometrics (arXiv 2604.00594, ICLR 2026 workshop) | IRT plus features from issue, repo, gold patch and tests, as LLM-judge features or embeddings; agent ability split into LLM and scaffold parts | SWE-bench Verified, SWE-bench Pro, GSO, Terminal-Bench 2.0 | Held-out task AUC 0.840 / 0.743 / 0.778 / 0.794 for LLM judge. Embeddings 0.824 / 0.752 / 0.756 / 0.767. Gold patch lifted GSO 0.728 to 0.774. No rho. | V |
| Krsteski and Meyer, Predicting Task Difficulty Without Rollouts (2608.05797) | 1PL difficulty, then a predictor on token entropy of reasoning traces, bge embeddings, length, and disagreement across 5 scorers | 17 benchmarks, 5,230 tasks, 497 agent configurations | Spearman 0.399 K-fold, 0.225 leave-one-benchmark-out. Entropy alone 0.193 / 0.137. Length 0.086 / 0.101. | V |
| Messier (2607.25891) | RidgeCV on bge-base embeddings of the instruction, or instruction plus environment, actions and gold answer | 9,299 tasks | Raw difficulty 0.484 / 0.508 (benchmark-mean baseline 0.465). Within-benchmark z-scored 0.263 / 0.298 (baseline -0.081). | V (full text) |
| Truong et al., amortized model-based evaluation (2503.13335, ICML 2025) | Difficulty predictor on Llama-3-8B embeddings with the dataset description prepended, trained on IRT difficulty | 22 benchmarks, 172 LMs, 217k questions | Train / test AUC of the amortized Rasch fit: 0.85 / 0.83. No rho reported in what I read. | V |
| Zhou et al., ADeLe (arXiv 2503.06378; Nature 2026, venue U) | GPT-4o annotates 18 demand scales, then a random forest predicts instance success | Many benchmarks and LLMs | AUROC 0.839 in distribution, 0.81 task-OOD, 0.75 benchmark-OOD. The embedding baseline falls to 0.48 benchmark-OOD. GPT-4o vs human annotation rho averages 0.86. | V |
| Lugoloobi et al. (2602.09924, ICLR 2026) | Linear probe on pre-generation activations predicts per-model success | E2H-AMC, MATH, code | Beats question length and TF-IDF. Routing saves up to 70% cost. The paper says the model's notion of difficulty differs from the human one. | V (page only) |
| Al-Haque and Johnson (2608.18280, ESEM 2026) | Static patch, repo and prompt features, ensemble models, SHAP | CoderForge-Preview | AUC 0.863. Patch fragmentation and repo scale dominate. Prompt wording matters only for mid-difficulty tasks. | V (abstract) |
| Zotos et al. (2508.03294, ECAI workshop 2025) | Features from a cheap LLM: first-token probability and answer sensitivity to shuffled options; SVM trained on 42 to 47 exam questions | Two ML course exams (NN, AML) | Spearman 0.853 (NN) and 0.582 (AML). Professors -0.02 and 0.17. Direct Gemini 0.28 and 0.35. Tiny n, very noisy. | V |
| Kapoor et al. (2502.20663) | Penalized regression on linguistic features, optional LLM embeddings | NY and TX reading items, grades 3-8 | Correlation 0.77, RMSE 0.59 on the page I fetched, 0.52 in a search snippet (conflict). Embeddings add little. | V, conflict |
| Hoyl (2602.00034) | LLM-extracted features (solution step count, cognitive complexity, misconceptions) plus linguistic features into a neural net, then difficulty from the predicted response pattern | 250k student math responses | Pearson about 0.78 on unseen questions. Trained on real human responses. | V (abstract) |
| Mozafari et al., Q-DAPS (2605.12398, ACL 2026) | Entropy of plausibility scores over candidate answers | TriviaQA, NQ, MuSiQue, QASC | Beats baselines; I did not retrieve a number. | V (abstract) |
| SWE-smith (2504.21798) | Fine-tuned Qwen 2.5 32B labels tasks easy, medium or hard from problem statement plus patch | SWE-smith tasks | 75.3% accuracy. Accuracy, not rho. | U (search snippet) |

**What made the difference where rho was 0.6 or higher:**
- Human respondents' difficulty is the target, with an answer key, and many respondents (Hoyl, Kapoor, Acquaye).
- Gold solution or test information is part of the input. The gold patch gave the biggest gain on GSO.
- Evaluation is within one dataset, not across benchmarks.
- A tiny supervised head sits on top of LLM uncertainty (Zotos). The n is 42 with cross-validation, so I would not read much into it.

## Thread 2: simulated respondents and LLM ensembles

| Paper | Method | Data | Result | Tag |
|---|---|---|---|---|
| Acquaye et al., Take Out Your Calculators (2601.09953) | Role-play students at 4 NAEP levels (25/35/25/15% mix). Score right or wrong, fit IRT. 10 open models; 50, 100 or 300 students per item. | NAEP math, grades 4/8/12 | r 0.75 / 0.76 / 0.82. Weaker Gemma models beat stronger Llama and Qwen. Diverse names beat ids or no id (0.57-0.67 → 0.72-0.74 at grade 4). Stratifying names by gender and race helped further. 50 → 300 students raised r from 0.62 to 0.69 at grade 4. Weighted Gemma ensemble rho 0.82 at grade 12. | V |
| Li et al., Can LLMs Estimate Student Struggles (2512.18880, ACL 2026) | Direct difficulty rating, with a low/average/high proficiency system prompt as an extra condition. 20+ models. | USMLE, Cambridge, SAT Reading, SAT Math | Baseline rho: USMLE 0.13, Cambridge 0.30, SAT Reading 0.29, SAT Math 0.41. Proficiency prompts barely change accuracy (under 1%). Averaging the 4 proficiency configurations raised GPT-5 0.34 → 0.47 and GPT-4.1 0.44 → 0.51; the authors read this as noise averaging, not real student simulation. Larger models converge to a "machine consensus". | V (page partly read) |
| Scarlatos et al., SMART (2507.05129, EMNLP 2025) | Fine-tune simulated students with DPO to follow an instructed ability, score with an LLM, fit IRT | Smarter Balanced (49 items), CodeWorkout (50) | PCC 0.649 / SCC 0.486 on Smarter Balanced. 0.393 / 0.423 on CodeWorkout. Zero-shot response generation PCC 0.265. Small item counts. | V |
| Ormerod (2601.02580) | LoRA Qwen-3 conditioned on 20 ability descriptors; build synthetic item characteristic curves | BEA 2024 (201 test items) | Pearson 0.381, RMSE 0.288 (Qwen-8B). Dummy baseline 0.31, ELECTRA 0.299. | V |
| Rogoz and Ionescu, UnibucLLM (2404.13343, BEA 2024) | Augment training data with answers from Falcon, Meditron and Mistral | USMLE | The abstract says difficulty is harder than response time. No numbers on the page. | V (abstract) |
| Brant et al. (2602.06631) | 10 LLMs vs official IRT parameters | ENEM, 1,031 items | Moderate rank correlation, systematic underestimation, worse on images. | V (abstract) |
| 2605.18562 | Absolute vs pairwise judgments, hard vs probability outputs, zero- vs few-shot | Primary-school math, 3 LLMs | Absolute judgments with token probabilities and few-shot examples matched pairwise. | V (abstract) |

**Tricks that helped:**
- Diversity of simulated respondents (names), many respondents, and weaker models.
- Averaging over personas and models, which works as noise reduction.
- Token-probability outputs and a few-shot calibration.
- DPO alignment to a target ability, which needs real response data.

**Tricks that did not help:** asking a strong model to act weak barely changes its accuracy, and bigger models do not align better.

## Thread 3: human reliability analysis (THERP, HEART, SPAR-H, CREAM)

| Item | What it provides | Tag |
|---|---|---|
| HEART (Williams, 1985) | Generic task type with a nominal error probability. A list of 38 error-producing conditions, each with a maximum multiplier and an "assessed proportion" a in [0,1]. The effect is (m-1)·a+1, multiplied across conditions. | The 38-condition count is V (search snippet). The formula is from memory, U. |
| SPAR-H (INL, NUREG/CR-6883) | A diagnosis part and an action part, each with a nominal error probability. Eight performance shaping factors with multipliers, plus a dependency adjustment. | Document exists, V. Multiplier details U. |
| THERP, CREAM | THERP uses event trees with tabulated probabilities and a dependency model. CREAM maps context to a control mode and an error probability interval. | U |
| Applying HRA to LLM agents | I found no paper that does it. | Not found |

Search results touching HRA and LLMs use LLMs to help human-reliability analysts (2502.00022 and 2605.23927, titles seen only in search snippets, U). A 2025 review (2507.01017) covers AI in HRA workflows (V, abstract). None assesses agent error conditions. So an HRA scoring of LLM agents would be new here, and it would be our own mapping with no prior validation.

## Thread 4: process-model complexity metrics (BPM)

| Paper | Finding | Tag |
|---|---|---|
| Mendling, Neumann and van der Aalst, Understanding the Occurrence of Errors in Process Models based on Metrics (venue U; I think OTM 2007) | Logistic regression of formal errors (deadlocks, soundness) on 2,003 EPC models. Nagelkerke R2 0.901. On 113 held-out textbook EPCs, 90.27% classified correctly (error base rate 21.4%). Metrics include coefficient of connectivity, connector degree, and the mix of connector types. | V (text read) |
| Mendling, Reijers and van der Aalst, Seven Process Modeling Guidelines (7PMG), Information and Software Technology 2010 (venue U) | Guidelines: few elements; low connector degree; one start and one end; structuredness; avoid OR; verb-object labels; decompose above 50 elements. Cites that error probability exceeds 50% above 50 elements, and that structural metrics such as OR-joins and average connector degree correlate negatively with understanding. | V (text read; the figures come from the cited papers, not this one) |
| Mendling and Strembeck, Influence Factors of Understanding Business Process Models (BIS 2008) | Metric-based factors for understanding; existence verified from the search result only. | U |
| Reijers and Mendling, IEEE SMC-A 41(3), 2011 | Factors that influence understandability (modeler expertise, size, structure). | Existence V, numbers not retrieved |

Two caveats for our use:
- These metrics predict formal modeling errors, or human comprehension scores, on process graphs. They do not predict LLM agent failure, and I found no application to SOPs or agents.
- The sample sizes are large and the targets are binary. Applying them to our small procedure sets is a different regime.

## Thread 5: pre-execution failure prediction and uncertainty

| Paper | Signal | Result | Tag |
|---|---|---|---|
| Kaddour et al., Agentic Uncertainty Reveals Agentic Overconfidence (2602.06948) | The agent states P(success) before, during and after the task | Overconfidence: agents succeeding 22% of the time predict 77%. Pre-execution gave better discrimination than post-execution review, not always significant. Adversarial "find the bug" framing calibrated best. | V (abstract) |
| Doomed from the Start (2607.06503) | Linear probes on hidden states, from round 1 | On TextCraft and WebShop, 60.2% and 54.9% fewer generated tokens at 90% recall; behavior-only scorers are near chance at round 1 | V (abstract) |
| No Answer Needed (2509.10625, ICLR 2026) | Question-only linear probe, trained on generic trivia | Beats verbalized confidence. Saturates in middle layers. Fails on math reasoning. | V |
| Lugoloobi 2602.09924 | See thread 1 | | V |
| Knowing Before Saying (2505.24362) | Prompt-only hidden states predict chain-of-thought success | 60% to 76.4% accuracy depending on dataset and model | U (search snippet) |
| Goal-Drift Probes (ICML 2026) | Mid-network probes | AUC 0.989 for failure three steps ahead | U (snippet, no arXiv id) |
| Zotos 2508.03294 | First-token probability and shuffled-option sensitivity | See thread 1 | V |

Probes in this literature are trained within one environment, and several fail on reasoning-heavy tasks. For us the training set is only a handful of procedures, so overfit is the main risk.

## Thread 6: automated discovery and hypothesis generation

| System | Mechanism | Overfitting safeguard | Tag |
|---|---|---|---|
| HypoGeniC (2404.04326, NLP4Science workshop at EMNLP 2024) | LLM proposes hypotheses from labeled examples. UCB-style reward and a wrong-example bank of misclassified cases that triggers new hypotheses. | Reports OOD and held-out evaluation. The authors warn that data-driven hypotheses can be tailored to the dataset. Gains over few-shot: +31.7% synthetic, +13.9% / +3.3% / +24.9% on three real datasets. | V (abstract) |
| Literature Meets Data (2410.17309) | Combines literature-derived and data-derived hypotheses | Gains of 8.97% over few-shot and 3.37% over data-only, plus a human study | V (abstract) |
| CAAFE (2305.03403, NeurIPS 2023) | LLM writes feature code from a dataset description | Keeps a feature only if it improves validation performance; mean ROC AUC 0.798 → 0.822, improved on 11 of 14 datasets | V (the validation rule is from the paper's design as I recall it, U) |
| FunSearch (Nature 2023) | LLM proposes programs, an automated evaluator scores them, evolutionary loop | Needs a fast reliable scoring function | U (secondary) |
| AlphaEvolve (2506.13131) | Evolves whole codebases, with evaluation cascades | Secondary summaries say it uses a held-out evaluator the LLM never sees. I could not verify this in the paper. | U |
| AI co-scientist (2502.18864) | Generate, debate, evolve, with an Elo tournament | Aimed at biomedical hypotheses. No guard against leakage of the kind we face. | V (abstract) |

Our evolutionary feature search (F1) already failed to hold on a fresh benchmark, which fits the warning in HypoGeniC about tailoring to the dataset.

## OpenScience (github.com/synthetic-sciences/openscience)

1. It is an open-source AI research workbench and agent: you give a goal in plain language and a lead agent plans, then executes with shell, Python and R kernels, files, literature and database connectors, and remote compute.
2. It is made by Synthetic Sciences, described as an independent project. The repository was created 2026-07-03, last pushed 2026-10-06, and has about 3,900 stars. These figures are from the GitHub API and are V.
3. The license is Apache-2.0 (V). Its listed pipeline is: planning, step-by-step tool use, parallel bounded workers (one lead plus five domain specialists), and an answer with the full trace and produced files.
4. Its safeguards are visible traces, per-file permissions, and approval gates for risky actions and remote compute. It also claims 75.7% on 70 Terminal-Bench Science tasks and 82.2% on BiomniBench-DA; I could not verify those figures.
5. I found no hypothesis tournament, held-out leakage control or statistical guard in the README. It is infrastructure, not a research method.

Ideas worth borrowing are small: a full audit trace per experiment, a per-session scratch space, and approval gates before spending money. I did not check the code or the GitHub-hosted docs beyond the README.

## Ranked recipes for our setting

Information levels follow the brief: L0 documentation only, L1 a few cheap LLM calls per case, E agents on synthetic cases. Effects are my guesses from the cited papers, not measurements on our data.

1. **Weak-model respondent panel with a substitute key (L1, per case).**
   - What: run 8 or more small or weak models (Gemma-style) with diverse student names, 100 to 300 simulated respondents per case. Each answers "perform or refuse, and which checks come first" on paper.
   - Key: score against a strong-model consensus, or against the compiled-rule outcome. Never score against gold actions. Fit 1PL on the simulated outcomes and take its case difficulty.
   - Why: this is the only recipe in the literature that reaches rho 0.75 or more, but only with a real answer key.
   - Risk: Li et al. found a "machine consensus" problem, so the substitute key may wash out the signal. Pilot on one benchmark first.
   - Cost: thousands of small-model calls per case, cheap locally.
2. **Probability-weighted rubric scores (L1, per case).** Replace argmax rubric ratings (c_lad, ADeLe, PLAN items) with expected values over the 0 to 5 token probabilities, with few-shot anchors from training procedures only, and average over 3 or more phrasings and models. Li et al. and 2605.18562 support averaging and token probabilities. Expected effect is a small gain on c_lad by reducing ties and annotator noise. Cost is near zero.
3. **Pre-execution self-forecast by the real solver model (L1, per case).** Ask Sonnet or Gemini for P(success) with the adversarial "find what could go wrong" framing (Kaddour), over k samples. Verbalized confidence is not on our tried list. Expected effect is small but it varies by case, and only a case-varying signal can move the SOPBench case level. Cost is a few calls per case.
4. **Answer-sensitivity features (L1, per case).** From Zotos: shuffle the option order, paraphrase the request, and measure how often a cheap model's perform or refuse answer flips, plus first-token probability. This differs from our failed first-action logprob margin because it measures stability under perturbation. It varies by case. Zotos reached 0.58 to 0.85, but with n of about 42, so expect much less.
5. **Leave-procedure-out hidden-state probe (L0 and L1).** Ridge on mid-layer last-token activations of policy plus tools plus request from the local Qwen, target is z-scored b within the benchmark, strong shrinkage, grouped by procedure. The literature gives AUC, not rho, and several probes fail on reasoning-heavy tasks. Expect little at the case level, possibly more at the procedure level. Risk is overfit with few procedures. Cost is moderate GPU time.
6. **HRA-style multiplicative scorer (L1 per procedure).**
   - What: define about 10 error-producing conditions for agents, such as skippable checks, ambiguity, conflicting rules, irreversible writes, many tools and numeric boundaries. An LLM rates each on 0 to 1 per case. Combine them as log nominal + Σ log((m-1)·a+1) with prior multipliers, not fitted ones.
   - Why: it is a different functional form from additive c_lad, and fixed priors cannot overfit with few procedures.
   - Risk: expected gain is small, +0.01 to +0.03, probably redundant with c_lad. Nothing validates the mapping to agents.
7. **Safeguarded feature-discovery loop (HypoGeniC, CAAFE).**
   - Rules: use the wrong-example bank (largest residual cases) to prompt new features. Require each feature to be code-computed from L0 or L1 inputs. Score only by leave-procedure-out on dev benchmarks and freeze before touching a fresh one.
   - Controls: count every candidate tried, and compare the best against a permutation null for the maximum over K features. Allow a feature in only if it helps in at least 2 benchmarks.
   - Why: F1 failed without these. This is a protocol fix more than a new signal.
8. **Two-level decomposition with honest targets (analysis, L0).** Predict b as procedure score (PRI, CMP, c_lad) plus a within-procedure term, and report both parts separately. Messier's within-benchmark ceiling of 0.26 to 0.30 from text embeddings suggests case-level text signal is weak. We have a ceiling table: on SOPBench the procedure × label cell-mean oracle is 0.614. It sets the right expectation and costs nothing.
9. **BPM metrics on the compiled policy graph (L0, procedure level).** Connector degree maximum and average, coefficient of connectivity, structuredness, OR counts, start and end counts, element count above 50. Known predictors of formal errors (R2 0.901 on EPCs), but the targets differ and these vary only by procedure. The council already listed them. The concrete step is a 7PMG checklist computed on our compiled programs, tested only at the procedure level.
10. **Solution-aware features (L1, already partly done).** Agent psychometrics got its largest gain from the gold patch on GSO. Our analogue is an LLM-written reference plan plus tests, which is PLAN. The remaining lever is to expand it with LLM-written test and edge-case lists, never gold.
11. **Amortized difficulty head with benchmark-level normalization.** Train on all benchmarks' z-scored b with dataset descriptions prepended and Llama-3-8B embeddings (Truong). Messier suggests the ceiling is about 0.3, so treat it as a baseline only.
12. **First-turn probe trained on emulated runs (E).** Run weak agents on synthetic cases, train a Doomed-from-the-Start style probe on the first round, apply it to real cases. Costly and unproven outside TextCraft and WebShop. Keep it as a last resort.

Recipes 1 to 4 are the cheapest and most likely to add case-level signal. Numbers I could not verify: SWE-smith 75.3%, the Goal-Drift AUC, the AlphaEvolve held-out evaluator claim, the Nature venue for ADeLe, the Kapoor RMSE conflict (0.52 vs 0.59), and the Mendling OTM 2007 venue.
