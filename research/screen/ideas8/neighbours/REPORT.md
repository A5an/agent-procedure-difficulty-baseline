# Swarm agent `neighbours`, round 8 (6 Oct 2026)

Saved by the main session from the agent's final message.

# Round 8, agent `neighbours`: transfer signals from neighbouring fields and public agent datasets

I wrote no files and ran no quick screen or agent runs. Every number below was read from the source unless it is marked "unverified". Ideas that go beyond the seed fields are marked "beyond seeds".

## Part A. What transferred across datasets in each field

**Query performance prediction in IR (QPP)**
- Pre-retrieval predictors are specificity (MaxIDF, AvgIDF), collection-query similarity (SCQ, SCS/AvICTF) and variance of term weights (VAR). The reference surveys are Hauff, Hiemstra and de Jong, CIKM 2008, and Carmel and Yom-Tov 2010.
- The 2025 reproducibility study (Saha et al., arXiv 2503.24251) "largely supports" the 2009 findings of Hauff et al. A search summary says MaxIDF, AvgIDF, MaxVAR and AvgVAR were consistently top across collections, while sparse predictors worked on TREC ROBUST and GOV2 but failed on WT10G and MS-MARCO. I did not open the full paper for that wording.
- Correlation numbers: unverified. I could not extract them from the papers.
- LLM-era QPP is QPP-GenRE (arXiv 2404.01012, ACM TOIS 2025). An LLM judges the relevance of the top-k results and the judgments are turned into a predicted metric. I did not check the numbers it reports.
- Feature families that held up: term specificity and term-weight variance (L0). Post-retrieval score-shape predictors (clarity, NQC) need a ranked list.
- Lesson: simple specificity predictors are the stable ones. Many others are collection-specific.

**Reference-free quality estimation (QE) in machine translation**
- Fomicheva et al. 2020 (arXiv 2005.10608, TACL) tested glass-box indicators on six language pairs. The table below gives Pearson r with human direct assessments, read from the paper's Table 2.

| Indicator | Si-En | Ne-En | Et-En | Ro-En | En-De | En-Zh |
|---|---|---|---|---|---|---|
| TP (sequence log-probability) | .399 | .482 | .486 | .647 | .208 | .257 |
| Softmax entropy (-) | .457 | .528 | .421 | .613 | .147 | .251 |
| Sent-Std (-) | .418 | .472 | .471 | .595 | .264 | .301 |
| D-TP (dropout average) | .460 | .558 | .642 | .693 | .259 | .321 |
| D-Lex-Sim (similarity of dropout samples) | .513 | .600 | .612 | .669 | .172 | .313 |
| AW:Ent-Min (attention entropy) | .097 | .265 | .329 | .524 | .000 | .067 |
| Supervised BERT-BiRNN | .473 | .546 | .635 | .763 | .273 | .371 |

- Lesson 1: absolute strength changes about three times across pairs, from about 0.2 to 0.7. The best indicator also changes from pair to pair.
- Lesson 2: probability-based and sample-consistency indicators are consistently usable. Attention entropy is the worst and sometimes about 0.
- Lesson 3: sampling several outputs (D-TP, D-Lex-Sim) beats one deterministic pass. Sent-Std (std of token log-probabilities) beats entropy on the two high-resource pairs, En-De and En-Zh.

**Algorithm selection and instance hardness (SATzilla, empirical hardness models)**
- SATzilla uses about 60 polynomial-time instance features and ridge regression on log runtime (confirmed by search summary).
- The authoritative evaluation is Hutter, Xu, Hoos and Leyton-Brown (arXiv 1211.0906, AIJ 2014): 11 algorithms, 35 instance distributions, SAT, TSP and MIP. A random forest reaches RMSE 0.47 in log10 runtime on SAT, a factor of about 3. Even that was enough to win SAT medals.
- Generalisation to new instances and new configurations was about as good as either alone, but the heterogeneous mix CPLEX-BIGMIX was much worse.
- Transfer lesson: models are trained per distribution. The cross-distribution generalisation I checked is only to new instances within a known distribution, not to a new distribution. I could not verify which feature family travels best across distributions, so I mark that "unverified".
- Instance space analysis (Smith-Miles et al., Computers & Operations Research 2014, vol. 45): the method projects instances onto a 2D space and draws per-algorithm footprints. It claims prediction on unseen instances "with high accuracy". I did not check the numbers.

**Software defect prediction**
- Zimmermann et al., ESEC/FSE 2009: 622 cross-project predictions across 12 applications. Only 21 (3.4%) had precision, recall and accuracy all above 0.75. Same-domain or same-process pairing did not guarantee transfer (search summary).
- Meta-analysis lesson: nearest-neighbour and decision-tree models do well, naive Bayes is average, and ensembles vary widely (search summary of an SLR on cross-project defect prediction). Transfer-learning methods (distribution alignment) helped for some projects.
- Lesson: raw feature values do not transfer across projects, because the scale of code metrics is project-specific. Normalising within a project is standard. This is exactly our pooled within-domain Spearman setup.

**Test item difficulty prediction (BEA 2024 shared task, Yaneva et al., aclanthology 2024.bea-1.39)**
- 667 retired USMLE items, 17 teams. The best RMSE was 0.299 (EduTec, ELECTRA) against a dummy regressor at 0.311.
- Linguistic features plus a linear model (British Council) ranked 5th to 7th with RMSE 0.305.
- LLM-answer features were no better: UnibucLLM got 0.308.
- Lesson: item text alone barely beats the mean (3.8% of RMSE). Do not expect text features to carry difficulty on their own.

**Crowdsourcing task difficulty (Yang et al., HCOMP 2016, "Modeling Task Complexity in Crowdsourcing")**
- Three feature classes: metadata, content (task description language) and visual. Description language and appearance predict perceived complexity. Complexity is perceived coherently across workers and is significantly influenced by task type.
- It reports no cross-platform transfer, as far as I read.
- Lesson: the task-type prior is the strongest stable factor.

**Process model understandability (Mendling, Reijers and Cardoso, BPM 2007; Mendling, Strembeck and Recker, DSS)**
- Size, connector structure, structuredness and labelling guidelines (the 7PMG rules, including verb-object labels) were the factors. The Cardoso and Mendling metrics are already on our list.
- Numbers: unverified.
- Not yet listed on our side: label quality (verb-object labelling), and missing-join connector mismatch.

## Part B. The idea catalog (14 ideas, tier 1 to 5)

For each idea, "model" means which model is used. All case-level claims say whether the signal varies between cases of the same procedure.

**N01. QPP-specificity of the request against the policy (tier 1)**
- Measures: how specific the request vocabulary is for the policy and tool text. Specific requests tell the agent which clause applies. Targets wrong-clause selection.
- Inputs: policy, tools, request. No model.
- Recipe: build one corpus from policy sentences plus tool descriptions. Compute IDF over that corpus. For each request, compute MaxIDF, AvgIDF, SumSCQ, MaxSCQ (SCQ = (1 + ln tf_coll) · ln(1 + N/df)) and the variance of tf-idf weights. Replace each feature by its percentile within the procedure.
- Level: L0, free. Varies by case.
- Universality: SOPBench yes. tau2 yes. Banking yes. New process yes. Requests are free text everywhere. Weakness: SOPBench requests are templated, so variance is low.
- Status: new. The nearest tried idea is "clause retrieval margin", which uses ranks, not term statistics.
- Evidence: Hauff et al. 2008 survey and the 2025 reproducibility study (see Part A). Numbers: unverified.
- Expected effect: case level small, ρ below 0.1 for SOPBench. The risk is high that it captures only length.

**N02. Within-procedure rank normalisation of every feature (tier 1, methodology)**
- Measures: nothing new. It removes the project-scale effect that broke defect-prediction transfer.
- Recipe: before pooling or fitting, replace every case feature by its within-procedure percentile. Also keep the raw value. Fit the pooled model only on percentiles plus a few procedure-level features.
- Level: L0, free. Varies by case.
- Universality: all four, because it is a transform.
- Status: new as a recipe. The related literature is normalisation in cross-project defect prediction. I did not verify a specific transfer-learning paper.
- Evidence: Zimmermann 2009 (3.4% cross-project success, see Part A). The normalisation advice is own reasoning.
- Expected effect: it can lift weak features and kill artifacts. The risk is that it removes real between-procedure information, which matters at the procedure level (0.5 to 0.7).

**N03. Request-type prior, normalised within type (tier 2)**
- Measures: the type of action requested (lookup, modify, cancel, refund, transfer). Task type is the strongest stable factor in crowdsourcing. Targets: write actions that need extra checks.
- Inputs: request and tools. One cheap LLM call, or a keyword rule on tool verbs.
- Recipe: map the request to the tool it most plausibly triggers by embedding similarity to tool descriptions. Take that tool's class (read, write, destructive) from the tool name. The feature is the class, plus the percentile of its schema complexity among tools of the same class.
- Level: L0 or L1. Varies by case only if the procedure has several request types. SOPBench: mostly one tool per request, so weak.
- Universality: tau2 yes (many tools). SOPBench partly. Banking yes. New process yes.
- Status: variant of PLAN (writes and reads). Difference: no plan generation, just a tool-class prior with within-class normalisation.
- Evidence: Yang et al. 2016 (task type significantly influences complexity).
- Expected effect: case level small on tau2 only.

**N04. Clarity and NQC of the clause retrieval (tier 2)**
- Measures: how sharply the request points at policy clauses. Post-retrieval IR predictors (clarity score, NQC) correlate with query quality. Targets clause confusion and skipped checks.
- Inputs: policy, request. A local embedding or a BM25 model.
- Recipe: split the policy into clauses. Score the request against each clause. Compute (a) NQC = std of the top-k scores divided by the mean score; (b) clarity = KL divergence between the unigram language model of the top-k clauses (weighted by score) and the whole policy; (c) the entropy of the softmax over clause scores.
- Level: L0, small cost. Varies by case.
- Universality: all four. It needs only clause splitting.
- Status: variant of "clause retrieval margin". It differs because it uses the whole score distribution (spread, KL, entropy) and not the gap between rank 1 and rank 2.
- Evidence: clarity score defined as relative entropy of query language model against collection language model (Cronen-Townsend et al., SIGIR 2002; definition confirmed from a search snippet; NQC paper not checked, "unverified").
- Expected effect: small, case level ρ about 0.05 to 0.1. High risk because SOPBench policies are short.

**N05. Sample-consistency of the agent's written plan (tier 2)**
- Measures: how much several samples of one cheap model disagree on the next required action and check list. Directly adapts D-Lex-Sim and D-TP from MT QE. Targets plan instability.
- Inputs: policy, tools, request. A black-box API (a cheap model, temperature above 0), 5 samples.
- Recipe: sample N=5 plans, each as an ordered list of tool names with arguments. Compute (a) the mean pairwise normalised edit distance of the tool-name sequences; (b) the Jaccard of checks; (c) for local models, the mean and std of token log-probabilities over the plan (TP, Sent-Std).
- Level: L1, 5 calls per case. Varies by case.
- Universality: all four.
- Status: variant of PLAN, compile disagreement (failed) and two planners (n.s.). Difference: within-model sampling consistency, with edit distance on the action sequence, and with sequence-level uncertainty from the same samples. The attention-entropy version should be skipped because Fomicheva found it worst.
- Evidence: Fomicheva 2020 Table 2 (D-Lex-Sim 0.513 to 0.669 on four lower-resource pairs, 0.172 to 0.313 on En-De and En-Zh).
- Expected effect: case level maybe +0.01 to +0.03 over c_lad. High risk, since three related ideas failed.

**N06. Else-coverage: share of conditions with no stated failure consequence (tier 3)**
- Measures: policy branches that open with "if" and never say what happens in the complement. This is the process-model connector mismatch (a split without a matching join), applied to text. Targets silent-skip failures, where agents act despite a failed condition (78% of must-refuse failures) because the policy does not say to refuse.
- Inputs: policy. A black-box LLM.
- Recipe: ask the LLM to list every condition in the policy and, for each, quote the sentence that states what happens when it fails, or say "none". The feature is the fraction with "none". For case level, check whether the request touches a condition whose complement is silent (one more LLM call per case: "which conditions are relevant to this request?").
- Level: L1, 1 call per procedure plus 1 per case. Varies by case in the second form, by procedure in the first.
- Universality: all four. It requires only policy text. SOPBench policies are mostly explicit, so the signal there may be low.
- Status: new. It differs from "polarity and exceptions", which counts negations and exceptions and not unstated complements.
- Evidence: Mendling et al., connector mismatch and the 7PMG guideline (understandability factors; numbers unverified). Own reasoning for the mapping.
- Expected effect: procedure level plausible at 0.3 to 0.5; case level small. Risk: LLM mislabels.

**N07. Label quality of policy clauses (verb-object labelling) (tier 3)**
- Measures: how clearly each clause names an action and its object. Process modelling guidelines identify this as an understandability factor.
- Inputs: policy. A cheap LLM or a part-of-speech tagger.
- Recipe: for each clause, test whether it parses as "verb + object" (an imperative with a clear tool or field). Score = the fraction of clauses that do. Also count nominalised or vague actions ("handle", "process", "ensure").
- Level: L0 with a tagger, or L1. Varies by procedure only.
- Universality: all four.
- Status: new. It differs from "condition packing density" and "polarity and exceptions".
- Evidence: 7PMG verb-object labelling (Mendling et al.; summarised via search, numbers unverified).
- Expected effect: procedure level small; no case-level effect. Risk: it correlates with text length.

**N08. Case-level constrainedness ratio (tier 3, beyond seeds)**
- Measures: the ratio of facts the agent needs to check to facts the request already supplies. This follows the clause-to-variable ratio in SAT hardness (Leyton-Brown and colleagues; the classic phase-transition result is unverified here). Targets skipped lookups, where the agent assumes what the user did not give.
- Inputs: policy, tools, request. A cheap LLM, one call.
- Recipe: from the policy, list the facts needed to decide this request (for example "membership level", "days since purchase"). From the request, mark which are stated. Feature = needed minus stated, divided by needed. Also the count of needed facts that are only available from a tool.
- Level: L1, 1 call per case. Varies by case, and strongly in tau2.
- Universality: tau2 yes. Banking yes. SOPBench not for the perform/refuse label, because the hidden database decides it, but the count of facts to fetch is still computable. New process yes.
- Status: variant of "user assertion and action readiness". Difference: a normalised ratio with an explicit fetched-versus-stated split, not a binary readiness flag.
- Evidence: own reasoning; SAT hardness analogy unverified.
- Expected effect: case level on tau2 possibly 0.05 to 0.1. Risk: it overlaps with PLAN reads.

**N09. Request diffusion entropy (tier 3, beyond seeds)**
- Measures: how widely a request spreads over entities, tools and policy sections, borrowed from just-in-time defect prediction (the "diffusion" of a change over files and subsystems). Targets agents that finish one part of a multi-part request and drop another.
- Inputs: policy, tools, request. One cheap LLM call.
- Recipe: extract the entities touched and the policy sections each requires. Compute the number of distinct sections, and the Shannon entropy of the distribution of required checks across sections.
- Level: L1. Varies by case.
- Universality: tau2 yes (multi-request users). SOPBench mostly single-section, so weak. Banking yes. New process yes.
- Status: variant of SCN (number of requests) and "simultaneous load". Difference: entropy over policy sections, not a count of requests.
- Evidence: change-diffusion features from Kamei et al., TSE 2013 and Hassan, ICSE 2009: unverified.
- Expected effect: tau2 case level small. Risk: same as SCN.

**N10. Pointwise information gain of the request given the policy (tier 4, beyond seeds)**
- Measures: how much the policy explains the request. If log p(request | policy) is barely above log p(request), the request is out of the policy's expected space, so the agent improvises. It is the language-model version of the query-clarity idea, with the request as the "query" and the policy as the "collection". Targets out-of-scope and unusual requests.
- Inputs: policy, request. A local open model (for example Qwen3-8B-Base), teacher-forced scoring only, no generation.
- Recipe: compute the mean token log-probability of the request twice, once with the policy in the prompt and once without. Feature = the difference, and also the mean log-probability under the policy prompt alone. Normalise within the procedure.
- Level: L1 for case-level reading of the request (L0 for the policy alone). It varies by case. Cost: two forward passes per case.
- Universality: all four.
- Status: new. PRI scores the policy rules alone, at the procedure level. This scores the request conditioned on the policy.
- Evidence: own reasoning. The clarity-score definition is the only support (see N04). The risk of confounding with request length is high, so use per-token averages.
- Expected effect: case level 0.03 to 0.08 if the request differs in how templated it is; for SOPBench the requests are templated, so little variation. High risk.

**N11. Per-agent empirical hardness models, then a weighted combination (tier 4, beyond seeds)**
- Measures: whether predicting difficulty separately for strong and weak agents (SATzilla trains one hardness model per solver) and then combining beats one pooled IRT difficulty. Our fact 4 says top-5 and bottom-5 agents agree only at ρ 0.41 on cases.
- Inputs: whatever features we already have; only the targets change. No model call.
- Recipe: from the per-case outcome matrix, compute the difficulty for the top-half and bottom-half of agents separately. Train one regressor per group on the existing features. Predict the pooled difficulty as a convex combination whose weight is chosen by within-domain cross-validation, and report which group each feature predicts better.
- Level: L0 relative to the features. Varies by case.
- Universality: all four, since it needs only the outcome matrix and features. It needs at least some agents on each benchmark, which we have.
- Status: new.
- Evidence: SATzilla and Hutter et al. 2014 build per-algorithm models (confirmed). Whether this helps a pooled difficulty target is own reasoning.
- Expected effect: case level +0.01 to +0.03 at best; its main value is diagnostic (which features track strong versus weak agents).

**N12. Cross-domain feature stability selection, from instance space analysis (tier 4)**
- Measures: which features keep the same sign and size of correlation with difficulty in every training domain. Instance space analysis selects features that separate algorithm performance consistently. Targets the universality requirement itself.
- Inputs: all existing features. No model call.
- Recipe: for every candidate feature compute the within-domain Spearman with IRT difficulty for each domain separately. Keep features whose sign agrees in all training domains and whose minimum absolute value exceeds a threshold. Rank by the worst-domain value, not the average. Fit the final model only on the survivors.
- Level: L0, free. Varies by whichever feature it keeps.
- Universality: yes, by design. It needs at least 2 training domains with several procedures each.
- Status: variant of F1 (evolutionary search, did not hold on a fresh benchmark). Difference: a conservative filter on worst-case sign stability, with no search over feature combinations.
- Evidence: Smith-Miles et al. 2014 (feature selection for instance space; the exact selection method I did not check).
- Expected effect: it should reduce the gap between training and fresh domains more than it raises the best number. Risk: it throws away features that matter in only one domain.

**N13. Last-token hidden-state probe on "policy plus request" (tier 5)**
- Measures: whether a local model's internal state after reading the policy and the request encodes a difficulty direction that transfers across domains. White-box analog of glass-box QE and of supervised estimators.
- Inputs: policy, request. A local open model.
- Recipe: extract the last-token hidden state at a middle and a late layer for each case. Train a ridge probe on difficulty with leave-one-domain-out evaluation. Compare with the pooled embedding baseline.
- Level: L1 (reads the case). One forward pass per case.
- Universality: all four.
- Status: variant of embeddings (bge, the fine-tuned encoder at 0.255, mostly tau2). Difference: a decoder-state probe on the model's own representation of the whole prompt, with leave-one-domain-out training.
- Evidence: Fomicheva's supervised BERT-BiRNN beat all unsupervised indicators on every pair (0.273 to 0.763), so supervised heads win when labels exist. No agent-domain result known.
- Expected effect: probably the same 0.25 level. Risk: overfits to domain.

**N14. Landmarking with a weak solver on paper (tier 5, beyond seeds)**
- Measures: how a deliberately weak model does on a small standard battery of rule-checking questions drawn from the policy, as meta-learning "landmarking" does (cheap learners as features; unverified here). It targets the procedure, not the case.
- Inputs: policy. A cheap or small local model.
- Recipe: generate 20 yes/no questions per policy clause from templates, with known answers from the policy ("Is a refund allowed after 30 days?"). Score the small model's accuracy and calibration. The feature is the accuracy per procedure.
- Level: L1, procedure level.
- Universality: all four.
- Status: variant of PRI and of the written-condition skip model. Difference: it measures a small model's ability to answer questions about each rule, a direct probe of how well a weak reader can use the policy.
- Evidence: own reasoning.
- Expected effect: procedure level 0.3 to 0.5, possibly correlated with PRI. No case-level effect.

## Part C. Top 3

1. **N12 (stability selection).** It attacks the universality requirement directly, it is free, and it can be tested immediately on existing features. The fresh-domain failure of F1 suggests our features are fitted to domains. A worst-domain filter is the cheapest defence. The literature support is thin, so it rests on defect-prediction transfer failure (3.4% cross-project success) as a warning.
2. **N08 (constrainedness ratio).** The brief says agents fail by skipping checks, and the mechanism is that the agent has to look something up that the user did not state. This ratio varies by case, needs one cheap call, and is computable on any procedure. Risk: overlap with PLAN reads.
3. **N06 (else-coverage).** It is the one idea that comes from a field (process model connector mismatch) and targets the 78% "act despite a failed condition" failure. The procedure-level version should work; the case-level version asks whether the request hits a silent branch.

Also worth testing for free: N02 (rank normalisation), because it matches our metric exactly.

## Part D. Public datasets with per-case outcomes for several agents

Each row was checked on the official GitHub or Hugging Face page. "Not stated" means the page I read did not say.

| Dataset | Policy text | Tool schemas | Request | Agents | Per-case outcomes public | License |
|---|---|---|---|---|---|---|
| tau-bench (airline, retail), github.com/sierra-research/tau-bench | yes | yes | yes (user instruction) | 2 models in `historical_trajectories` (GPT-4o, claude-3.5-sonnet-new); repo says tasks are no longer updated | yes, JSON per model and domain | MIT |
| tau2 / tau3-bench (airline, retail, telecom, mock, banking_knowledge), github.com/sierra-research/tau2-bench | yes | yes | yes | many on taubench.com; exact count not stated | yes: trajectories are on S3 (`s3://sierra-tau-bench-public/submissions/...`), not in the repo; submission has Pass^k per domain | MIT |
| tau-Knowledge (banking_knowledge, inside tau2-bench) | knowledge base documents instead of one policy | yes | yes | leaderboard includes knowledge results | yes, via the same S3 submissions | MIT; v1.0.0 notes 75+ task fixes |
| ThinkingBox-Bench, github.com/microsoft/thinkingbox, arXiv 2608.19741 | yes (prompts built from policy) | yes (MCP servers) | yes (scenario with initial state) | paper tests at least Claude Opus 5 and Kimi-K3; I found conflicting top numbers (66.50% pass@1 in the abstract fetch, 65.36% in a search summary), so unverified | the repo does not release multi-model traces; the per-task numbers would be in the paper's tables | code MIT, paper CC BY 4.0 |
| CRMArena-Pro, HF `Salesforce/CRMArenaPro` | in the system prompt metadata | via the Salesforce environment | yes (`query`, `persona`) | none in the dataset (8,614 rows, 22 task names, 19 tasks) | no model outputs; ground truth only | CC-BY-NC-4.0 on HF (a search summary said CC BY 4.0, so check before use) |
| WorkArena (L1, L2/L3, ++), github.com/ServiceNow/WorkArena | no explicit policy | via the ServiceNow UI | yes (33 L1 tasks, 19,912 instances; WorkArena++ 682 tasks) | via AgentLab leaderboard; not stated | not stated | a LICENSE file exists; type not shown (unverified) |
| ToolSandbox, github.com/apple/ToolSandbox | no | yes | yes (scenarios, scripted user) | one example trajectory (gpt-3.5-turbo-0125) in README | no multi-model release | LICENSE file exists; type not shown (unverified) |
| BFCL v3 multi-turn, github.com/ShishirPatil/gorilla | no | yes (function docs) | yes | large list in SUPPORTED_MODELS.md | yes: `result/MODEL/BFCL_v3_*_result.json` and `score/MODEL/*_score.json` | Apache 2.0 |
| AppWorld, github.com/StonyBrookNLP/appworld | no | yes (API docs) | yes | leaderboard repo `stonybrooknlp/appworld-leaderboard` | encrypted `.bundle` files; test-set ground truth withheld | Apache 2.0 with encryption requirements |
| TheAgentCompany, github.com/TheAgentCompany/TheAgentCompany | task description and workspace | environment tools | yes (`instruction/task.md`) | several models in `TheAgentCompany/experiments` (for example OpenHands with Claude 3.5 Sonnet, Gemini 1.5 Pro) | yes: `results/eval*.json` and `trajectories/*.json.gz`, 175 tasks | repo MIT; the experiments repo states no license |
| HAL traces, hal.cs.princeton.edu and github.com/benediktstroebl/hal-harness | depends on benchmark | depends | depends | many agents across SWE-bench Verified, USACO, AppWorld, CORE-bench, tau-bench, SciCode and others | yes, full traces downloadable but encrypted (`hal-decrypt`); repo archived 1 July 2026, no new submissions | not specified on the leaderboard page |

Notes on use for universality tests:
- Best match to our three inputs (policy, tool schemas, request) with several agents and per-case outcomes: tau2 including banking_knowledge, then tau-bench's two small trajectory sets, then ThinkingBox if its per-task results can be extracted from the paper or released traces.
- ThinkingBox is the best new, unseen source of new business procedures (retail, hospitality, auto insurance, neobank IT, consulting). It needs a check on whether per-case outcomes for more than two models exist. The repo does not show them.
- BFCL and AppWorld have per-case outcomes for many models but no policy text, so they test tool-schema signals only.
- TheAgentCompany has policy-like task text, outcomes for several models and 175 tasks, but the agents are open-ended software agents, not customer-service procedures.
- HAL and AppWorld outputs are encrypted on purpose, so using them needs the decrypt step, and I did not verify the licence terms for traces.
- Not verified: the number of agents on the tau2 leaderboard, the licence of WorkArena and ToolSandbox, and whether CRMArena-Pro's licence is CC-BY or CC-BY-NC (the HF page says NC).

## Quick screen
Not run.
