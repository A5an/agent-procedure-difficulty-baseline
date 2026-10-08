# Literature check for the overnight round (2026-10-04)

Project: predict from procedure documentation (policy text plus customer request) how reliably LLM agents will execute it, before running them. Baseline is Agent psychometrics (Ge et al., arXiv 2604.00594, two-stage IRT then ridge) with ADeLe demand levels (Zhou et al., Nature 2026) as features. Metric is within-domain Spearman (protocol of Krsteski and Meyer, arXiv 2608.05797).

Verification key. "Crossref" means the DOI resolved at api.crossref.org and title, year and first authors matched. "arXiv" means export.arxiv.org id_list returned the matching title and first author. "ACL" means the ACL Anthology bib file for the paper id matched. "OpenLibrary" means a book record matched. "IEC" means the IEC webstore page title matched. Semantic Scholar was used for venue only and was rate limited (429) on later calls. Scripts: cr.sh, crq.sh, ax.sh in this folder. Honest limits: abstracts were read, full texts were not. Venue claims that a script could not confirm are marked "venue unconfirmed". Where a source is arXiv only, it is marked preprint. The arXiv ids in the 2026 range were all resolved by the arXiv API in this session, but I did not cross-check them against any other index.

## Baseline anchors (for reference)

| Source | Verified | What it shows |
|---|---|---|
| Zhou et al. 2026, Nature, General scales unlock AI evaluation with explanatory and predictive power. doi 10.1038/s41586-026-10303-2 (arXiv 2503.06378) | yes, Crossref plus arXiv | ADeLe demand levels as task features for predicting LLM success |
| Ge, Kryvosheieva, Fried 2026, arXiv 2604.00594, Agent psychometrics (preprint) | yes, arXiv | IRT plus task features for agent success on coding benchmarks, our baseline family |
| Krsteski and Meyer 2026, arXiv 2608.05797v2, Predicting Task Difficulty Without Rollouts (preprint) | yes, arXiv | ex ante difficulty prediction over 17 agentic benchmarks, says AUC can mislead (abstract only read) |

## Idea 1. LLM dry run of a case against the policy, used as difficulty features

| Source | Verified | What it shows | Novelty risk |
|---|---|---|---|
| Kadavath et al. 2022, arXiv 2207.05221, Language Models (Mostly) Know What They Know (preprint) | yes, arXiv | models predict P(IK), whether they will answer a question correctly, before answering. Supports the idea that a pre-execution self-assessment carries difficulty signal | low |
| Yao et al. 2024, arXiv 2406.12045, tau-bench (preprint, later conference paper, venue unconfirmed) | yes, arXiv | defines the policy plus user request plus tool setting we predict on. Not a dry run method | none, context only |
| Winston, Winston, Just 2026, arXiv 2603.20449, Solver-Aided Verification of Policy Compliance in Tool-Augmented LLM Agents (preprint) | yes, arXiv | translates policy text to SMT constraints to check tool calls against policy. Related in spirit (formalising which conditions are checkable) but used for enforcement, not difficulty prediction | low to medium |
| Li et al. 2025, arXiv 2503.08669, SOPBench (preprint) | yes, arXiv | procedure-following benchmark with executable constraints. Source of our domain data, not a method | none |
| Cemri et al. 2025, arXiv 2503.13657, Why Do Multi-Agent LLM Systems Fail? (preprint) | yes, arXiv | failure taxonomy that could seed which policy conditions to probe in a dry run | low |

Closest existing work. I found no paper that runs an LLM dry run of a case against a policy and uses the output (count of checkable conditions, predicted violations) as features for agent difficulty. Nearest are P(IK) style self-prediction (Kadavath 2022) and policy-to-logic compliance checking (Winston 2026). Small caveat: my arXiv keyword search is weak, so a hit may exist under other wording such as "plan-then-verify" or "pre-execution feasibility". Novelty risk: low to medium.

## Idea 2. Comparative judgment, Bradley-Terry scores from LLM pairwise comparisons for difficulty

| Source | Verified | What it shows | Novelty risk |
|---|---|---|---|
| Thurstone 1927, Psychological Review, A law of comparative judgment. doi 10.1037/h0070288 | yes, Crossref | the original scaling of stimuli from pairwise judgments | none |
| Bradley and Terry 1952, Biometrika, Rank analysis of incomplete block designs I. doi 10.2307/2334029 | yes, Crossref | the BT model | none |
| Pollitt 2012, Assessment in Education, The method of Adaptive Comparative Judgement. doi 10.1080/0969594X.2012.665354 | yes, Crossref | adaptive pairwise judging gives reliable scales in assessment. (A second DOI, ...694697, is a one page item, not used) | none |
| Qin et al. 2024, Findings of NAACL, LLMs are Effective Text Rankers with Pairwise Ranking Prompting. doi 10.18653/v1/2024.findings-naacl.97 | yes, Crossref plus arXiv 2306.17563 | LLM pairwise prompts rank items well | none |
| Liusie, Manakul, Gales 2024, EACL, LLM Comparative Assessment. doi 10.18653/v1/2024.eacl-long.8 | yes, ACL bib plus arXiv 2307.07889 | pairwise LLM judgments beat scoring prompts, with BT style aggregation | none |
| Liusie et al. 2024, arXiv 2405.05894, Efficient LLM Comparative Assessment, Product of Experts (preprint) | yes, arXiv | fewer than all pairs suffice to recover the ranking | none |
| Ballon et al. 2025, arXiv 2512.14220, Estimating problem difficulty without ground truth using LLM comparisons (preprint) | yes, arXiv | LLM pairwise "which is harder" plus Bradley-Terry scores. Pearson r at least 0.80 with human difficulty annotations on n=1876 | HIGH for the generic method |
| Wu et al. 2026, arXiv 2605.17110, Evidence-Calibrated Query Clustering (preprint) | yes, arXiv | BT model on posterior model comparisons to describe query capability demand | low |

Closest existing work. Ballon et al. 2025 (preprint) is a direct hit: an LLM performs pairwise difficulty comparisons and Bradley-Terry scores serve as difficulty. They validate against human annotations of math style problems, not against agent success rates and not on procedure documents. So the generic method is NOT novel. What remains open is applying it to procedure documentation and testing it against fitted agent IRT difficulty with within-domain Spearman. Novelty risk: high for method, low to medium for the application.

## Idea 3. In-context regression with labelled anchors

| Source | Verified | What it shows | Novelty risk |
|---|---|---|---|
| Vacareanu et al. 2024, From Words to Numbers: Your LLM Is Secretly A Capable Regressor When Given In-Context Examples. arXiv 2404.07544 | arXiv yes. COLM 2024 venue unconfirmed by script (Semantic Scholar lists only arXiv, WebSearch showed only arXiv) | LLMs given in-context (x,y) pairs do regression rivalling random forest and gradient boosting | none |
| Kolesnikova, Fedyanin, Hofman 2026, arXiv 2605.18562, Estimating Item Difficulty with LLMs as Experts (preprint) | yes, arXiv | factorial study of LLMs as difficulty raters with zero-shot and few-shot prompts, few-shot gives examples with known empirical difficulty (from abstract and WebSearch snippet) | HIGH for education items |
| Razavi and Powers 2025, arXiv 2504.08804, Estimating Item Difficulty Using LLMs and Tree-Based ML (preprint) | yes, arXiv | direct LLM rating versus LLM-extracted features plus tree models for K-5 items | medium |
| Wang et al. 2026, arXiv 2607.28634, Can LLMs Really Understand Item Difficulty Levels? (preprint) | yes, arXiv | prompt strategies compared. Zero-shot GPT-4.1 QWK 0.578 but a fine-tuned ConvBERT was better at 0.625 | medium. Also a caution: LLM direct estimates may lose to supervised encoders |
| Acquaye, Huang, Carpuat 2026, arXiv 2601.09953, Take Out Your Calculators (preprint) | yes, arXiv | abstract states LLMs are poor direct judges of item difficulty, while simulated classrooms fitted with IRT work better | medium. Warns that direct difficulty numbers are weak |

Closest existing work. Kolesnikova et al. 2026 (preprint) already gives LLMs items with known empirical difficulty and asks for a new item's difficulty, on primary-school math. Two other papers report direct LLM difficulty ratings are weak. No hit on agent or procedure tasks. Novelty risk: high for education items, low to medium for agent procedures. Expect weak signal and plan a supervised baseline next to it.

## Idea 4. FMEA performed by LLMs, and FMEA for AI agents

| Source | Verified | What it shows | Novelty risk |
|---|---|---|---|
| IEC 60812:2018, Failure modes and effects analysis (FMEA and FMECA) | yes, IEC webstore page title matched | the standard, severity times occurrence times detection scoring | none |
| Stamatis 1995, Failure Mode and Effect Analysis: FMEA from Theory to Execution, ASQC Quality Press | yes, OpenLibrary (ISBN 9780873893008, author and year match, exact title wording not checked) | textbook procedure | none |
| El Hassani and Masrour 2025, Design Science, AI-driven FMEA: integration of LLMs. doi 10.1017/dsj.2025.7 | yes, Crossref | LLMs generate FMEA tables for faster risk analysis | low |
| Hassani and Masrour 2024, Proc. Design Society, Integrating LLMs for improved FMEA. doi 10.1017/pds.2024.204 | yes, Crossref | framework and case study | low |
| Lin and Zhang 2026, Studies in Computational Intelligence, From Failure Modes to Reliability Awareness in Generative and Agentic AI Systems. doi 10.1007/978-3-032-18585-3_10 | yes, Crossref (content not read) | book chapter on failure modes of agentic systems | low to medium |
| Chat-of-Thought, arXiv 2506.10086 (preprint) | yes, arXiv | multi-agent LLM system generating FMEA documents for industrial assets | low |

Closest existing work. LLMs generating FMEA tables for industrial equipment is established (Design Science 2025). FMEA applied to AI agents appears only in trade or chapter-level material (a Gravitee web page, one book chapter). I found no paper that uses LLM-produced FMEA scores (severity, occurrence, detection) as predictors of agent success on procedures, and no validation of FMEA ratings against measured agent failure rates. Novelty risk: low, but the supporting evidence base is thin and mostly non-peer-reviewed for agents.

## Idea 5. Learning using privileged information, mixture IRT, two-part models

| Source | Verified | What it shows | Novelty risk |
|---|---|---|---|
| Vapnik and Vashist 2009, Neural Networks, A new learning paradigm: LUPI. doi 10.1016/j.neunet.2009.06.042 | yes, Crossref | extra features available only at training time can help the learned predictor | none |
| Lopez-Paz et al. 2016, ICLR, Unifying distillation and privileged information. arXiv 1511.03643 | arXiv yes (journal_ref says ICLR proceedings). No DOI | generalised distillation, teacher on privileged features, student on test-time features | none |
| Rost 1990, Applied Psychological Measurement, Rasch models in latent classes. doi 10.1177/014662169001400305 | yes, Crossref | mixture Rasch model, items can behave differently in latent classes | none |

Closest existing work. For our use, privileged information would be things known only after running agents (traces, response patterns, execution logs) used to train a teacher, with a student that sees only the documentation. I found no paper doing this for LLM or agent difficulty prediction, but my search here was brief (one arXiv query, which returned only generic LUPI papers). Mixture IRT for LLM benchmarks was not searched. I did not find a two-part model source for item difficulty, so I make no claim there. Novelty risk: low but unconfirmed.

## Idea 6. Multi-source training, within-domain standardisation, stacking

| Source | Verified | What it shows | Novelty risk |
|---|---|---|---|
| Wolpert 1992, Neural Networks, Stacked generalization. doi 10.1016/S0893-6080(05)80023-1 | yes, Crossref | stacking of base learners via a meta-learner | none |
| van der Laan, Polley, Hubbard 2007, Stat. Appl. Genet. Mol. Biol., Super Learner. doi 10.2202/1544-6115.1309 | yes, Crossref | cross-validated stacking is asymptotically as good as the best candidate | none |
| Krsteski and Meyer 2026, arXiv 2608.05797 (preprint) | yes, arXiv | the evaluation protocol we use, same cross-domain setting | context |

Closest existing work. I found nothing that applies multi-source training with per-domain standardisation or super learner stacking to LLM difficulty prediction, but this was a thin search (the arXiv keyword queries for cross-domain difficulty prediction returned unrelated papers). Our own pilot memory says embeddings do not transfer across benchmarks, so any support here is methodological only and does not show it will work. Novelty risk: unknown, probably low.

## Idea 7. Adaptive testing, calibration with few responses, efficient IRT evaluation, pilot runs

| Source | Verified | What it shows | Novelty risk |
|---|---|---|---|
| Polo et al. 2024, ICML, tinyBenchmarks. arXiv 2402.14992 | arXiv yes. ICML venue from Semantic Scholar only | about 100 curated items per benchmark recover full-benchmark scores using IRT | none |
| Kipnis et al. 2024, arXiv 2407.12844, metabench (preprint, venue unconfirmed) | yes, arXiv | sparse benchmark built with IRT and information, reconstructs full scores from about 3 percent of items | none |
| Zhuang et al., arXiv 2306.10512, Efficiently Measuring the Cognitive Ability of LLMs: An Adaptive Testing Perspective (latest arXiv version is retitled "Position: AI Evaluation Should Learn from How We Test Humans", preprint, ICML venue unconfirmed) | yes, arXiv | computerized adaptive testing for LLMs needs fewer items | none |
| Li et al. 2025, arXiv 2511.04689, Adaptive Testing for LLM Evaluation (preprint) | yes, arXiv | psychometric adaptive testing as an alternative to static benchmarks | none |
| Balkir et al. 2026, arXiv 2601.13885, Confident Rankings with Fewer Items (preprint) | yes, arXiv | adaptive LLM evaluation with continuous scores | none |
| Vivek et al. 2023, arXiv 2309.08638, Anchor Points (preprint, EMNLP version not checked) | yes, arXiv | a small set of anchor items predicts full benchmark performance | none |
| Perlitz et al. 2023, arXiv 2308.11696, Efficient Benchmarking of Language Models (preprint) | yes, arXiv | cost reduction by item selection | none |
| She et al. 2026, arXiv 2609.21267, Efficient Benchmarking in Production: A Study of an Evolving LLM Agent (preprint) | yes, arXiv | efficient benchmarking for agents (abstract not read in detail) | medium, read before claiming novelty |

Closest existing work. All of these select items to estimate model ability. Our use is the other direction: use a few pilot runs to calibrate the difficulty of new items, which is closer to cold-start item calibration. She et al. 2026 and Balkir et al. 2026 should be read in full before the round claims anything about agent benchmark efficiency. Novelty risk: medium for "few runs to calibrate new procedures", low for the rest.

## Idea 8. Question difficulty estimation from text

| Source | Verified | What it shows | Novelty risk |
|---|---|---|---|
| Benedetto et al. 2023, ACM Computing Surveys, A Survey on Recent Approaches to Question Difficulty Estimation from Text. doi 10.1145/3556538 | yes, Crossref | survey, text features and neural models predict item difficulty | none |
| Yaneva et al. 2024, BEA workshop, Findings from the First Shared Task on Automated Prediction of Difficulty and Response Time for Multiple-Choice Questions. ACL Anthology 2024.bea-1.39 | yes, ACL bib (title and first authors Yaneva, North, Baldwin). Crossref has no record | shared task on USMLE items, many teams used LLMs and encoders | none |
| Razavi and Powers 2025, arXiv 2504.08804 (preprint) | yes, arXiv | LLM-extracted features plus trees for item difficulty | low |
| Zhu et al. 2025, arXiv 2509.12886, The LLM Already Knows (preprint) | yes, arXiv | difficulty as perceived by the LLM from hidden states, no output tokens | medium for the agent setting |
| Hoyl 2026, arXiv 2602.00034, Synthetic Student Responses (preprint) | yes, arXiv | LLM-extracted features such as step count and misconceptions for IRT difficulty | low |
| Scarlatos et al. 2025, arXiv 2507.05129, SMART (preprint) | yes, arXiv | simulated students aligned with IRT for difficulty prediction | low |

Closest existing work. A mature literature on human item difficulty from text exists. Closest to our setting are Zhu 2025 (LLM-perceived difficulty) and Krsteski 2026 (agentic benchmarks). I found no paper doing text-based difficulty estimation for policy or procedure compliance by agents. Novelty risk: low for our domain, but reviewers will expect these citations.

## Idea 9. Metrics: selective prediction, AURC, power, decision-focused evaluation

| Source | Verified | What it shows | Novelty risk |
|---|---|---|---|
| Geifman and El-Yaniv 2017, NeurIPS, Selective Classification for Deep Neural Networks. arXiv 1705.08500 | arXiv yes. No DOI. NeurIPS venue from Semantic Scholar | risk-coverage trade-off, the base of AURC | none |
| Card et al. 2020, EMNLP, With Little Power Comes Great Responsibility. doi 10.18653/v1/2020.emnlp-main.745 | yes, Crossref | NLP comparisons are often underpowered, power analysis and minimum detectable effect | none |
| Miller 2024, arXiv 2411.00640, Adding Error Bars to Evals (preprint) | yes, arXiv | standard errors and paired comparisons for LLM evals | none |
| Krsteski and Meyer 2026, arXiv 2608.05797 (preprint) | yes, arXiv | argues AUC can mislead for difficulty prediction | context |

Closest existing work. Selective prediction and power analysis are well established and I found no paper applying risk-coverage curves to ex ante agent difficulty prediction, although the search was short. I did not find a verified source for "decision-focused evaluation" and leave it unsupported. Novelty risk: low, mostly a rigor argument.

## Summary of novelty threats (strongest first)

1. Idea 2 generic method: Ballon et al. 2025 (arXiv 2512.14220) already does LLM pairwise difficulty plus Bradley-Terry. Only the agent procedure application and the test against fitted agent IRT difficulty remain.
2. Idea 3 for education items: Kolesnikova et al. 2026 (arXiv 2605.18562), plus evidence that direct LLM difficulty ratings are often weaker than supervised models (Wang 2026, Acquaye 2026).
3. Idea 7: She et al. 2026 (arXiv 2609.21267) unread in detail.
4. Ideas 1, 4, 5, 6: no direct hit found, but searches for 5 and 6 were thin. Treat "not found" as weak evidence.
