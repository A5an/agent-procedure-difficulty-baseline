# Literature check: pre-write failure monitor (A) and progressive autonomy (B)

Date: 2026-11-03. Scope: only what novelty-report.md and screen/mf/LIT.md do not cover. Every arXiv id below was fetched from export.arxiv.org in this session (title and authors matched). "Full" = pdftotext of the PDF, relevant sections read, not cover to cover. "Abs" = abstract only. All 2026 items are preprints unless a venue is named. Nothing under data/external/handbook was opened.

## A. Learned runtime failure monitor

### Closest papers

| Paper | What they did, numbers | Read |
|---|---|---|
| PrefixGuard, Huang et al., 2026, 2605.06455 | GRU on TF-IDF step features predicts failure from prefixes, per benchmark: WebArena, tau2-bench, SkillsBench, TerminalBench. Beats log-anomaly baselines by 12.0-21.9 pp AUPRC; recall 56-93% at 10% false-alarm budget. Zero-shot DeepSeek-V4-Pro judge at H=3: AUPRC .45/.40/.08/.11. Random split with same tasks in train and test; paper itself says "performance on unseen tasks and agents remains unmeasured". | Full (key parts) |
| TRACER, Tayebati et al., 2026, 2602.11409 | Training-free trajectory uncertainty (needs logprobs) on tau2-bench airline/retail/telecom, 3 Gemini/GPT models. Pooled AUROC 0.67-0.81 vs 0.43-0.69 for entropy baselines. | Full (results) |
| Automata from Agent Traces, Cho et al., 2026, 2608.23670 | FSM per corpus, per-state behavioral features, 12 datasets. Held-out AUROC up to 0.94. Cross-model on tau2-bench 0.786 vs 0.877 same-model. Cross-dataset transfer only between two SWE corpora (FSM features: 0.715 and 0.648 cross vs 0.681 and 0.780 in-dataset). | Full (key parts) |
| From Confident Closing to Silent Failure, Advani, 2026, 2606.09863 (single author) | Detects "false success" (agent claims done, state says no) on 9,876 tau2-bench trajectories. TF-IDF on closing message: task-disjoint AUROC 0.83, leave-one-model-out 0.91, **leave-one-domain-out airline to retail 0.66-0.70**. Five LLM judges never above 0.65; 0.54 on AppWorld. | Full (key parts) |
| Disentangling Task Difficulty from Run-Level Failure, EsfandyariDoulabi et al., 2610.05572 | Pooled AUROC 0.85-0.94 is mostly task difficulty (>99.93% of pairs cross-task; difficulty oracle 0.945 pooled, 0.50 within task). Early within-task AUROC 0.50-0.55 for trajectory predictors, released monitors, probes; checked on tau2-bench too. Early stopping pays only above within-task AUROC about 0.84-0.93. | Full (key parts) |
| Continue or Restart?, Srinivasan et al., 2610.35971 | 147,002 SWE traces, fine-tuned 0.6B monitor: within-task AUC 0.50 [0.48, 0.51] at 5k-20k tokens. | Abs |
| PolicyGuardBench, Wen et al., 2510.03485 (not the PolicyGuard of 2606.29225 in the report) | 60k policy-trajectory pairs (web agents), 4B guard, prefix detection. Leave-one-domain-out accuracy 0.908 vs 0.933 in-domain. Synthetic traces, accuracy not AUROC. | Full (LODO table) |
| StopLoss 2610.25525; FailFast 2608.03222; EarlyEval 2609.02783; Doomed from the Start 2607.06503; OnTrack 2610.12375 | Small learned monitors on SWE/WebShop/TextCraft. StopLoss beats six LLM judges at 1-2 orders lower cost. FailFast transfers across policies (SWE-bench only), saves 14.6-20.4% tokens. EarlyEval LightGBM on behavioral features, 13-26% fewer steps. | Abs |
| Uncertainty for agents: 2609.07395 (UQ taxonomy, TC-ECE), 2608.29685 (verbal confidence AUROC 0.85 at completion, below 0.60 at 50% progress), 2602.11619 (run-to-run consistency AUROC 0.62-0.78, HotpotQA) | Own-confidence signals weak before the end. | Abs |
| Process reward models for agents: AgentPRM 2511.08325, ToolPRMBench 2601.12294, Choudhury 2502.10325, WebArbiter 2601.21872 | Step-level rewards for search or RL guidance. Not calibrated outcome prediction, not tested across benchmarks. | Abs |
| Calibration transfer, Cao, 2609.25647 | Leave-one-agent-out on SWE-bench: two target agents miscalibrated. | Abs |
| Argument provenance PACT 2605.11039; NOMOS 2610.11030 and SCOPE 2610.31062 | Rule-based or compiled gates on tool arguments (security, policy). Not learned failure prediction. | Abs |
| Decision value of critics, Vasudev et al., 2602.03338 | Critic AUROC 0.94 can still collapse success by 26 pp; 50-task pilot predicts harm. Already in the report as a rule; the pilot test is new. | Abs |

### What is done, what is not

- Done: predicting episode failure from trajectory prefixes with learned classifiers, including on tau2-bench (PrefixGuard, TRACER, Cho, Advani). Trained small monitors beating LLM judges (Advani, PrefixGuard, StopLoss). Behavioral or structural features (Cho, EarlyEval). Cross-model transfer (Cho 0.786 vs 0.877; Advani 0.91; FailFast).
- Not found: a monitor evaluated by training on one benchmark and testing on another benchmark. Closest are airline to retail inside tau2 (0.66-0.70, Advani), between two SWE corpora (Cho), and web-domain LODO (PolicyGuardBench). I found no failure-prediction paper on SOPBench trajectories (arXiv and web search).
- Not found: decision point fixed at the first state-changing call (or final message if none), with features such as "arguments grounded in earlier tool outputs or user turns" and "explicit confirmation before write". Advani's closing-message detector is the nearest (final message, no write).
- Not found: a two-model disagreement signal as a failure predictor compared with a trained classifier. Adjacent only: ReDAct 2604.07036 (uncertainty deferral, ALFWorld), PolicyGuard in the report, run-to-run consistency 2602.11619, Rao and Callison-Burch in the report (judge errors correlated).
- Main trap, now published twice: pooled AUROC across tasks mostly measures task difficulty. Early within-task AUROC is near 0.5 (2610.05572, 2610.35971). The write call is late, so signal may be stronger there, but that is my inference, not a result in these papers. With pooled AUROC, a "procedure difficulty" feature alone will look like a good monitor.
- Not found, could not locate by title: ActAsk, Verifier-Grounded Runtime Mediation, Rejectors in the Wild (flagged in the report). Treat as unverified.

### Verdict A: partially done

The generic idea (learned prefix monitor, beats LLM judge, works on tau2-bench) is done and must not be claimed. The cross-benchmark SOP-style test with grounding/confirmation features at the write boundary was not found.

Safe phrasing: "Prior learned failure monitors are trained and tested within one benchmark, or transfer across models or domains of one benchmark family (PrefixGuard; Cho et al.; Advani). To our knowledge, none trains on trajectories from one customer-service or SOP benchmark and tests on another, nor fixes the decision point at the first state-changing call with grounding and confirmation features. We report within-task AUROC, following EsfandyariDoulabi et al."

Do not claim: first failure predictor on tau2 or SOPBench; first to use structural features; that trained classifiers beat LLM judges as a new finding.

Design consequences: group splits by task and procedure; report within-task AUROC and AUROC against a difficulty-only baseline; add leave-one-domain-out inside tau2 as the Advani reference (0.66-0.70); calibration on the target benchmark (2609.25647 shows target-specific miscalibration).

## B. Progressive autonomy / shadow mode

### Closest papers

| Paper | What they did | Read |
|---|---|---|
| READY, 2609.02095 (in report) | New detail from the text: qualification uses a fixed-sample one-sided Clopper-Pearson bound on held-out cases (375 in the case study). I found no mention of shadow mode, sequential or anytime-valid bounds, or priors. | Full (grep of key terms) |
| PP-LUCB, Ao et al., 2026, 2603.10400 | Biased LLM proxy on every case plus selectively bought human audits, inverse-propensity residuals, anytime-valid confidence sequences. Best configuration of a service system on customer-support tickets: 40/40 correct, 90% audit cost cut. Selects among configurations, does not decide human vs agent hand-off. | Abs |
| Conformal Selective Acting, 2605.20270 | Per-deployment e-process, anytime-valid selective risk, release-rate guarantee. LLM answers, not agent-vs-human. | Abs |
| E-valuator, 2512.03109 | E-process sequential test turning verifier scores into step-wise valid decisions, 6 datasets. | Abs |
| AgentAssay, 2603.02601 | SPRT for regression testing of agent workflows, 78% fewer trials, 7,605 trials. Release gate against a prior version, no human baseline. | Abs |
| Act-or-defer with local reliability bounds, 2606.29654 | kNN lower confidence bound on correctness, acts only above threshold, up to 84% automation at 96% accuracy. Per decision, not per procedure. | Abs |
| Parloa Labs, "A Bayesian framework for A/B testing AI agents", 2025-11-01 (blog, not peer reviewed; page fetched) | Hierarchical Beta-Binomial with partial pooling over scenarios, simulated conversations (about 30 scenarios x 10 runs). No human baseline, no simulation-derived prior, no shadow stage. | Page read |
| Progressive Autonomy as Preference Learning, 2605.19151; Levels of Autonomy, 2506.12469 | Formalizations of when an action needs approval. No outcome comparison against a human, no sample sizes. | Abs |
| Classical statistics, no fetch needed beyond ids | Power prior (Ibrahim and Chen, Stat. Sci. 2000, DOI 10.1214/ss/1009212673, Crossref-checked); robust meta-analytic-predictive priors (Schmidli et al., Biometrics 2014, DOI 10.1111/biom.12242, Crossref-checked); confidence sequences (Howard et al. 1810.08240; Waudby-Smith and Ramdas 2010.09686); Learn then Test 2110.01052; Conformal Risk Control 2208.02814; Learning to Defer 1711.06664. | Metadata only |
| Practice | Clinical "silent trials" (Kwong et al. 2022, Frontiers in Digital Health, via search snippet) and the VALIDATE shadow-mode protocol (medRxiv, 2026, no per-site sample size found in retrieved text). Practitioner shadow-mode guides give heuristic gates such as "90% agreement over 30 consecutive cases" (one web doc, non-peer-reviewed). A journal paper on SPRT-gated migration of analytical agents (jisem-journal.com) was seen only in a snippet, no arXiv version found: unverified. | Snippets |

### What is done, what is not

- Done in pieces: fixed-sample lower-bound qualification of an agent plus human oversight (READY); anytime-valid and sequential tests for agent regression and proxy-plus-audit selection (AgentAssay, PP-LUCB, CSA, E-valuator); hierarchical Beta-Binomial pooling for agent A/B tests (Parloa); prior borrowing in clinical trials (power and MAP priors).
- Not found: shadow-mode comparison of agent against human on the same cases, with a hierarchical model across procedures, a simulation-probe prior, and a promotion rule based on a bound on expected loss versus the human.
- Not found: any answer to "how many shadow cases per procedure", with or without a probe prior. Nothing in the sources above reports it.
- READY is fixed-sample and per process; the sequential, multi-procedure, prior-informed version is a gap, but one an adjacent paper can close quickly. Rank it as exposed.

My own arithmetic, not from a source: with zero failures in n shadow cases, the 95% exact upper bound is 1 - 0.05^(1/n), so n = 29 certifies a failure rate at or below 10%, n = 59 at or below 5%, n = 299 at or below 1%. Beating a human on expected loss needs paired outcomes and more cases when the gap is small. A Beta prior of strength m is worth about m cases, so 12 probes can save at most about 12, and less if their bias or flat response (rho 0.09 for weak agents) forces discounting. The prior matters most when the loss gap is large and failure rates are moderate, not when certifying rare failures.

### Verdict B: not found as a whole, partially done in parts

Safe phrasing: "Fixed-sample qualification of agents on held-out cases exists (READY), as do anytime-valid tests for agent regression and proxy-plus-audit selection (AgentAssay, PP-LUCB, CSA). To our knowledge, no prior work combines a hierarchical Bayesian model of per-procedure loss, a synthetic-probe prior, and shadow-mode outcomes against a human baseline to give the number of shadow cases needed before hand-over, with and without the prior."

Do not claim: first statistical gate for deploying agents; first hierarchical Bayesian treatment of agent evaluation (Parloa, HiBayES); first anytime-valid agent test.

Baselines worth including: READY-style Clopper-Pearson with no prior; power prior with discount parameter; robust MAP mixture prior (guards against bad probes); anytime-valid betting confidence sequence for the stopping rule.
