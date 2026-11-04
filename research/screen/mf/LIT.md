# Literature check: multi-fidelity Bayesian fusion for per-procedure agent reliability

Date: 2026-11-02. Scope: only what novelty-report.md did not cover (it already has IntellAgent, Nubank, Lost in Simulation, Mind the Sim2Real, CUE, READY, Risk-Aware Adaptive Evaluation 2609.38914, Singh 2609.09245, Krsteski-Meyer, RouteGuard, Bayesian self-escalation 2608.24087; not repeated).
Verification: every arXiv id below was fetched from export.arxiv.org in this session (title, authors, abstract matched). "Full" = I read the PDF text (pdftotext; key sections, not cover to cover). "Abs" = abstract only. Nothing under data/external/handbook was opened.

## Q1. Synthetic/simulated outcomes + few real outcomes, statistically principled

**Prediction-powered inference (PPI) family.** Core idea: cheap biased predictions on many units, plus a "rectifier" estimated on a few labelled units, gives valid CIs.
- Angelopoulos et al., *Prediction-Powered Inference*, 2023, arXiv 2301.09633 (abs). PPI++ 2311.01453 (title only).
- Boyeau, Angelopoulos, Yosef et al., *AutoEval Done Right: Using Synthetic Data for Model Evaluation*, 2024, arXiv 2403.07008 (abs). Synthetic (AI-labelled) data cuts human labels needed; "up to 50%" larger effective sample on GPT-4 experiments. Mean-level only.
- Fisch et al., *Stratified PPI for Hybrid Language Model Evaluation*, 2024, arXiv 2406.04291 (abs). Stratified sampling improves PPI intervals.
- Hofer et al., *Bayesian Prediction-Powered Inference*, 2024, arXiv 2405.06034 (abs). Bayesian PPI for LLM autoraters, discrete and nonlinear score-to-human relationships. Closest Bayesian PPI, but one parameter, not hierarchical per unit.
- Zrnic & Candès, *Active Statistical Inference*, 2024, arXiv 2403.03208 (abs); Ma & Candès, *OPAL*, 2026, arXiv 2606.03211 (abs). Choose which units to label by model uncertainty.
- **Emmenegger, Stahler, Podimata (MIT), *PPI Across Many Tasks for AI Evaluation*, 2026-05, arXiv 2605.29249 (full, preprint).** Many related tasks, only 20-40 labels per task, cheap LLM proxy (gpt-5-mini) vs human labels, borrows strength across tasks via cross-task recalibration of proxy to truth (isotonic), keeps per-task rectification and per-task CIs. Proves affine recalibration gives no gain over power-tuned PPI; gains need nonlinear proxy-to-truth structure. Case study: auditing LLMs on 2024 US election information. Not agents, no cross-agent term, no decision rule. Structurally the nearest statistical relative.
- Li & Ignatiadis, *Prediction-Powered Adaptive Shrinkage*, 2025, arXiv 2502.14166 (abs; ICML 2025 per the MIT paper). Empirical-Bayes shrinkage of many means using PPI predictions.
- Cowen-Breen et al., *MultiPPI*, 2026, arXiv 2603.27414 (abs); Brawand et al., *Active Multiple-PPI*, 2026, arXiv 2605.08429 (abs). Several predictors of different cost, budget-adaptive allocation. Ours has the same shape (text prior, probes, other agents are different-cost predictors) but they target one population mean.
- Ao et al., *Best Arm Identification with LLM Judges and Limited Human*, 2026, arXiv 2601.21471 (abs). Biased arm- and context-dependent proxy, selectively audited ground truth, inverse-propensity-weighted residuals. States that standard multi-fidelity methods can mis-select when the proxy bias is not corrected. Frequentist.

**PPI applied to agents.** *GLIDE*, Martinon, Merad, Raki, 2026, arXiv 2605.31278 (full, preprint, workshop paper). Library of PPI++/stratified/active estimators. Agent case study: R-Judge, 568 agent trajectories, LLM-judge predicts "safety risk", expert labels are truth, judge bias about 13 points, mean estimation only. Its own limitations: single proxy, no multi-proxy aggregation, and for stochastic agents "the open question is how to allocate budget between input coverage and output replication". So the agent-reliability version is explicitly listed as open by these authors.

**Multi-fidelity / co-kriging.** Kennedy & O'Hagan, *Predicting the output from a complex computer code when fast approximations are available*, Biometrika 87(1):1-13, 2000, DOI 10.1093/biomet/87.1.1 (metadata verified via Crossref, not read here; the autoregressive model is from memory). I searched arXiv for Kennedy/O'Hagan or co-kriging with LLM/agent evaluation and found nothing. Only adjacent: Lu et al., *Adaptive Multi-Fidelity Bandits with Improving Proxies*, 2026, arXiv 2605.08558 (abs; bandits, LLM as low-fidelity source); Hübel et al., *BT-Agent*, 2026, arXiv 2610.13862 (abs; "multi-fidelity benchmark" = text vs Habitat simulation, no statistical fusion).

**Sim-to-real.** Already in the report. New: RealUserSim 2605.20204 (abs) and Chen, *Simulated Customers Never Walk Away*, 2606.20708 (abs) improve or measure the simulator, they do not fuse simulated and real outcomes.

**Bayesian evaluation with few labels.** Luettgau et al., *HiBayES*, 2025, arXiv 2505.05602 (abs): hierarchical GLM framework for AI evals, covers agentic evals, "fewer than 20 data points". Gao et al., *Bayesian Calibration of Win Rate Estimation with LLM Evaluators*, 2024, arXiv 2411.04424 (title only). AISTATS 2026 "Maven" poster, hierarchical Bayesian fusion of expert and crowd labels: seen only in a search snippet, **unverified**.

**Answer Q1.** Principled proxy-plus-few-labels fusion exists (PPI family, Bayesian PPI, multi-task PPI) and has been applied to LLM judges and once to agent trajectories (GLIDE). I found no work that applies it, or Kennedy-O'Hagan, to simulated test outcomes versus real outcomes for per-task or per-procedure agent success.

My reasoning, not from a source: PPI needs paired (prediction, truth) on the same unit. Our probes are different cases from the real cases, so the right frame is a procedure-level latent theta_j with probe counts ~ Bin(12, g(theta_j)) and real counts ~ Bin(n_j, theta_j), g monotone and biased. That is Kennedy-O'Hagan-style, not literal PPI. The MIT result says g must be nonlinear to give gains over the plain estimate.

## Q2. How many probes / pilots, adaptive allocation

- Polo et al., *tinyBenchmarks*, 2024, arXiv 2402.14992 (abs): about 100 curated MMLU items reproduce the full score (IRT-based). Same group: PromptEval 2405.17202 (abs).
- Hofmann et al., *Fluid Benchmarking*, 2025, arXiv 2509.11106 (abs): IRT item selection adapted to each LM's ability.
- Li et al., *Active Evaluation Acquisition*, 2024, arXiv 2410.05952 (abs); Kossen et al., *Active Testing*, 2021, arXiv 2103.05331 (abs).
- Zheng et al., CAT for medical LLM benchmarks, 2026, arXiv 2603.23506 (abs).
- She & Lin, *Efficient Benchmarking in Production: Evolving LLM Agent*, 2026, arXiv 2609.21267 (abs): 574 historical runs, multidimensional 2PL adaptive testing, 200 questions (38.5% of full run) gives 1.03 pp MAE. Agent-specific, but it estimates an agent's overall score, not per-deployment-unit reliability.
- **Shen et al., *Speculative Evaluation of Stochastic LLMs*, 2026, arXiv 2609.28560 (full, preprint).** Uniform pilot per task, hierarchical Beta model pools per-task success counts, then exact integer Neyman allocation of the remaining rollouts; 107 benchmark-checkpoint profiles, budgets 8-64 rollouts per task. Target is the benchmark mean variance, no proxy, no decision cost.
- Pilditch, *Knowing When to Stop: Bayesian Optimal Stopping for LLM Evaluations*, 2026, arXiv 2608.14425 (abs): hierarchical Bayesian precision-based stopping, 57-97% of trials removed in illustrative runs.

**Answer Q2.** Allocation across items to estimate one model's aggregate is well covered. What I did not find: allocation per deployment unit with a decision loss (misrouting cost, expected value of sample information), or a probes-versus-pilots trade-off with two sources of different cost and bias. The probe-count curve (rho vs number of probes) is still not in the literature, consistent with the report.

## Q3. Hierarchical / beta-binomial for pass^k

- Hariri et al., *Don't Pass@k: A Bayesian Framework for LLM Evaluation*, 2025, arXiv 2510.04265 (abs): Dirichlet posterior of a model's success probability replaces pass@k/avg@N, credible intervals. Per model, not hierarchical over tasks.
- Shen et al. 2609.28560 (above): p_i ~ Beta(xi*kappa, (1-xi)*kappa), the same hierarchy we want.
- Manousaridis et al. (Pfizer R&D), *Characterizing LLM performance via a Bayesian lens*, Front. Appl. Math. Stat., 2026-07-31, DOI 10.3389/fams.2026.1854495 (Crossref + web page, first 100k of 114k characters read): per-question Beta, binomial over repeated trials, partial pooling, NUTS; 100 questions x 100 trials, 11 LLMs, multiple choice. No pass^k, no agents.
- Aghazadeh-Chakherlou et al., *HIP-LLM*, 2025, arXiv 2511.00527 (abs): hierarchical imprecise-probability reliability over operational profile, subdomain to system level.
- Morandi & Viswanathan, 2026, arXiv 2605.12177 (abs): hierarchical Beta-Binomial for selection bias in thumbs feedback; Khanal et al. 2603.29231 (abs) and Tran-Truong & Le 2604.24579 (abs) give reliability-decay and Markov-chain views of pass^k.
- Parloa "Bayesian framework for A/B testing AI agents" (hierarchical Beta-Binomial): X post, not opened, **unverified**.

Under a Beta(a,b) posterior for theta, pass^k = E[theta^k] = B(a+k,b)/B(a,b), closed form (standard math, my derivation, not a citation). The Singh et al. identifiability limit (in the report) means k > n is a prior-driven extrapolation; state it.

**Answer Q3.** Hierarchical Beta-Binomial over tasks is established for LLM accuracy (Manousaridis, Shen). Nobody I found combines it with pass^k on agent procedures plus an external-fidelity covariate.

## Q4. Closest prior work and how ours differs

No single paper does it. Closest three, in order:
1. **Multi-task PPI (2605.29249):** few labels per task, cheap proxy, borrowing across related tasks, per-task intervals.
2. **HBN (2609.28560):** hierarchical Beta-Binomial per task, pilot, budget allocation.
3. **GLIDE (2605.31278):** PPI on agent trajectories, explicitly lists multi-proxy and stochastic-agent allocation as open.

Where ours differs (each is checkable):
- Proxy = LLM-written probes run through the actual agent, not an LLM judge on the same item; unit-level, unpaired.
- Target is per-procedure theta_j for a specific agent, not a mean over a benchmark.
- Cross-agent IRT-style term (agent x procedure) inside the same posterior; PPI papers have one model or one proxy.
- Outputs P(success) and pass^k via Beta-Binomial, scored by calibration (Brier/ECE), not CI coverage only.
- Value-of-information rule tied to the routing decision (agent / agent+verifier / human), not variance of a mean.
- Honest limit: PPI gives distribution-free validity, ours is model-based, so validity needs held-out procedures and calibration checks. Weak-agent failure (rho 0.09) means g must be allowed to be flat.

## Verdict

1. Components exist: PPI with few labels, multi-task borrowing, hierarchical Beta-Binomial with pilot allocation, Bayesian pass@k. The fusion as a whole, applied to agent procedures, I did not find.
2. Status: **partially done** (parts done, the combination for agent reliability and routing is new in my search; coverage of arXiv search was limited, and Kennedy-O'Hagan was searched only on arXiv).
3. Do not claim "first multi-fidelity or PPI approach to agent evaluation" (GLIDE applies PPI to agent trajectories; multi-task PPI covers many-task AI evaluation).
4. Safe phrasing: "To our knowledge, no prior work fuses LLM-generated probe outcomes, a few real pilot runs and cross-agent outcomes in one hierarchical Bayesian model to give calibrated per-procedure P(success) and pass^k, and uses its posterior to decide how many probes and pilots to buy for routing."
5. Cite and compare against: multi-task PPI 2605.29249, GLIDE 2605.31278, HBN 2609.28560, Bayesian PPI 2405.06034, Kennedy and O'Hagan 2000; include a plain-mean baseline and a PPI/StratPPI baseline in experiments.
