# Literature scan: pre-attempt difficulty prediction in neighbouring fields (agent `lit6`, round 6, 6 Oct 2026)

Saved by the main session from the agent's final message. Abstracts fetched unless marked. No peer-reviewed work on
pre-attempt difficulty for agent tasks found, nearly all agent-side work is 2026 preprints or workshops.

Key papers: 2608.05797 Krsteski and Meyer (COLM 2026 workshop, entropy of a reasoning trace scored by open base models,
rho 0.399 K-fold, 0.225 leave-one-benchmark-out); 2608.18280 (static features of the gold patch, AUC 0.863, oracle-like);
2606.14530 (linear probe on hidden state predicts own code correctness, AUC 0.88, 444 tasks); 2610.05572 (pooled AUROC is
cross-task, within-task near chance: use within-domain metrics); 2602.11619 (action-sequence diversity across repeated runs,
AUROC 0.62 to 0.78, needs runs); 2604.22750 (agents predict own token cost at r <= 0.39); 2606.28186 Epi2Diff (episodes of
reasoning traces predict human item difficulty); 2507.05129 SMART (simulated students aligned with IRT via DPO);
2512.18880 (prompted proficiency does not make LLMs behave weaker); 2504.08804 (LLM-extracted features + trees beat direct
rating, r up to 0.87); 2605.18562 (pairwise > absolute); 2407.05327 (model uncertainty weak proxy for item difficulty);
BEA 2024 shared task (UnibucLLM 2404.13343, ITEC 2024.bea-1.43: text-only difficulty is hard, simple models win);
2607.28634; 2602.04577; 2605.00238; 2605.06334 MANTRA; Hasic and Vanthienen 2019 DMN complexity metrics (snippet only);
Mendling 2008 process model metrics (snippet only).

Top transferable ideas: (1) entropy profile of a think-aloud trace scored by open models (L1; note we tried Krsteski
entropy on 28 Sep with little effect); (2) partial evaluation of the compiled policy with facts stated in the customer
message plus environment-level marginals for unknown fields, giving P(perform | message) (L1, marginals between L0 and L2);
(3) diversity of k sampled dry-run plans (L1); (4) a real weak-to-strong agent ladder on synthetic cases (E); (5) NLI
contradiction between policy conditions and message claims (L1). Cautions: never report pooled AUC, report new_domain
separately, single-LLM difficulty judgement from text is the weakest family.
