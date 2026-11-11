# Memory for a new procedure (plan, written 11 Nov 2026 before any run)

Idea from Agent-as-a-Router (arXiv:2606.22902): a Memory keyed by embeddings stores observed outcomes and a new item
is scored from its nearest stored neighbours. Our question: can outcomes of already checked procedures replace or
reduce the probes of a new procedure?

Data: SOPBench, 830 cases, 70 procedures, 7 domains, 28 agents (`data/sopbench`), embeddings `bge_base_en_v1.5`.
No LLM calls. Targets: the 28 procedures with >= 8 cases. For each target, n probe cases (n = 0, 1, 2, 4) are drawn at
random and the remaining cases are the test; 30 draws. A probe here is a real held-out case of the procedure (the
best possible probe).

Methods, each gives P(success) for an agent on the target procedure:
- Prior: mean success rate of the agent over the stored procedures.
- ProcMemory: kNN over procedure embeddings (mean of case embeddings); similarity-weighted mean of the stored
  procedures' success rates of the agent. k in {3,5,10,all}, temperature in {0.02,0.05,0.1,inf}, chosen by
  leave-one-procedure-out inside the memory (the target is never used).
- CaseMemory (the paper's form): kNN over case embeddings; memory = cases of stored procedures plus the n probe
  cases; prediction per test case. k in {5,10,20}, temperature in {0.02,0.05}, chosen inside the memory.
- Probes with a prior: Beta(m*mu, m*(1-mu)) updated by the n probes, mu = Prior or ProcMemory, m in {2,4,8}.

Memory content: (a) in-domain: all other procedures; (b) new domain: only procedures of the other six domains.

Metrics: Brier on test cases and Brier skill against Prior with n = 0; decision regret for "give to the agent iff
P >= tau", utility = rate - tau, tau = 0.5 and 0.7. Bootstrap over procedures (2000 resamples), paired differences.

Questions fixed in advance:
1. n = 0: does Memory beat Prior, in-domain and on a new domain?
2. Does a Memory prior reduce the number of probes needed for the same Brier?
3. Stream: procedures arrive in random order, memory starts empty and stores each finished procedure with its
   outcomes (full feedback assumed). How fast does Memory become useful?
