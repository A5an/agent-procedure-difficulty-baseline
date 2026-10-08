# PREREG ablation of the pre-mortem (round 10, 8 Oct 2026, before any call)
Questions for the paper: (1) does the pre-mortem framing matter, or is it enough to ask Sonnet for p_success directly?
(2) how reliable is one pre-mortem call (test-retest)?
Calls (Sonnet 5.5, claude -p, no tools), on tau2 + banking (375 prompts) and TAC + MCPMark (302 prompts):
- D: direct rating. Same inputs and the same description of the agent generation and grading as P1, but no failure
  analysis: "Estimate the probability that a typical agent of that generation fully succeeds on this case/task.
  Return only {"p_success": ...}".
- R: retest of P1 with an appended line "(Independent second assessment.)" so the cache key differs.
Tests: raw rho of D vs P1 (paired bootstrap) on each benchmark; test-retest Spearman of P1 vs R; mean of P1 and R vs P1.
Hypothesis: P1 > D. No prompt changes after the first look.

## Added after the ablation result (D >= P1 on tau2), before any SOPBench D call
- Run D on all 830 SOPBench cases and on the 240 DrafterBench cases (same prompt builder).
- Report D under the exact protocol cells on all four scenarios (zero-shot, nothing fitted).
- One harness run: f6 = f2 + D (one feature, fc_diff of D) vs f2; f7 = c_lad + GAP + D vs g1. Primary: f6 > f2.
