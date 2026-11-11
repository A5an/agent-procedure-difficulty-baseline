# Diagnostic test: what information makes success predictable. Result (11 Nov 2026)

Plan `PREREG.md`, code `diag.py`, predictions `preds.jsonl`, output `diag.out`. SOPBench, 66 procedures, 242 test
cases, four agents (GPT-5, Gemini 2.5 Flash, Claude 3.7 Sonnet, GPT-4o mini; mean success on test 0.631).
Predictor: Claude Haiku through `claude -p`, 344 calls on the subscription, $0.

| Input of the predictor | AUC | Brier skill vs BaseRate | Success in the top half |
|---|---|---|---|
| BaseRate (agent's average, no LLM) | 0.500 | 0 | 0.631 (= mean rate) |
| V0 Vanilla: case text only | 0.509 [0.436, 0.577] | -0.082 | 0.636 |
| V1 +Category: domain and procedure names | 0.504 [0.446, 0.565] | -0.105 | 0.630 |
| V2 +Stats of similar procedures | 0.590 [0.530, 0.651] | -0.027 | 0.684 |
| DomainRate (same stats, no LLM) | 0.590 [0.518, 0.658] | +0.010 | 0.684 |
| V3 +Stats of this procedure | 0.629 [0.571, 0.684] | -0.052 | 0.723 |
| ProcRate (same stats, no LLM) | 0.661 [0.608, 0.713] | +0.052 | 0.742 |
| Oracle | 1.000 | 1 | 0.971 |

("Success in the top half": give the agent the half of the cases with the highest prediction and count how many
it solves. The script prints 0.583 for BaseRate because a constant forecast is tie-broken by file order; the right
reference is the mean rate 0.631.)

## Reading

1. The paper's picture repeats. With the text alone the predictor is at chance (AUC 0.509), and adding the category
   labels changes nothing (0.504; Brier skill difference -0.023 [-0.070, +0.023]). In the paper: 41.41 vs 41.18.
2. Ranking improves only when outcome data enters the prompt: similar procedures 0.590, the procedure itself 0.629.
   In the paper: 47.74 with performance statistics.
3. The LLM adds nothing to the statistics. Counting without any LLM is as good (DomainRate 0.590) or better
   (ProcRate 0.661; Brier skill of V3 minus ProcRate -0.104 [-0.151, -0.064]). In the paper DimensionBest (47.50)
   equals the LLM with statistics (47.74).
4. The LLM's probabilities are badly calibrated: every LLM variant has negative Brier skill, even V3.

So on SOPBench the information about agent success is in observed outcomes, not in the text, and a strong reader of
the text does not change that.

## Limits

- One predictor (Haiku), one benchmark, 242 test cases; intervals are wide, V2 vs V1 on Brier skill is
  +0.078 [-0.006, +0.156].
- AUC intervals of single variants are given, paired AUC differences were not bootstrapped.
- History cases of a procedure are real cases from the same template (best possible probes), and there are only
  1-17 of them per procedure, so ProcRate is noisy.
- The four agents are scored in one call; the predictor may spread its answers across agents instead of judging each
  one independently.
- The predictor may know these models by name; this can only help V0, and V0 is still at chance.
- 10 calls failed on the subscription session limit and were rerun with the same script.
