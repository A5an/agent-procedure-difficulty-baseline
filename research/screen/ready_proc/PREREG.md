# READY-style qualification at the procedure level + matched-budget baselines: plan (11 Nov 2026)

Written before the script was run. Recorded data only, no LLM calls, $0.

## Question

READY (2609.02095) qualifies an agent on cases: a dev set fixes the accept rule, a separate qualification set gives a
one-sided 95% lower bound on accuracy among accepted cases. We apply the same protocol with the procedure as the unit:
does a rule "hand the procedure to the agent if its probe-based estimate is >= t" hold its target on procedures that
were not used to choose t, and how much is handed over?

## Data

- TauIndianBank (TIB): 50 families (20 of `tib/`, 30 of `combo_rep/`), agents gemini-3.8-flash (g38) and
  gemini-2.5-flash (flash), 6 real tasks per family, 6 documentation-only probes and 6 seeded probes per family
  (one family has 5 documentation probes). Probe scoring v2 for the first 20 families, as in the earlier reports.
- SOPBench: the 28 procedures with >= 8 cases are targets, 28 agents. A probe is a real held-out case of the procedure
  (n = 2, 4, 6 drawn at random per split); the remaining cases are the test cases. The other 42 procedures are history.

## Protocol (per agent, per probe type, per estimator)

1. Split target procedures at random into dev and qualification halves; 1000 splits (TIB), 300 splits (SOPBench).
2. Dev: real outcomes of dev procedures are known. Threshold t = the lowest value of the estimate such that pooled
   real success over cases of dev procedures with estimate >= t is >= target + 0.05. Targets 0.90 and 0.85.
   If no t satisfies it, nothing is handed to the agent.
3. Qualification: procedures with estimate >= t go to the agent. Report share of procedures and of cases handed over,
   pooled real success among them, one-sided 95% Clopper-Pearson lower bound on cases, and the 5th percentile of a
   bootstrap over accepted procedures (300 resamples; cases of one procedure are correlated).
4. Over splits: mean of the above, share of splits with nothing accepted, share of splits where realised success
   >= target, share where each lower bound >= target.

## Estimators of success (item 2)

With probes (k successes of n):
- raw: k/n.
- bb: Beta-binomial (m*pi + k)/(m + n), pi = agent's real success rate on dev (TIB) or on all non-qualification
  procedures (SOPBench), m in {2, 4, 8} chosen by Brier on dev.
- bb_knn: same, pi replaced by the kNN text forecast below.
- mf (TIB only): the frozen hierarchical model `tib/mf_params.pt` (`mf/model.py`), own-agent probes, as in `tib/analyze.py`.
With 0 probes:
- const: agent's real success rate on dev / non-qualification procedures (all-or-nothing rule).
- knn: similarity-weighted mean of real rates of the 5 nearest known procedures (temperature 0.05). SOPBench: bge
  procedure embeddings as in `aar_memory/memory.py`. TIB: no embeddings are stored, so cosine over TF-IDF of the
  one-line purpose plus the family-name prefix; for dev procedures the forecast is leave-one-out.

Brier: per-case Brier of the success forecast on qualification procedures, mean over splits.

## Notes fixed in advance

- At equal n, bb is a monotone transform of raw, so the accept sets can differ only through ties; it is expected to
  change Brier, not qualification.
- All numbers are averages over random splits of the same 50 (28) procedures; splits overlap, so the spread over splits
  is not a confidence interval.
- No multiple-comparison correction.
