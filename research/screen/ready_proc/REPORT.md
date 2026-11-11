# READY-style qualification at the procedure level + matched-budget baselines: result (11 Nov 2026)

Plan `PREREG.md`, code `ready_proc.py`, output `ready_proc.out`, numbers `results.json`. No LLM calls, $0.
Exploratory rerun without the dev margin: `ready_proc_margin0.out`, `results_margin0.json`.

Protocol: procedures are split at random into dev and qualification halves (1000 splits on TauIndianBank, 300 on
SOPBench). On dev the lowest threshold on the estimate is taken such that pooled real success among accepted dev
procedures is >= target + 0.05. On qualification the procedures above the threshold go to the agent; we report what
share goes, real success among them, and a one-sided 95% lower bound (Clopper-Pearson on cases; 5th percentile of a
bootstrap over accepted procedures). "qualified" = lower bound >= target; splits with nothing accepted count as not
qualified. "none" = share of splits where nothing is handed to the agent.

## 1. TauIndianBank, 50 families, 6 real tasks and 6 probes per family

Always-agent real success: gemini-3.8-flash (g38) 0.657, gemini-2.5-flash (flash) 0.680.

Target 0.90 (dev must reach 0.95):

| agent | estimator | Brier | none | cases to agent | success | held (>= 0.90) | CP bound | qualified CP | qualified bootstrap |
|---|---|---|---|---|---|---|---|---|---|
| g38 | const, 0 probes | 0.2315 | 1.00 | 0.00 | - | - | - | 0.00 | 0.00 |
| g38 | knn text, 0 probes | 0.2313 | 0.74 | 0.04 | 0.802 | 0.23 | 0.563 | 0.00 | 0.04 |
| g38 | doc raw | 0.2393 | 1.00 | 0.00 | 0.589 | 0.00 | 0.480 | 0.00 | 0.00 |
| g38 | doc bb | 0.2166 | 1.00 | 0.00 | 0.589 | 0.00 | 0.480 | 0.00 | 0.00 |
| g38 | doc bb_knn | 0.2147 | 0.52 | 0.07 | 0.868 | 0.46 | 0.652 | 0.00 | 0.07 |
| g38 | doc mf | 0.2303 | 0.92 | 0.02 | 0.603 | 0.00 | 0.442 | 0.00 | 0.00 |
| g38 | seeded raw | 0.1775 | 0.86 | 0.05 | 0.855 | 0.00 | 0.756 | 0.00 | 0.00 |
| g38 | seeded bb | 0.1713 | 0.86 | 0.05 | 0.855 | 0.00 | 0.756 | 0.00 | 0.00 |
| g38 | seeded bb_knn | 0.1699 | 0.28 | 0.15 | 0.910 | 0.58 | 0.735 | 0.01 | 0.20 |
| g38 | seeded mf | 0.1700 | 0.53 | 0.11 | 0.860 | 0.22 | 0.706 | 0.00 | 0.00 |
| flash | const, 0 probes | 0.2246 | 1.00 | 0.00 | - | - | - | 0.00 | 0.00 |
| flash | knn text, 0 probes | 0.2513 | 0.39 | 0.12 | 0.875 | 0.57 | 0.684 | 0.02 | 0.28 |
| flash | doc raw | 0.2028 | 0.86 | 0.06 | 0.776 | 0.00 | 0.668 | 0.00 | 0.00 |
| flash | doc bb | 0.1919 | 0.74 | 0.09 | 0.804 | 0.00 | 0.690 | 0.00 | 0.00 |
| flash | doc bb_knn | 0.1960 | 0.14 | 0.19 | 0.908 | 0.64 | 0.755 | 0.04 | 0.35 |
| flash | doc mf | 0.1861 | 0.72 | 0.08 | 0.739 | 0.00 | 0.599 | 0.00 | 0.00 |
| flash | seeded raw | 0.1472 | 0.67 | 0.17 | 0.887 | 0.43 | 0.809 | 0.00 | 0.00 |
| flash | seeded bb | 0.1472 | 0.67 | 0.17 | 0.887 | 0.43 | 0.809 | 0.00 | 0.00 |
| flash | seeded bb_knn | 0.1495 | 0.07 | 0.35 | 0.935 | 0.79 | 0.839 | 0.14 | 0.42 |
| flash | seeded mf | 0.1491 | 0.65 | 0.18 | 0.885 | 0.40 | 0.804 | 0.00 | 0.00 |

Target 0.85 (dev must reach 0.90):

| agent | estimator | none | cases to agent | success | held (>= 0.85) | CP bound | qualified CP | qualified bootstrap |
|---|---|---|---|---|---|---|---|---|
| g38 | knn text, 0 probes | 0.55 | 0.10 | 0.798 | 0.30 | 0.600 | 0.01 | 0.06 |
| g38 | doc raw | 0.97 | 0.01 | 0.596 | 0.00 | 0.486 | 0.00 | 0.00 |
| g38 | doc bb_knn | 0.25 | 0.15 | 0.855 | 0.58 | 0.671 | 0.02 | 0.19 |
| g38 | seeded raw (= bb) | 0.41 | 0.22 | 0.880 | 0.90 | 0.782 | 0.00 | 0.02 |
| g38 | seeded bb_knn | 0.09 | 0.33 | 0.891 | 0.77 | 0.773 | 0.10 | 0.25 |
| g38 | seeded mf | 0.28 | 0.24 | 0.894 | 0.92 | 0.775 | 0.00 | 0.11 |
| flash | knn text, 0 probes | 0.25 | 0.18 | 0.869 | 0.58 | 0.698 | 0.11 | 0.37 |
| flash | doc raw | 0.63 | 0.17 | 0.792 | 0.00 | 0.692 | 0.00 | 0.00 |
| flash | doc bb_knn | 0.05 | 0.29 | 0.881 | 0.59 | 0.756 | 0.20 | 0.39 |
| flash | seeded raw (= bb) | 0.19 | 0.45 | 0.892 | 0.87 | 0.821 | 0.25 | 0.23 |
| flash | seeded bb_knn | 0.01 | 0.51 | 0.909 | 0.85 | 0.834 | 0.45 | 0.43 |
| flash | seeded mf | 0.19 | 0.46 | 0.891 | 0.86 | 0.820 | 0.25 | 0.23 |

All 50 families, real success by the number of probes passed (no split, descriptive):

| agent, probes | 6 of 6 | 5 of 6 | 4 of 6 | 3 or fewer |
|---|---|---|---|---|
| g38, seeded | 18 families, 98/108 = 0.91 (bound 0.85) | 15 families, 62/90 = 0.69 | 2 families, 8/12 = 0.67 | 15 families, 29/90 = 0.32 |
| flash, seeded | 24 families, 134/144 = 0.93 (bound 0.89) | 5 families, 23/30 = 0.77 | 5 families, 13/30 = 0.43 | 16 families, 34/96 = 0.35 |
| g38, doc | 17 families, 74/102 = 0.73 (bound 0.64) | 11 families, 51/66 = 0.77 | 5 families, 17/30 = 0.57 | 17 families, 55/102 = 0.54 |
| flash, doc | 15 families, 80/90 = 0.89 (bound 0.82) | 5 families, 24/30 = 0.80 | 5 families, 18/30 = 0.60 | 25 families, 82/150 = 0.55 |

What the tables say:

- The 0.90 claim cannot be certified with this amount of data. The only accept rule seeded probes support is "6 of 6
  passed", whose real success is 0.91 to 0.93. A dev half holds about 9 to 12 such families, so dev success reaches the
  required 0.95 only by luck (nothing is accepted in 86% of splits for g38, 67% for flash), and a qualification half
  with about 60 accepted cases cannot push a Clopper-Pearson bound to 0.90 even when it is accepted. At true success
  0.93 the bound reaches 0.90 with about 278 accepted cases (about 46 families of 6 cases), reaches 0.85 with about 62
  cases (about 10 families).
- At target 0.85 seeded probes hand 22% (g38) and 45% (flash) of cases to the agent with real success 0.88 to 0.89;
  the realised success is >= 0.85 in 90% and 87% of splits. The lower bound averages 0.78 and 0.82; it clears 0.85 in
  0% (g38) and 25% (flash) of splits. So the rule is right on average, but one half of 25 families is too small for
  the formal statement.
- Documentation-only probes do not support any rule for g38 (6 of 6 passed gives 0.73 real success) and are
  borderline for flash (0.89).
- Without the dev margin (exploratory, `ready_proc_margin0.out`), target 0.90, seeded raw: g38 hands over 22% of
  cases with success 0.882 (held in 36% of splits), flash 46% with 0.895 (held in 47%). The margin is what turns
  "right on average" into "nothing accepted"; it does not make the bound reachable.

## 2. Baselines at the same probe budget (item 2)

- Raw k/n against Beta-binomial: at equal n the Beta-binomial estimate is the same ordering of procedures, so the
  qualification outcome is identical (rows "seeded raw" and "seeded bb" coincide; for doc probes the one family with
  5 probes changes ties). It only fixes the level: Brier 0.2393 -> 0.2166 (g38 doc), 0.1775 -> 0.1713 (g38 seeded),
  0.2028 -> 0.1919 (flash doc), no change for flash seeded (0.1472).
- Frozen hierarchical model (mf): Brier about the same as Beta-binomial on seeded probes (0.1700 vs 0.1713 g38;
  0.1491 vs 0.1472 flash), worse on doc probes for g38 (0.2303 vs 0.2166), better for flash (0.1861 vs 0.1919).
  Qualification: for g38 seeded it accepts something more often (none 0.53 vs 0.86 at 0.90; 0.28 vs 0.41 at 0.85) with
  the same success; for flash it is the same as raw. It does not beat simple counting by a clear margin.
- 0 probes: the agent's overall rate never accepts anything (0.66 to 0.68 is below any target). kNN over the purpose
  text using real outcomes of the dev families accepts 4 to 18% of cases with success 0.80 to 0.88, Brier no better
  than the constant (0.2313 vs 0.2315 g38; 0.2513 vs 0.2246 flash).
- Probes plus the kNN prior (bb_knn) is the most permissive rule and the only one that sometimes qualifies at 0.90:
  seeded bb_knn hands over 15% (g38) and 35% (flash) of cases with success 0.910 and 0.935, bootstrap-qualified in
  20% and 42% of splits. Its Brier is the same as plain Beta-binomial (0.1699 vs 0.1713; 0.1495 vs 0.1472), so the
  gain is in separating the 6-of-6 families, not in overall accuracy. Note that the kNN prior uses real outcomes of
  the 25 dev families of the same benchmark (templated family groups), the favourable case for Memory.

## 3. SOPBench, 28 target procedures, probes = real held-out cases

Top 7 agents (always-agent success 0.700); target 0.85 in brackets:

| estimator | Brier | none, 0.90 (0.85) | cases to agent | success | held | CP bound | qualified bootstrap |
|---|---|---|---|---|---|---|---|
| const, 0 probes | 0.2125 | 1.00 (1.00) | 0.00 (0.00) | - | - | - | 0.00 (0.00) |
| knn memory, 0 probes | 0.1958 | 0.58 (0.37) | 0.06 (0.15) | 0.911 (0.883) | 0.64 (0.68) | 0.683 (0.703) | 0.20 (0.28) |
| n=2 raw | 0.2686 | 0.99 (0.94) | 0.00 (0.04) | 0.780 (0.798) | 0.06 (0.28) | 0.716 (0.733) | 0.00 (0.01) |
| n=2 bb | 0.2052 | 0.98 (0.93) | 0.01 (0.04) | 0.785 (0.796) | 0.06 (0.26) | 0.718 (0.731) | 0.00 (0.01) |
| n=2 bb_knn | 0.1943 | 0.53 (0.32) | 0.07 (0.17) | 0.908 (0.888) | 0.62 (0.69) | 0.686 (0.714) | 0.21 (0.31) |
| n=4 raw | 0.2233 | 0.94 (0.78) | 0.02 (0.09) | 0.826 (0.829) | 0.07 (0.44) | 0.733 (0.740) | 0.00 (0.04) |
| n=4 bb | 0.1990 | 0.93 (0.78) | 0.03 (0.09) | 0.830 (0.827) | 0.12 (0.43) | 0.740 (0.739) | 0.00 (0.04) |
| n=4 bb_knn | 0.1922 | 0.49 (0.29) | 0.07 (0.17) | 0.914 (0.893) | 0.64 (0.71) | 0.684 (0.712) | 0.24 (0.35) |
| n=6 raw | 0.2080 | 0.84 (0.62) | 0.04 (0.12) | 0.855 (0.855) | 0.29 (0.57) | 0.721 (0.735) | 0.02 (0.12) |
| n=6 bb | 0.1951 | 0.84 (0.62) | 0.04 (0.12) | 0.855 (0.853) | 0.28 (0.56) | 0.721 (0.736) | 0.02 (0.11) |
| n=6 bb_knn | 0.1904 | 0.46 (0.26) | 0.07 (0.17) | 0.916 (0.895) | 0.65 (0.71) | 0.673 (0.712) | 0.25 (0.37) |

Clopper-Pearson qualification is 0.00 to 0.03 at target 0.90 and at most 0.13 at 0.85 in every row. For the middle 14
and bottom 7 agents (success 0.45 and 0.19) almost nothing is accepted by any rule (`ready_proc.out`).

Even with the best possible probes (real cases) and the strongest agents, 2 to 6 probes per procedure alone almost
never produce an accept rule (none 0.84 to 0.99 at target 0.90). The Memory prior does most of the work here: kNN with
0 probes already gives 6% of cases at success 0.911, and adding 6 probes moves it to 7% at 0.916. Beta-binomial again
changes Brier (0.2686 -> 0.2052 at n=2), not the accept set.

## Reading

1. As a point rule the earlier claim holds out of sample: seeded probes "all 6 passed" give real success 0.91 to
   0.93 on 36 to 48% of the 50 families. As a certified claim it does not hold yet: with 25 qualification families of
   6 cases a 95% lower bound of 0.90 is out of reach, 0.85 is reached in at most 25 to 45% of splits.
2. The size needed is known: about 10 accepted families (60 cases) for a 0.85 bound, about 46 accepted families for a
   0.90 bound at the observed 0.93. So a certified 0.85 statement needs roughly 50 qualification families per agent
   (twice the present half), or more real cases per family.
3. Simple counting is as good as the hierarchical model for the accept decision. What changes the decision is extra
   information per procedure: seeded instead of documentation probes, and a prior from real outcomes of similar
   procedures.

## Limits

- 50 families of one templated benchmark, 2 agents; the first 20 families were looked at in earlier experiments,
  the split protocol here was run on all 50 together.
- Seeded probes saw two other benchmark tasks of the same family (upper bound for "past conversations").
- Averages over overlapping random splits; the spread over splits is not a confidence interval, and "share of splits
  qualified" is a description of this dataset, not a probability for a new one.
- Clopper-Pearson treats cases as independent; cases inside a family are correlated, which is why the bootstrap over
  families is also shown. With few accepted families the bootstrap bound is optimistic (one accepted family with 6/6
  gives 1.0), so the two "qualified" columns bracket the answer rather than agree.
- The TIB kNN prior uses hand-made TF-IDF over a one-line purpose plus the family-name prefix (no stored embeddings);
  the prefix is benchmark metadata a real deployment would not have.
- SOPBench success is pooled over cases, so the largest procedures (up to 99 cases) dominate; 5 missing response
  cells were imputed and rounded to 0/1.
- The frozen hierarchical model was not refitted on dev; a per-split refit and the SOPBench version of the model were
  not computed.
- The zero-margin run and the 6-of-6 table were added after seeing the first output (exploratory).
- Many comparisons, no multiple-comparison correction.
