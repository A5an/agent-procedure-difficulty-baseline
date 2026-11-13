# How the c_lad / f2 0.68-0.72 on fixed SOPBench labels was computed (13 Nov 2026, item 4 of the 13 Nov plan)

No new runs; this answers the question from the existing outputs (`data/sopbench_fixed/CLAD_F2_REPORT.md`,
`score_fixed.txt`, code in `sopaudit/clad_fixed/`).

The 0.68-0.72 are held-out numbers, not in-sample. Protocol of EVALUATION_PROTOCOL.md, unchanged: features and the
Ladder model are fit on training folds and predict procedures they never saw, in two schemes, new_procedures (5x5 folds
over procedures) and new_domain (7 folds, one domain held out). 70 procedures in 7 domains, 806 cases (830 minus the 24
broken ones), the 28 published agents of the baseline repo with every episode re-scored by the fixed grader (1032 of
23 240 labels change). Target: the procedure's IRT difficulty fitted on all 28 agents (the label, as in the baseline);
statistic: pooled within-fold Spearman between predicted and true difficulty, bootstrap over procedures.

| procedure level, new_procedures / new_domain | ADeLe (baseline) | c_lad | f2 |
|---|---|---|---|
| published labels | 0.22 / 0.26 | 0.44 / 0.45 | 0.43 / 0.46 |
| fixed labels | 0.63 / 0.65 | 0.68 / 0.68 | 0.72 / 0.72 |
| case level, published labels | 0.13 / 0.11 | 0.25 / 0.22 | 0.27 / 0.22 |
| case level, fixed labels | 0.09 / 0.13 | 0.26 / 0.24 | 0.29 / 0.26 |

Intervals on fixed labels, new_procedures: c_lad 0.68 [0.50, 0.80], f2 0.72 [0.51, 0.82], ADeLe 0.63 [0.40, 0.74].

Two things to state with these numbers:
- The rise comes mostly from the labels, not from the features: the ADeLe baseline rises the same way (0.22 -> 0.63), and
  the gain of c_lad/f2 over it shrinks to +0.03 to +0.09 with intervals that include 0.
- The earlier 0.2-0.35 for new procedures were case-level numbers or published labels; at case level the fixed labels
  change little (0.25 -> 0.26-0.29). "Ranks procedures at 0.68-0.72" is a procedure-level, held-out statement.
- Exploratory: the fixed-label analysis was not pre-registered (stated in CLAD_F2_REPORT.md).
