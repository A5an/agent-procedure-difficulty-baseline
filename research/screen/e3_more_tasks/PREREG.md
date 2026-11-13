# E3-B: the unused TauIndianBank tasks (pre-registration, 13 Nov 2026)

Written before any episode. Plan of 13 Nov (Asanali), Claude subscription through `claude -p`, $0.

Why. No source adds new procedures, but 700 tasks of the 50 seedk families were never run as evaluation tasks; 249 of
them were seeds of the family's own k1/k2/k5 probes, 451 were never used at all. Running the 451 gives 2-3 times more
real tasks per family and a tighter lower bound on real success among accepted families.

Design. `select_new.py` -> `new_tasks.json`: every task of the 50 families outside the 300 evaluation tasks, with flags
seed_own (a seed of its own family's probes: the generator saw it; excluded from the primary analysis) and seed_foreign
(a seed of another family's foreign probes; kept, flagged). 451 non-seed tasks in 46 families (min 1, median 8, max 19
per family; 4 families have none). Agents Claude Haiku and Claude Sonnet, customer Claude Haiku as in E1-C, 1 attempt,
`../claude_tib/run_claude_tib.py --set new`, outputs `../claude_tib/runs/new_<agent>.jsonl` (copied here at the end).
Deviation from the plan, fixed now: the 249 own-seed tasks are run only after the 451 and only if the subscription
allows (`--set newseed`); they are excluded from the primary analysis in any case.

Primary outcome. For the rule "all 6 probes passed" with k=2 and with k=5 probes (E1-C runs), per agent: real success on
the combined set of tasks (the old 6 evaluation tasks plus the new non-seed tasks) of the accepted families, the one-sided
95% Clopper-Pearson lower bound over cases and the 5th percentile of a bootstrap over accepted families (2000), with the
number of accepted families and cases. Cases of one family are dependent, so the Clopper-Pearson bound is optimistic;
both bounds are reported.

Secondary. Per family, success on the new tasks against the old six (Spearman over families with at least 3 new tasks,
mean difference with a bootstrap interval), to see how noisy the 6-task estimate was; the E1-C Spearman between probe
failure rate and real failure rate (k0, k2, k5, foreign) recomputed on the extended set.

Code `analyze_e3b.py`, hashed into `FREEZE_PRE.txt` before the first episode. Later additions are exploratory.
