# PREREG across-benchmark check (round 10, 8 Oct 2026, written before any LOBO run with new features)

Model and target exactly as repo src/lobo.py (their FeatureBasedPredictor: StandardScaler + RidgeCV, ALPHAS grid,
z(beta) per benchmark, fit on two benchmarks, Spearman on the third, bootstrap over task groups). common/lobo2.py
reproduces the repo numbers (banking: human_time 0.582, length 0.532, adele 0.476).

Variants (features that exist for all three benchmarks):
- references: length, adele, human_time
- L_gap = adele + GAP (13, ideas9/gap)
- L_prior = adele + PRIOR (9, ideas10/prior)
- L_doc = adele + PLAN (9, plan3; banking plan from ideas7/evo_check/bank_plan.csv) + GAP + PRIOR + procedure rubric
  (rubric/features/rub_proc_<bench>.csv, numeric columns)
- L_doc_len = L_doc + length + human_time
Primary: L_doc_len vs length on held-out banking (paired bootstrap CI). Secondary: every variant on all three held-out
benchmarks, vs adele and vs length. Banking GAP and PRIOR were computed with tool names only and no knowledge base.
One run, no feature changes after seeing results.
- Added before any LOBO result was computed: L_doc_fc = L_doc_len + FC (fc/PREREG.md).
