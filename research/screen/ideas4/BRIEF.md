# Brief for round 4 (5 Oct 2026, evening): new cheap methods that beat the baseline

Read research/screen/ideas3/BRIEF.md first (all its rules apply: information
levels L0/L1/L2, shared harness, metrics_extra, cpu_lock, Vertex gemini-3.8-flash with caching, no leakage, no HANDBOOK).
Then read prototype/screen/ideas3/combo/REPORT.md and PREREG.md and run.py: that is the current best model.

## Where we stand (pooled within-domain rho over sop-proc, sop-dom, tau2-proc, tau2-dom)
- baseline adele 0.131 (.120 .125 .128 .150)
- lad2_adele (rubric ladder + ADeLe) 0.182 (.181 .179 .168 .199)
- c_lad (lad2 ladder + ADeLe + PLAN + CMP) 0.258 (.237 .212 .298 .285), paired CI vs baseline [+0.010, +0.198]
- tau2 gold-length oracle (uses gold, not allowed as a feature) 0.47 / 0.40 on tau2 scenarios: most tau2 headroom is in
  knowing how many actions and writes a task needs
- PLAN features: prototype/screen/ideas3/plan_tools/features/ (tau2) and plan_tau/features/ (SOPBench)
- CMP features: prototype/screen/ideas3/compile_consistency/unit_features.csv (len_mean, branch_mean per policy unit)

## What you must deliver
A NEW method that is honestly better than the baseline, and ideally adds to c_lad. Fix 2 to 3 variants before the first
full run, one of them "c_lad + your features" built exactly like combo/run.py builds c_lad. Run the harness once (benches
sopbench and tau2 unless your idea is tau2-only, schemes new_procedures and new_domain), metrics_extra, paired CIs vs adele
and vs c_lad if you can (score_combo.py shows how the combo agent compared columns).

## Time and budget
The user's subscription window closes in about an hour. Finish and report within 50 minutes. Keep LLM calls under the
limit in your task. If the full run will not fit, run the tau2 part only and say so.

## Output
Folder prototype/screen/ideas4/<your_name>/. Return the FULL report as text in your final message (idea in two sentences
and level, variants, calls, main table with paired CIs, verdict in 3 lines, limitations, paths). REPORT.md writes may be
refused for subagents.
