# PREREG prior conflict (round 10, 8 Oct 2026, written before the full LLM run and before any evaluation)

Hypothesis. Agents fail mostly by skipping checks and by acting despite failed conditions (ideas2/steps). Both are what an
assistant does by default. So a case is hard when the policy requires things a policy-blind assistant would not do.

Measurement (L1, prior.py). Per case: 6 naive plans WITHOUT the policy (gemini-3.8-flash t0 + 2 samples t=1,
gemini-2.5-flash-lite t0 + 2 samples t=1), 1 policy plan WITH the policy (flash t0, lists rules that apply and rates each
1-5 for how expected it is without being told), 1 matching call (flash t0) that marks which naive plans cover each rule.
Only the 10-case test (test_raw.json) was looked at, for format only.

Features PRIOR (9, fixed now): pr_n_rules, pr_unant_sum = sum over rules of (1 - coverage/6), pr_unant_max,
pr_n_unanticipated (rules covered by at most 1 of 6), pr_default_mean, pr_n_lowdefault (default <= 2), pr_dec_conflict
(share of naive decisions different from the policy decision), pr_extra_actions (mean count of naive action calls whose
tool is not in the policy plan), pr_tool_unant (sum over policy-plan tools of 1 - share of naive plans calling it).
Missing values (failed parse, no rules) are filled with the benchmark median.

Variants (one harness run, sopbench + tau2, new_procedures + new_domain, Ladder exactly as ideas9/gap/run.py):
- pr1_clad_prior = c_lad + PRIOR
- pr2_cladgap_prior = c_lad + GAP + PRIOR      (primary)
- pr3_adele_prior = ADeLe + PRIOR
- references t0_clad, g1_clad_gap (rebuilt, should reproduce 0.258 and 0.311)
Primary: pr2 > g1 on the pooled metric (paired CI). Secondary: pr1 > t0_clad; SOPBench procedure level (new domain).
Fresh check: within-domain Spearman of each PRIOR feature with b on tau-Knowledge banking (never used for selection),
sign compared with the pooled sign on sopbench + tau2.
