# PREREG pre-mortem forecaster (round 10, 8 Oct 2026, written after a 3-case format test, before the full run)

Measurement (L1, fc.py): Sonnet 5.5 via headless claude -p, no tools, one call per distinct prompt. Reads policy,
tools and request (same inputs as ideas9/gap), assumes a 2024-2025 agent will fail, lists up to 6 case-specific failure
modes with probabilities, and gives p_success for a typical agent of that generation.

Features FC (5, fixed now): fc_diff = logit(1 - p_success) (clipped to [0.02, 0.98]), fc_n_modes, fc_sum_p, fc_max_p,
fc_noisy_or = 1 - prod(1 - p_i). Missing values filled with the benchmark median.

Variants (one harness run, sopbench + tau2, new_procedures + new_domain):
- f1_cladgap_fc = c_lad + GAP + FC   (primary: f1 > g1_clad_gap)
- f2_all_fc = c_lad + GAP + SCN (tau2) + PRIOR + FC (vs c1_all of the zs run)
- f3_adele_fc = ADeLe + FC
- f0_fc_only = FC alone in the ladder (diagnostic)
Fresh check: within-domain Spearman of fc_diff with b on tau-Knowledge banking, and the LOBO run (lobo/PREREG.md) gets an
added variant L_doc_fc = L_doc_len + FC, declared here before any LOBO result is seen.
