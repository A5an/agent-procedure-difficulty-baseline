# PREREG within-domain standardisation (round 10, 8 Oct 2026, before any run)

Motivation: the metric only ranks cases inside a domain, but the Ladder is fitted on several domains at once, so feature
weights also try to explain differences between domains (and between procedures of different domains). Replacing each
feature by its z-score inside its domain (label-free, uses only the feature values of that domain, also on the held-out
domain) removes domain offsets from the features. Rubric step counts and global rubric columns are transformed the same way.

Variants (one harness run, sopbench + tau2, new_procedures + new_domain, Ladder exactly as ideas9/gap/run.py otherwise):
- z0_clad = c_lad, all inputs z-scored within domain
- z1_cladgap = c_lad + GAP, z-scored within domain   (primary: z1 > g1_clad_gap)
- z2_cladgap_prior = c_lad + GAP + PRIOR, z-scored within domain
- references t0_clad, g1_clad_gap, pr2_cladgap_prior (unchanged inputs)
Report pooled and per scenario, paired CI vs g1_clad_gap and vs adele, SOPBench procedure level.

## Added before any run (same harness run): all validated cheap families together
- c1_all = c_lad + GAP + SCN (tau2 only; ideas4/user_scenario/scn_tau2.csv, 10 columns) + PRIOR
- c1z_all = c1_all with within-domain z-scoring
Hypothesis: c1 >= pr2 (SCN adds tau2 scenario structure that GAP and PLAN do not cover). Report vs g1_clad_gap.
