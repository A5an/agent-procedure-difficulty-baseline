# Pre-registration, ideas8/textcount (6 Oct 2026, written before the first harness run)

Exploratory follow-up of the round-8 swarm. The two pure-code request features were picked AFTER the swarm agents
screened many features against full-data IRT difficulty on the same SOPBench and tau2 cases, so any gain here is
post hoc and needs a blind test (HANDBOOK, ThinkingBox) before it counts.

Features (feats.py, L0, identical code on every benchmark, request = SOPBench customer message or tau2 scenario):
- DIG: dig_chars, dig_nums, dig_ids (agent `surface`, S01: within procedure and label rho about +0.11 on SOPBench)
- CRED: cred_maxent, cred_word (agent `request`, R2: within procedure rho about -0.23 on SOPBench)

Variants (ladder lad2 exactly as ideas4/ablation/run.py builds a4_full = c_lad):
- t0_clad: ADeLe + PLAN + CMP (reference, should reproduce c_lad 0.258)
- t1_clad_dig: t0 + DIG
- t2_clad_dig_cred: t0 + DIG + CRED  (primary hypothesis: t2 > t0 on the pooled four-scenario rho)
- t3_adele_dig_cred: ADeLe + DIG + CRED, no LLM features (pure L0 + ADeLe)

Run once: benches sopbench and tau2, schemes new_procedures and new_domain, paired bootstrap over procedures as in
ideas4/ablation/score.py, comparisons vs adele and vs t0_clad. No tuning, no reruns with other features.
Report telecom caveat and the SOPBench-only numbers.
