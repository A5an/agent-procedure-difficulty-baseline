# Compile self-consistency as a documentation-only predictor (agent `compile_consistency`, round 3, 5 Oct 2026)

Saved by the main session from the agent's final message. Level L0 (policy text, tool list, schema, LLM-written
synthetic databases, no real case records, no labels). SOPBench only. About 1,252 gemini-3.8-flash calls.
Files: compile_more.py, programs_t1_s3.json, programs_t1_s4.json, synth_gen.py, synthetic.json, run_programs.py,
runs.json, features.py, unit_features.csv, cons_sopbench.csv, run_eval.py, run_ctrl.py, paired.py, out/run, out/ctrl,
diag.py, diag_out.txt, procedure_table.csv, tool_disagreement.csv, scatter_by_domain.png.

## Setup
5 programs per policy unit (t0, 2 existing t1, 2 new t1), 415 units. 10 synthetic scenarios per unit from the schema
skeleton of each domain's default_data (not from benchmark cases), all load in the env. 5 x 10 x 415 runs, deterministic.
Programs perform 20% of scenarios. 10 features fixed in advance: dec_dis, dec_ent, tool_pair, crash, tool_err, n_fields,
len_mean, branch_mean, cond_pair, cond_n_diff. Variants: cons, cons_cat_adele, lad2_adele_cons (rubric ladder L2 + ADeLe
+ the 10 features), averages with adele and with lad2_adele.

## Results (SOPBench, within-domain rho)

| method | sop-proc | sop-dom | mean | paired CI vs baseline (proc, dom) | procedure-level new dom [paired CI] | rho given label |
|---|---|---|---|---|---|---|
| adele | 0.120 | 0.125 | 0.123 | | 0.231 | 0.130, 0.142 |
| cons | 0.058 | 0.123 | 0.091 | [-0.198, 0.062], [-0.130, 0.107] | 0.362 [-0.095, 0.360] | 0.060, 0.137 |
| cons_cat_adele | 0.130 | 0.139 | 0.135 | [-0.079, 0.119], [-0.071, 0.105] | 0.415 [-0.040, 0.384] | 0.114, 0.146 |
| lad2_adele_cons | 0.207 | 0.217 | 0.212 | [-0.050, 0.233], [+0.012, 0.188] | 0.585 [+0.108, 0.605] | 0.195, 0.220 |
| lad2_adele_cons + adele | 0.193 | 0.218 | 0.206 | [-0.024, 0.173], [+0.035, 0.164] | 0.476 [+0.101, 0.429] | 0.180, 0.223 |
| cons + lad2_adele | | | | | 0.509 [+0.074, 0.480] | 0.171, 0.223 |

References: lad2_adele 0.181 / 0.179 (procedure level 0.411), grp_proc_adele 0.194 / 0.179 (0.438).
Paired vs lad2_adele (post hoc, case bootstrap, seed 0): case +0.042 [-0.015, +0.089] / +0.039 [-0.006, +0.082],
procedure level +0.174 [-0.034, +0.375]. Not significant.
Hard recall 20%: lad2_adele_cons 0.283 (avg with adele 0.306), baseline 0.257. Selective lift: 0.067 vs 0.036.
Brier skill 0.082 / 0.088 vs 0.014 / 0.005.

Post-hoc controls added to the ladder: program length and branch count only 0.219 / 0.199, procedure level 0.543.
Disagreement and error features only 0.192 / 0.197, 0.500. All 10: 0.207 / 0.217, 0.585. The gain is code complexity as
much as disagreement.

## Diagnostics
Procedure-level univariate rho with mean b: dec_dis +0.23, dec_ent +0.22, tool_pair -0.04, crash +0.16, len_mean +0.30,
branch_mean +0.28, cond_n_diff +0.18, tool_err +0.52 (mostly an artefact of env date helpers and the broken bank
procedures). Real-case gold-scored compile failure rate has 0.55 for comparison, synthetic consistency correlates with it
only weakly (+0.07 to +0.17).
Disagreement by step type: datetime 0.41, threshold 0.33, availability 0.32, status 0.19, existence 0.18, action 0.14,
relation 0.12, auth 0.03. Programs disagree most where agents skip most (datetime, availability, threshold), relation is
a counterexample. The procedure-level relation is flat per domain.

## Verdict
1. Consistency features alone do not beat the baseline at the case level. Procedure level 0.362, CI includes 0.
2. Added to the rubric ladder: best SOPBench numbers of the round (case 0.21 / 0.22, procedure level 0.585), gain over
   lad2_adele not significant, program length and branch count alone give the same gain.
3. "Disagreement means ambiguity means agent slips" not supported at the procedure level, although disagreement sits on
   the step types agents skip.

## Limitations
Ten features used without selection, controls post hoc. Wide procedure-level intervals (70 procedures). Synthetic
scenarios mostly refusals. tool_err and crash partly env quirks. SOPBench only.
