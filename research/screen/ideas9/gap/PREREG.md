# Pre-registration, ideas9/gap (6 Oct 2026, written after the 10-case prompt test, before the full LLM run and before any harness run)

Idea (round-8 R1, R6, S2, A3, N08, F1): one cheap L1 call per case labels where each needed value comes from.
Model gemini-3.8-flash on Vertex, temperature 0, thinking low, JSON output. One prompt template for every benchmark (gap.py INSTR),
only the three inputs change: policy, tool schemas, request. The 10-case test (test10.json, 4 SOPBench, 4 tau2, 2 banking) was inspected
only for format and sense. The prompt was not changed afterwards.
Inputs: SOPBench policy and customer message are split out of the statement, tool schemas are the domain `actions` of the SOPBench
source filtered to the statement's tool list. tau2: request = statement, policy = domain policy files (telecom: main policy + tech support
manual), tools = ideas3/plan_tools/tool_text.json. Banking: agent policy header and additional instructions, tool NAMES only, the 700-document
knowledge base is not given (too long), so banking labels are weaker.

Features (feats.py, fixed now). Argument rows are de-duplicated by (tool, argument). Counts are per case:
n_args, n_given, n_derivable, n_fetch, n_ask, n_missing, frac_not_given = 1 - n_given/n_args, n_cond, n_cond_given, n_cond_fetch,
n_cond_ask, n_cond_unknown, frac_cond_not_given. Parse failures are imputed with the benchmark median.

Variants (ladder lad2 built exactly as ideas8/textcount/run.py builds t0_clad = ADeLe + PLAN + CMP):
- g1_clad_gap: c_lad + all 14 GAP features. PRIMARY HYPOTHESIS: g1 > t0_clad on the pooled four-scenario rho.
- g2_adele_gap: ADeLe + all 14 GAP features (no PLAN, no CMP).
- g3_clad_cond: c_lad + the 6 condition-source features only (n_cond ... frac_cond_not_given).
- g4_clad_3: c_lad + n_fetch, frac_not_given, n_cond_fetch (the three features of the round-8 diagnostics).
Reference: t0_clad (reproduce 0.258).

Run once, benches sopbench and tau2, schemes new_procedures and new_domain, paired bootstrap over procedures (copy of
ideas8/textcount/score.py), comparisons vs adele and vs t0_clad, plus SOPBench procedure level. No tuning, no reruns with other features
(a crash rerun is fine). A gain counts only if the paired 95% interval vs t0_clad is above 0.
Diagnostics (not variants): within-procedure Spearman with b of n_cond_fetch and frac_not_given on SOPBench (L2 reference +0.17),
Spearman of GAP features with PLAN features, banking within-domain Spearman with b (banking was never used to choose anything).
Budget: 1205 cases, 1089 distinct prompts, at most 1500 calls including the 10 test calls.
