# Customer-scenario complexity for tau2 (agent `user_scenario`, round 4, 5 Oct 2026)

Saved by the main session from the agent's final message. Level L0. 232 gemini-3.8-flash calls (3 clause lists, 169
scenario calls, 60 stability calls). tau2 only, pooled number reuses the SOPBench part unchanged.
SCN features (10): n_goals, n_mind_changes, withholds_info, policy_forbidden_request, pressure (0 to 2), n_conditional,
needs_computation, ambiguous_reference, n_constraints, n_clauses. Clause features: shrunk mean difficulty per policy clause
(k=3) learned inside training folds, mean and max over a task's clauses.

| method | tau2 new_proc | tau2 new_dom | pooled4 | vs adele (pooled) | vs c_lad (pooled) |
|---|---|---|---|---|---|
| adele | .128 | .150 | .131 | | -0.127 |
| c_lad | .298 | .285 | .258 | +0.127 [+.066,+.170] | |
| scn_adele | .094 | .413 | .188 | +0.057 [-.086,+.076] | -0.070 |
| cls_adele | .115 | .150 | .127 | -0.003 | -0.131 |
| clad_scn | .306 | .370 | .281 | +0.150 [+.076,+.183] | +0.023 [-.013,+.038] |
| clad_scn_cls | .302 | .379 | .283 | +0.152 [+.074,+.181] | +0.024 [-.024,+.043] |

tau2 paired vs c_lad: clad_scn new_proc +.007 [-.100,+.062], new_dom +.085 [-.008,+.129]; clad_scn_cls new_dom +.094
[-.003,+.145]. Bootstrap over procedures (score_us.py), not the harness bootstrap. metrics_extra not run.
Stability (60 texts at temperature 0.7): flags and goal counts Spearman 0.73 to 0.95, n_clauses 0.57, clause Jaccard 0.74.
Verdict: scenario features help tau2 new_domain (.370 vs .285, CI touches 0), pooled +0.023 over c_lad not significant.
Clause features do nothing. No ablation of single features.
Files: gen.py, features.py, runus.py, score_us.py, score.txt, scn_tau2.csv, clauses.json, clause_map.json, raw.json.
