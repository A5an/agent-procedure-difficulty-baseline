# tau2 tool-to-ID dependency graph (agent `tau_graph`, round 7, 6 Oct 2026)

Saved by the main session from the agent's final message. Level L0 + existing L1 plans, tau2 only, no LLM calls.
Hand-checked ID table for 47 agent tools (tool_table.json): every lookup depth is 0 or 1, the tau2 tool graph is shallow.
16 features over planned writes (depth, lookups, unstatable IDs, list-of-object args, nesting, enums, cross-argument
constraint words, depth-weighted call and write counts, telecom user steps).
Partial Spearman with b controlling for plan call count: airline and retail nothing beyond +-0.15 except the depth-weighted
call count on airline (+0.46, a call-count variant); telecom numbers driven by 3 procedures, not trusted.

| column | pooled | scenarios | vs c_lad |
|---|---|---|---|
| c_lad | 0.258 | .237 .212 .298 .285 | |
| g1 c_lad + 16 graph columns | 0.274 | .237 .212 .366 .283 | +0.016 [-0.009, +0.029] |
| g2 c_lad, PLAN + depth-weighted counts | 0.264 | .237 .212 .322 .284 | +0.006 [-0.000, +0.018] |

tau2 new procedures: g1 +0.068 [-0.013, +0.106] (telecom and airline), g2 +0.024 [+0.007, +0.040].
Verdict: idea not confirmed, the shallow ID graph carries the same information as the call count.
Files: build_table.py, tool_table.json, graph_feats.py, features/, partial.py, partial_spearman.csv, run7.py, score7.py, out/.
