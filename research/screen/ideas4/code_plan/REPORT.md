# Code plans per case and richer AST features (agent `code_plan`, round 4, 5 Oct 2026)

Saved by the main session from the agent's final message. 762 gemini-3.8-flash calls (tau2 167, SOPBench 595 of 823,
stalled). Code plan (L1): one call per case writes handle(tools) for that request, never executed, 15 static AST features
(calls, writes, reads, if-branches, nesting, comparisons, date ops, loops, early returns, refusal returns, guarded calls,
guarded writes, distinct tools, boolean ops, lines). CMP+ (L0): the same AST features from the existing SOPBench policy
programs (5 per unit), no LLM calls.

tau2 (full): adele .128 / .150, c_lad .298 / .285, cp_adele .214 / .212, c_lad_cp .315 / .309.
c_lad_cp vs c_lad tau2 pooled +.021 [+.002, +.045], vs adele +.173 [+.072, +.239].
SOPBench c_lad_cmpp (CMP replaced by CMP+): .255 / .243 vs c_lad .237 / .212, vs c_lad +.018 [-.016, +.090] and +.031
[-.015, +.094]. Pooled four scenarios about .270 (tau2 part from c_lad). Procedure level SOPBench new domain .503 (c_lad .486).
Not done: L1 code plan on SOPBench (595 of 823 generated, resumable: gen.py sop, build_cp.py, run.py main, score.py main).
Bootstrap over procedures, 300 draws. SOPBench write/read split by tool-name heuristic.
Files: gen.py, astf.py, build_cp.py, build_cmpplus.py, run.py, score.py, code_tau2.json, cp_tau2.csv,
cmpplus_sopbench.csv, out_tau2/, out_sopL0/, score_tau2.txt, score_sopL0.txt.

## Finished SOPBench part (agent `code_plan_finish`, 6 Oct 2026)
230 more calls, all 823 SOPBench prompts cached, 830 cases parsed. 300-draw procedure bootstrap (score.py).

| variant | pooled [95% CI] | scenarios | vs adele | vs c_lad |
|---|---|---|---|---|
| adele | .131 | .120 .125 .128 .150 | | |
| c_lad | .258 | .237 .212 .298 .285 | +0.127 [+0.071, +0.172] | |
| cp_adele | .149 | .057 .113 .214 .212 | +0.018 [-0.019, +0.061] | -0.109 |
| c_lad_cp | .257 | .206 .196 .315 .309 | +0.126 [+0.069, +0.175] | -0.002 [-0.013, +0.019] |
| c_lad_cmpp | .270 | .255 .243 .298 .285 | +0.140 [+0.087, +0.189] | +0.012 [-0.002, +0.038] |

SOPBench: c_lad_cp vs c_lad -0.031 / -0.016 (n.s.), code plan adds nothing there (cp_writes nearly constant). tau2 gain
+0.021 [+0.004, +0.047] stands. Procedure level new domain: c_lad 0.486, c_lad_cp 0.493, c_lad_cmpp 0.503.
rho_given_label c_lad_cp .222 / .191 below c_lad .253 / .218.
Verdict: code plan helps only tau2 (small), CMP+ is the best SOPBench variant (+0.012 pooled, n.s.).
