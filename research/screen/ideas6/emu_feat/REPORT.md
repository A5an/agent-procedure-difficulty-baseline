# Emulation features into c_lad (agent `emu_feat`, round 6, 6 Oct 2026)

Saved by the main session from the agent's final message. Level E (smart agents on documentation-generated synthetic
cases), L2+E for the pre-check mixture. No LLM or agent calls. Files: feats.py, emu_case_features.csv, run.py, eval.py,
score.py, score.txt, out/.

Features per case: fp, fr (procedure failure on perform / refuse synthetic cases, mean of Gemini and Sonnet), emu_est =
0.35 fp + 0.65 fr, emu_mean, mixtures p*fr + (1-p)*fp with p from plan (AUC .556), dry run (.638), rubric (.606),
compiled pre-check (L2, .770).

| column | pooled [CI] | scenarios | vs c_lad | vs adele |
|---|---|---|---|---|
| c_lad | .258 [.160, .327] | .237 .212 .298 .285 | | +.127 [+.070, +.167] |
| e1 c_lad + emu_est (L0+E) | .276 [.180, .341] | .252 .270 .298 .285 | +.018 [+.000, +.026] | +.146 |
| e2 c_lad + mix_plan (L1+E) | .270 | .252 .244 .298 .285 | +.012 [-.006, +.019] | +.139 |
| e3 ladder + mix_pre (L2+E) | .178 | .190 .229 .141 .154 | -.080 | +.048 |
| e4 c_lad + mix_pre (L2+E) | .289 [.194, .363] | .282 .289 .298 .285 | +.031 [+.003, +.055] | +.158 |

tau2 unchanged (no synthetic runs). SOPBench only: e1 .261 vs .225 (+.037 [+.001, +.051]), e4 .286 (+.061).
Procedure level SOPBench (mean of two scenarios): c_lad .469, e1 .585, e4 .580. fr alone at procedure level .623.
Smart-agent target (SOPBench case level): c_lad .116, e1 .236, e4 .297, raw mix_dry .330; procedure level c_lad .343,
e1 .711, raw emu_est .766.
Verdict: small borderline gain on the protocol target (+.018), only L2+E clears zero (+.031). Large gain against the
smart-agent target, which is expected since the same agents ran the synthetic cases.
