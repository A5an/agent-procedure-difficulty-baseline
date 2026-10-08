# Simulated agent personas on paper (agent `personas`, round 6, 6 Oct 2026)

Saved by the main session from the agent's final message. Level L1. About 1,060 gemini-3.8-flash calls. Negative.
One call per case role-plays 5 personas (rushed, literal, forgetful, average, expert): tool names, decision, skipped
conditions. 13 features (share of correct-looking personas, personas acting without checks, decision disagreement, weakest
correct persona, skips, calls).

| variant | pooled [CI] | scenarios | vs c_lad |
|---|---|---|---|
| c_lad | 0.258 | .237 .212 .298 .285 | |
| pers | 0.189 | .158 .118 .316 .165 | -0.069 |
| pers_cat_adele | 0.165 | .202 .165 .185 .107 | -0.093 [-0.132, -0.039] |
| c_lad_pers | 0.225 | .230 .187 .253 .232 | -0.033 [-0.045, +0.002] |

Procedure level SOPBench new domain: pers -0.236, c_lad_pers 0.334 (c_lad 0.486). Variance was created (SOPBench share_ok
mean 0.17, sd 0.19) and stable (0.85 to 0.90 between runs), raw rho with difficulty up to 0.3, but personas choose perform
in 96% of SOPBench cases (label AUC 0.53 to 0.55) and add nothing to c_lad. tau2 new procedures 0.316 alone comes from the
PLAN proxy inside the feature.
Files: plib.py, collect.py, feats.py, run.py, score.py, features/, cache.jsonl, out/.
