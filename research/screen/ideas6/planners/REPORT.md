# Second planner (Sonnet 5.5) and planner agreement (agent `planners`, round 6, 6 Oct 2026)

Saved by the main session from the agent's final message. Level L1. 992 Sonnet plans (claude -p, no tools, subscription,
about 25 USD equivalent). Files: gen_sonnet.py, sonnet_*.json, feats6.py, features/, run6.py, out/, score6.py,
ab_score_main.txt, validate6.py, validation6.json.

| column | pooled [CI] | scenarios | vs c_lad (paired) |
|---|---|---|---|
| b_clad (c_lad rebuilt) | 0.258 | .237 .212 .298 .285 | |
| v1_int (PLAN replaced by intersection of Gemini and Sonnet plans) | 0.289 [.152, .354] | .254 .232 .292 .380 | +0.031 [-.060, +.060] |
| v2_son (c_lad + Sonnet plan) | 0.272 | .238 .217 .312 .323 | +0.014 [-.023, +.046] |
| x_avg (PLAN = mean of both) | 0.275 | .255 .228 .304 .313 | +0.017 [-.020, +.040] |
| v3_dis (c_lad + disagreement) | 0.139 | .229 .198 .160 -.030 | -0.119 [-.173, +.003] |

v1_int tau2 new domain +0.095 [-0.085, +0.181]. Procedure level SOPBench new domain: v1_int 0.500, x_avg 0.513 (c_lad 0.486).
Validation vs gold (calls / writes): airline Gemini .23/.62, Sonnet .31/.69, intersection .24/.67; retail .17/.82,
.18/.80, .16/.84; telecom Sonnet over-plans (10.4 calls vs gold 1.1). Planned refusal AUC for the refuse label 0.55 to
0.57. Planners agree on the outcome in 93% of SOPBench cases.
Verdict: a second planner does not recover the missing half of the gold-length signal. Best +0.031, not significant.
