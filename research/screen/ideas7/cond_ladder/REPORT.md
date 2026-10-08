# Written-condition skip model fitted inside the c_lad ladder (agent `cond_ladder`, round 7, 6 Oct 2026)

Saved by the main session from the agent's final message. No LLM calls. PREREG.md before the run. CondLadder fits COND on
training procedures of each fold (out-of-fold for training rows), poison test confirms no test-procedure leakage.
First launches were killed by another agent's pkill, results from one complete rerun after 22:10.

| column | pooled | scenarios | vs c_lad |
|---|---|---|---|
| c_lad / rerun | 0.258 | .237 .212 .298 .285 | 0 |
| v1 c_lad + COND score | 0.263 | .241 .228 .298 .285 | +0.005 [-0.002, +0.015] |
| v2 c_lad + COND summaries | 0.257 | .242 .204 .298 .285 | -0.001 [-0.009, +0.008] |

Procedure level SOPBench new domain: v1 0.422, v2 0.365 (c_lad 0.486, drops not significant). Brier and log-loss unchanged.
Verdict: correctly placed, COND adds nothing; its information is already in c_lad's columns.
Files: PREREG.md, condladder.py, run.py, eval.py, score.py, poison.py, out/, cl_score_main.txt.
