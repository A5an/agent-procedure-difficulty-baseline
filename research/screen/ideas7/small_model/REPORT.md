# Small open model first-action scoring (agent `small_model`, round 7, 6 Oct 2026)

Saved by the main session from the agent's final message. Level L1, local MLX (Qwen3-8B-Base 8-bit), about 17 min GPU,
laptop stayed cool. PREREG.md before evaluation.
SM features: logP(target action first) minus max logP(check first), entropy, rank (SOPBench); write vs read mass (tau2).
PRI features: policy rule surprisal (mean, max NLL per token) and customer-message NLL (SOPBench).
Analysis: margin vs b within procedure -0.014; vs smart-agent failure about 0; label correlation <= 0.07. Message NLL about -0.15.

| column | pooled | scenarios | vs c_lad |
|---|---|---|---|
| c_lad | 0.258 | .237 .212 .298 .285 | |
| v_a c_lad + SM | 0.231 | .230 .221 .234 .240 | -0.027 [-0.044, +0.009] |
| v_b c_lad + SM + PRI | 0.251 | .263 .266 .234 .240 | -0.007 [-0.034, +0.039] |
| v_c per-family penalties | 0.198 | .225 .238 .105 .223 | -0.060 |
| d_sm_only | 0.125 | .197 .235 .019 .050 | -0.133 |

SOPBench only, v_b vs c_lad: +0.026 / +0.054 (n.s.). Procedure level SOPBench new domain: v_b 0.689 [0.447, 0.794],
+0.202 [+0.007, +0.392] vs c_lad (likely from PRI, procedure-level, not pre-registered as the main hypothesis).
Verdict: first-action margin does not work; tau2 hurt; PRI is a lead at the procedure level only.
Files: llm_scores.py, feats.py, analysis.py, run.py, eval.py, score.py, out/, sm_*.csv, analysis_sop.csv, out_h/, score.txt.
