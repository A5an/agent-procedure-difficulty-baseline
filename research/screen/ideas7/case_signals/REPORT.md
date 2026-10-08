# Case-level signals: message vs policy (agent `case_signals`, round 7, 6 Oct 2026)

Saved by the main session from the agent's final message. Level L0, SOPBench only, local CPU models (DeBERTa NLI,
bge-base). Families: NLI message vs gating conditions (incl. skip-weighted, fitted per fold), retrieval margin of
conditions vs message + action, action readiness (argument cues), boundary margin (message numbers vs policy thresholds).
Within-procedure analysis: NLI nothing (policy-style hypotheses are neutral), retrieval the only signal (+0.15 to +0.18,
AUC 0.5, maybe message length), readiness and boundary nearly constant within procedures.

| column | pooled | scenarios | vs c_lad |
|---|---|---|---|
| c_lad | .258 | .237 .212 .298 .285 | |
| s1 NLI | .248 | .211 .196 | -0.010 [-0.024, +0.001] |
| s2 retrieval | .264 | .250 .222 | +0.006 [-0.019, +0.036] |
| s3 readiness | .250 | .225 .191 | -0.008 |
| s4 boundary | .253 | .226 .203 | -0.005 |
| s5 all four | .250 | .231 .187 | -0.008; procedure level .277 (-0.192 [-0.328, -0.038]) |

Smart-agent target: none beats c_lad. Verdict: none helps, the families carry no perform/refuse information.
Files: prep.py, feats.py, nli.py, skipf.py, run.py, analysis.py, score.py, fam_*.csv, analysis.csv, score.txt, out/.
