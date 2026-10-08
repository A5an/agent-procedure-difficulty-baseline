# Better ADeLe measurement (agent `adele_plus`, round 7, 6 Oct 2026)

Saved by the main session from the agent's final message. 3,548 gemini-3.8-flash calls (one call per task instead of 18;
format confound). Vertex has no logprobs for this model.
A: expected level from stated probabilities per scale (A0 = argmax control). B: 90 yes/no checklist questions (5 per scale,
checklists.py, written before results). C: rating relative to 3 anchor tasks of the same domain (not fold-aware).
Retest (40 shared tasks, mean per-scale rho): original 0.85, A 0.73, A0 0.66, B 0.80, C 0.92 (C partly copies the anchors).
Agreement with the original ADeLe weak except quantitative and logical scales (0.6 to 0.9).

| column | pooled | scenarios | vs adele | vs c_lad |
|---|---|---|---|---|
| adele | .131 | .120 .125 .128 .150 | | |
| c_lad | .258 | .237 .212 .298 .285 | +0.127 | |
| bA | .022 | .102 .125 -.020 -.118 | -0.108 | |
| bB | .176 | .047 .089 .294 .274 | +0.045 [-.018, +.123] | |
| bC | .136 | .034 .075 .337 .096 | +0.005 | |
| cA (c_lad with A) | .256 | .205 .195 .328 .296 | | -0.002 [-.062, +.042] |
| cB | .225 | .198 .177 .243 .280 | | -0.034 |
| cC | .215 | .171 .163 .302 .223 | | -0.044 |

Verdict: ADeLe measurement noise is not the bottleneck; cleaner variants do not raise predictive power. B helps the
baseline only on tau2. Files: run_llm.py, prompts.py, checklists.py, cache.jsonl, build.py, features/, reliab.py,
reliability.json, corr.py, adele_plus_harness.py, out/, ap_score_main.txt.
