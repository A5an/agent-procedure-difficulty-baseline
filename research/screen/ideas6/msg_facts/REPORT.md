# Compiled policy evaluated on facts stated in the message (agent `msg_facts`, round 6, 6 Oct 2026)

Saved by the main session from the agent's final message. Level L1 with L0 priors. 821 gemini-3.8-flash calls (fact
extraction only). 50 Monte Carlo databases per case from the domain's shipped default_data. Negative result.

AUC of P(perform | stated facts) vs the real label: 0.538 (post hoc prior 0.557), dry run 0.638, plan 0.556, rubric 0.606,
compiled pre-check reading the real record (L2) 0.770. Mean p_perf 0.205 for perform vs 0.150 for refuse cases.

| column | pooled | scenarios | vs c_lad |
|---|---|---|---|
| c_lad | .258 | .237 .212 .298 .285 | |
| f1 ADeLe + features | .238 | .194 .176 .298 .285 | -.020 [-.038, -.006] |
| f2 c_lad + features | .252 | .235 .191 .298 .285 | -.006 [-.017, +.004] |
| f3 c_lad + p_perf | .261 | .242 .220 .298 .285 | +.003 [+.000, +.007] |

Procedure level SOPBench: features hurt (f1 .346, f2 .374 vs c_lad .469). Why: messages state mostly action parameters,
which pin about one condition; the verdict rides on record fields the prior cannot know. Not adopted.
Files: extract.py, facts.json, sim.py, run_sim.py, msg_features*.csv, auc.py, run.py, eval.py, score.py, out/.
