# Ablation of c_lad (agent `ablation`, round 4, 5 Oct 2026, post hoc, exploratory)

Saved by the main session from the agent's final message. No LLM calls. Same ladder, penalties by inner CV, target all_irt.
Paired CIs: bootstrap over procedures, 300 draws (score_combo.py machinery).

| # | column | pooled [95% CI] | SOP-np | SOP-nd | TAU-np | TAU-nd | vs adele | vs col 1 | proc-level SOP-nd |
|---|---|---|---|---|---|---|---|---|---|
| ref | adele | 0.131 | .120 | .125 | .128 | .150 | | | 0.231 |
| 1 | lad2 + ADeLe | 0.182 | .181 | .179 | .168 | .199 | +0.051 [-.016,+.082] | | 0.411 |
| 2 | + PLAN | 0.245 | .206 | .192 | .298 | .285 | +0.115 [+.054,+.158] | +0.064 [+.025,+.111] | 0.390 |
| 3 | + CMP | 0.196 | .219 | .199 | .168 | .199 | +0.066 [+.002,+.095] | +0.015 [+.005,+.028] | 0.543 |
| 4 | c_lad | 0.258 | .237 | .212 | .298 | .285 | +0.127 [+.071,+.172] | +0.076 [+.037,+.128] | 0.486 |
| p5 | lad2 + PLAN, no ADeLe | 0.226 | .157 | .159 | .302 | .284 | +0.095 [+.014,+.147] | +0.044 [-.007,+.097] | 0.223 |
| p6 | c_lad + pairwise BT | 0.233 | .230 | .200 | .290 | .213 | +0.102 | +0.051 | 0.504 |

Decomposition of +0.076 over lad2_adele: PLAN +0.064 pooled (tau2 +0.108: new_proc +0.130 [+.048,+.200], new_dom +0.086
[-.057,+.225]; SOPBench +0.019), CMP +0.015 pooled (SOPBench +0.030: new_proc +0.038 [+.016,+.073], new_dom +0.021
[-.007,+.052]; procedure level 0.411 to 0.543, +0.133 [+.021,+.253]). Gains roughly additive.
ADeLe redundant on tau2 once PLAN is in, still matters on SOPBench. BT does not help. c_lad + COND skipped (Ladder takes
a static table, COND is fitted per fold).
Files: run.py, eval.py, score.py, out/, ab_score_main.txt.
