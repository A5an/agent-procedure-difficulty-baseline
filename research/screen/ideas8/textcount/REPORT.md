# ideas8/textcount: c_lad plus pure-code request counts (6 Oct 2026, main session)

Exploratory, post hoc: the features were chosen after swarm agents screened them on the same data (see PREREG.md).
Features in feats.py (L0, same code on every benchmark): DIG = dig_chars, dig_nums, dig_ids; CRED = cred_maxent, cred_word.
Harness: sopbench and tau2, new_procedures and new_domain, paired bootstrap over procedures (score.py, copy of
ideas4/ablation/score.py). Full lines in tc_score_main.txt.

| variant | pool4 | sop-proc | sop-dom | tau2-proc | tau2-dom | vs c_lad (pool4) | proc sop-dom |
|---|---|---|---|---|---|---|---|
| adele (baseline) | 0.131 | .120 | .125 | .128 | .150 | -0.127 | 0.231 |
| t0_clad (reproduces c_lad) | 0.258 | .237 | .212 | .298 | .285 | | 0.486 |
| t1 c_lad + DIG | 0.250 | .222 | .210 | .290 | .277 | -0.008 [-0.022, +0.005] | 0.497 |
| t2 c_lad + DIG + CRED (primary) | 0.263 | .257 | .227 | .296 | .273 | +0.005 [-0.013, +0.028] | 0.454 |
| t3 ADeLe + DIG + CRED (no PLAN, no CMP) | 0.192 | .226 | .209 | .149 | .182 | -0.066 [-0.118, -0.024] | 0.393 |

Verdict: the primary hypothesis fails, t2 is +0.005 over c_lad (n.s.). SOPBench alone +0.020 and +0.015 (n.s.), tau2 slightly
negative. The within-procedure single-feature signals of the swarm screens (+0.11 digits, -0.23 credentials) are already
covered by c_lad's PLAN and CMP features. Side result: t3, which uses only ADeLe and five regex counts, reaches 0.226 and
0.209 on SOPBench (c_lad .237 and .212; vs adele +0.105 [-0.008, +0.249] and +0.085 [+0.005, +0.202]), so on SOPBench free
request counts recover most of what the LLM plan and compiled code add. On tau2 they add nothing.
