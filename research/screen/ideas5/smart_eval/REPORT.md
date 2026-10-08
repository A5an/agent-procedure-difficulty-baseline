# Smart-agent target and smart-agent emulation (main session, 6 Oct 2026)

Inputs: gemini-3.8-flash (H0, Vertex, ideas5/smart_gemini, 35.5 USD) and Claude Sonnet 5.5 (Claude Code headless, bridge,
ideas5/smart_sonnet, subscription, about 183 USD equivalent) on all 830 SOPBench cases and all 1,365 synthetic cases of
emulate_run/synth_all. Real success: Gemini 0.849, Sonnet 0.766 (case agreement Spearman 0.667). Synthetic: 0.856, 0.799.
Smart target per case = mean failure of the two. The two broken bank procedures (target tool missing) excluded (68 left).

## Existing documentation predictors vs the smart target (prelim2.py, prelim_smart2.json; obs files out-of-fold)
| method | case new_proc / new_dom | procedure new_proc / new_dom | procedure vs public IRT |
|---|---|---|---|
| adele | .034 / .066 | .050 / .053 | .229 / .231 |
| lad2_adele | .088 / .148 | .228 / .238 | .389 / .411 |
| c_lad | .098 / .134 | .351 / .336 | .448 / .486 |
| c_lad_cmpp | .143 / .215 | .341 / .370 | .456 / .503 |
Public IRT difficulty vs smart failure: case .464, procedure .528. (Two broken procedures not excluded in this table.)

## Emulation with smart agents (emu_smart.py, emu_smart.json, emu_smart_proc.csv)
Per procedure estimate = 0.35 x perform-class failure + 0.65 x refuse-class failure on synthetic cases. Pair-weighted
within-domain Spearman, 1000-draw bootstrap over procedures within domain.
| predictor | vs smart real failure | vs Gemini real | vs Sonnet real | vs public IRT |
|---|---|---|---|---|
| Gemini synthetic | .756 [.60, .90] | .817 [.69, .94] | .703 [.52, .85] | .566 [.37, .73] |
| Sonnet synthetic | .749 [.58, .87] | .624 [.41, .78] | .762 [.59, .87] | .478 [.24, .65] |
| both | .792 [.62, .91] | .699 [.50, .84] | .778 [.60, .91] | .523 [.29, .70] |

## Case level vs smart failure (emu_case.py, no CIs)
emulation procedure estimate .357; emulation + compiled pre-check class (L2, record read by code) .466; emulation + true
label (oracle) .555; pre-check alone .281.

## Reading
For smart agents, emulation on LLM-generated cases ranks procedures at about 0.8, documentation-only c_lad at about 0.35.
Cost of the Gemini emulation: 22.5 USD for 1,365 synthetic episodes plus 4.5 USD generation, about 0.40 USD per procedure.
Friendly test: SOPBench's own cases were LLM-generated from the same kind of policy and schema. One run per case.
