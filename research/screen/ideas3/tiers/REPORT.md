# Difficulty for strong and modern agents (agent `tiers`, round 3, 5 Oct 2026)

Saved by the main session from the agent's final message. No LLM calls.
Code: tiers.py, score.py, report.py, refit.py, modern.py, dump_modern.py. Results: report_existing.txt, report_refit.txt,
modern_results.json, modern_cases.json, tiers_sopbench.json, tiers_tau2.json, refit/<tier>/<bench>/obs_*.csv.gz.

## Setup
Targets per case within domain: all_irt (current), fail_all_clean, fail_top5/top8, fail_bot5/bot8 (tiers by overall
success, gpt-5, gpt-5-mini, gemini-2.5-pro excluded on SOPBench). SOPBench top5 solve 0.66 to 0.75 (o4-mini-high,
qwen3.5-4b, gemini-2.5-flash, gemini-2.0-flash-thinking, gpt-4.1), tau2 top5 0.85 to 0.89. On the 150-case sample: modern
targets mod_gem6 (6 Gemini arms), mod_cc2 (Claude Code native and guard), mod_all8. Existing out-of-fold predictions
scored without refit (predicted difficulty = -mean logit), 300 bootstrap draws over procedures stratified by domain.
Refit: fold IRT only on one tier, adele and lad2_adele refitted through harness.py.

## Case level, pooled rho [95% CI], existing predictions

| method | all_irt | fail_top5 | fail_top8 | fail_bot5 | fail_bot8 |
|---|---|---|---|---|---|
| adele | .131 [.06,.21] | .113 [.06,.18] | .121 [.08,.19] | .130 [.06,.22] | .125 [.05,.21] |
| lad2_adele | .182 [.09,.23] | .163 [.10,.22] | .172 [.12,.22] | .165 [.08,.23] | .149 [.05,.22] |
| grp_proc_adele | .172 | .136 | .143 | .151 | .140 |
| dry_cat_adele | .146 | .157 | .180 | .109 | .110 |
| dry_grp_adele | .189 [.11,.27] | .158 [.11,.22] | .173 [.12,.23] | .165 | .165 |
| lupi | .151 | .134 | .144 | .126 | .124 |
| human_time | .107 | .046 | .036 | .132 | .112 |
| length | .043 | .011 | .016 | .085 | .058 |

Full-data IRT oracle correlates only 0.72 to 0.79 with top-tier targets (SOPBench 0.80, tau2 0.64).

L2 precheck (SOPBench): all_irt .282 [.19,.36], fail_top5 .182 [.09,.24] (+.08 [+.03,+.14] over adele), top8 .204,
bot5 .228, bot8 .220. Label oracle: 0.51 on all_irt, 0.34 on top5. The hidden state matters least for the strongest agents.

## Procedure level, SOPBench (mean of the two schemes)

| method | all_irt | fail_top5 | fail_top8 | fail_bot5 | fail_bot8 |
|---|---|---|---|---|---|
| adele | .211 | .049 [-.09,.19] | .083 | .286 | .401 |
| lad2_adele | .392 | .188 [.04,.32] | .241 | .343 | .469 |
| grp_proc_adele | .424 | .311 [.14,.44] | .312 | .241 | .376 |
| lupi | .454 | .291 | .323 | .270 | .384 |
| dry_cat_adele | .390 | .148 | .233 | .302 | .468 |

Oracle ceiling for top5 procedure means only 0.68 (target noise). tau2 procedure level flat across tiers.

## Refit on the top tier (pooled case level)
adele refit does not help. lad2_adele refit top5: .187 [.13,.23] on fail_top5 (existing .163), .184 on all_irt.
Refit top5 lad2_adele beats refit adele by +0.07 [+0.02, +0.11] on fail_top5. Refit on bot5 hurts (.10 to .15).
Procedure level SOPBench: lad2_adele refit top5 .368 [.18,.50] on fail_top5 (existing .188), not paired, suggestive.

## 150-case sample: modern agents vs public difficulty (within-domain rho)

| modern target | all_irt | fail_top5 | fail_top8 | fail_bot5 | label (perform = hard) |
|---|---|---|---|---|---|
| mod_gem6 | .53 [.35,.66] | .58 [.41,.68] | .57 | .29 [.13,.43] | .19 |
| mod_cc2 | .36 [.11,.55] | .43 [.16,.61] | .40 | .22 | .12 [-.04,.33] |
| mod_all8 | .54 | .64 [.46,.73] | .61 | .31 | .18 |

Gemini and Claude Code arms agree with each other at 0.38 [0.09, 0.64]. Existing documentation predictions do not predict
modern per-case failures on these 150 cases: about 0 for Gemini arms, 0.15 to 0.22 for Claude Code arms (lad2_adele, dry),
intervals mostly include 0. The precheck does not predict modern failures (about -0.07 to 0).

## Verdict
1. Documentation methods rank strong-agent difficulty about as well as population difficulty at the case level, worse at
   the SOPBench procedure level (partly target noise). Refit on top-tier IRT recovers lad2_adele to 0.33 to 0.37 there.
2. Modern agents' failures follow top-tier public difficulty (0.43 to 0.64) more than weak-tier or population difficulty.
   The perform/refuse label explains little of them (0.05 to 0.19). No documentation method predicts them case by case
   on this sample.
3. If the target is the deployed strong agent, use the top5 or top8 IRT as target and train on it.

## Limitations
5-agent targets are coarse. 150-case analysis has 46 procedures and pools folds into domain cells. SOPBench "top" tier is
2025 models, only tau2 has modern frontier agents. Refit vs existing not formally paired. A stray file was briefly written
into the external SOPBench folder and moved back.
