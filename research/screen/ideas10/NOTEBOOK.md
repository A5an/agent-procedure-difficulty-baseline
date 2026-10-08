# Round 10 lab notebook (night 7-8 Oct 2026)

## Ceilings (ceil.py, ceil2.py, ceil3.py; full-data b, within-domain Spearman pooled by pairs)
- SOPBench: procedure-mean oracle 0.231; label oracle 0.520; procedure x label oracle 0.614.
  86% of within-domain variance is within procedure, 47% within procedure x label.
- Policy text varies inside a procedure (405 distinct policy texts for 830 cases; 228 constraint sets). An identical-
  policy-text oracle looks like 0.68 but that is an artifact of 187 singleton texts; the leave-one-out version is about 0.
- tau2: procedure-mean oracle 0.852, identical-text oracle 0.895 (telecom 0.771: 5 texts for 114 cases).
- Visible label cues on SOPBench: random mixed-case credentials (>= 4 case switches in a 10+ char token) flag 74 cases,
  70 of them must-refuse, mean b -2.90 vs -0.28 for other refuse cases and +1.64 for perform cases (labelvis.py).
- Conclusion: pooled case-level 0.6 at L0/L1 is out of reach on these benchmarks; SOPBench is capped by the hidden label.

## Per-domain look at the current best (c_lad + GAP), tau2 (resid_tau2.py, tau2scan.py)
- new_domain: airline 0.553, retail 0.312, telecom 0.505. new_procedures: airline 0.461, retail 0.236, telecom -0.404
  (telecom procedures are alone in their fold cells, so the cell rho rests on home vs abroad texts).
- Retail: statement length alone 0.44 beats the model (0.31). Strongest single features across all three domains:
  SCN n_clauses (0.62 / 0.33 / 0.69), needs_computation (0.34 / 0.35 / 0.44), GAP n_args (0.30 / 0.34 / 0.55).
- Gold action count (analysis only): airline 0.51, retail 0.07, telecom 0.69.

## Literature (lit/REPORT.md)
- Text-only agent difficulty: Krsteski and Meyer 0.399 K-fold, 0.225 leave-one-benchmark-out; Messier within-benchmark
  z-scored 0.26 to 0.30. Results >= 0.6 come from education with an answer key and many simulated respondents.

## Experiment prior (prior/PREREG.md): what a policy-blind assistant would do vs what the policy requires
- Running: 6 naive plans + 1 policy plan + 1 matching call per case, about 9.6k Vertex calls.

## Experiment fc (fc/PREREG.md): pre-mortem by Sonnet 5.5 (claude -p, no tools), 1 call per case.

## Analysis of what makes SOPBench cases hard inside a procedure (execdrivers.py; uses gold fields, analysis only)
- Label-conditional rho of current models is weak: c_lad + GAP ranks must-perform cases at 0.07 to 0.16 and must-refuse
  cases at 0.23 to 0.29 (condrho.py).
- Must-perform cases: the number of values the customer knows (user_known) has within-procedure rho -0.71; visible in the
  message as "gives a credential" (cred_word -0.68), "offers an admin password" (-0.63).
- Policy variants that do NOT mention login or credentials are much harder inside a procedure: rho -0.34 over all cases,
  -0.69 on must-perform, -0.28 on must-refuse. Mean b: perform without login rule 3.20 vs 0.82 with; refuse 0.05 vs -0.91.
  Only bank and dmv have both kinds of variants (hotel has no login rules at all).
- Reading: agents apply default checks the policy does not ask for (authentication), the customer cannot satisfy them,
  and the agent refuses a case that should be performed. 56 cases have success <= 4% (53 must-perform, b about 4 to 7):
  credential gaps, verification tools that themselves need login, time comparisons, a crashing tool.
- Pre-registered as a POST HOC family OVER (over/PREREG.md) computed from the prior run's plans.

## Result prior (prior_score_main.txt): NEGATIVE
- pr2 (c_lad + GAP + PRIOR) 0.305 vs g1 0.311, -0.006 [-0.025, +0.015]; pr1 (c_lad + PRIOR) 0.243 vs c_lad 0.258.
  PRIOR hurts tau2 in the model (0.251 / 0.253 vs 0.298 / 0.285) although raw features correlate +0.35 on tau2.
- Procedure level SOPBench new domain: pr2 0.463 vs g1 0.482. Banking raw signs mostly positive (0.2 to 0.25).
- OVER raw scan: ov_auth_extra within-procedure -0.36 (opposite to the hypothesis). The naive plans do not reproduce the
  login mechanism: when the policy has no login rule the customer also gives no credential, so the naive plan skips login.

## Result fc (fc_score_main.txt): POSITIVE on tau2, nothing on SOPBench
- f2 = c_lad + GAP + SCN + PRIOR + FC: pooled 0.347 [0.231, 0.384]; vs g1 +0.036 [+0.005, +0.051]; vs c1_all +0.029
  [+0.004, +0.052]; vs baseline +0.216 [+0.126, +0.246]. By scenario .242 .193 .421 .532.
- f1 = c_lad + GAP + FC: 0.337, vs g1 +0.026 [-0.002, +0.045] (the pre-registered primary narrowly misses).
- tau2 new domain: f2 0.532 (vs g1 +0.111 [+0.025, +0.169]), f1 0.506.
- Zero-shot (raw fc_diff, nothing fitted), within-domain: tau2 0.517 [0.289, 0.622] (airline 0.71, retail 0.40,
  telecom 0.60), banking 0.596 [0.468, 0.700] (pre-registered fresh check; ties with length 0.532 and LLM human-time
  0.582, differences n.s.), SOPBench 0.057.
- Cost: 1,205 Sonnet calls via claude -p, about 34 USD equivalent on the subscription.

## Result lobo (lobo_result.json): NEGATIVE
- Every many-feature ridge trained on two benchmarks transfers worse to banking than length (L_doc_fc 0.441, L_doc 0.401,
  length 0.532, human_time 0.582). A single zero-shot score transfers better than a fitted combination.

## Launched fcx (fcx/PREREG.md): pre-mortem ensemble over 2 prompts x 2 model families.

## Results zs and over: NEGATIVE
- zs (within-domain z-scoring of all inputs): z1 0.280 vs g1 0.311 (-0.032 [-0.085, -0.008]); z0 0.223 vs c_lad 0.258.
  c1_all (c_lad + GAP + SCN + PRIOR, plain) 0.318, +0.007 over g1, n.s. Z-scoring hurts on both benchmarks.
- over (post hoc OVER family): o1 0.294, o2 0.299, both slightly below g1. Not pursued.

## Scout (scout/REPORT.md): no fresh customer-service benchmark with outcomes. Fresh tasks with many agents:
  TheAgentCompany (175 tasks, 17 agents), MCPMark (127 tasks, 15 agents), DrafterBench (1,920 cases, 30 agents).
  AgentSuite has 30 new agents on the same tau2 tasks (independent agent population, same tasks).
## Launched fresh/ (fresh/PREREG.md): zero-shot generic pre-mortem (Sonnet, primary), Gemini version, length and
  LLM human-time on TAC, MCPMark and a 240-case DrafterBench sample.

## fcx partial (tau2 and banking; SOPBench Sonnet P2 still running), raw within-domain, nothing fitted
| source | tau2 | banking |
|---|---|---|
| S1 Sonnet P1 (fc/) | 0.520 [0.290, 0.629] | 0.596 [0.469, 0.700] |
| S2 Sonnet P2 (QA-lead prompt) | 0.460 | 0.628 [0.517, 0.714] |
| G1 Gemini P1 | 0.108 | 0.387 |
| G2 Gemini P2 | 0.494 | 0.616 |
| FCX (mean of 4 z-scores, pre-registered) | 0.483 | **0.650 [0.537, 0.740]** (vs length +0.117 [-0.039, +0.296]) |
| S12 (two Sonnet prompts) | 0.493 | 0.646 |
| length / LLM human time | 0.211 / 0.124 | 0.532 / 0.582 |
H1 of fcx/PREREG.md: holds on banking (0.650 > 0.596), not on tau2 (0.483 < 0.520; the Gemini P1 source is weak there).

## newpop/ : independent agent population on the same tau2 tasks (AgentSuite, 30 models of 2025, 1 trial each)
- Old 15-agent b vs new-population difficulty (1 - success rate): within-domain 0.776 (airline .86, retail .72, telecom .82).
- Zero-shot pre-mortem vs the NEW population: S1 0.634 (.73 .42 .82), G2 0.659, FCX 0.629, S12 0.622;
  length 0.372, LLM human time 0.252. Nothing was fitted; the prompts were written before any tau2 result.

## fresh/ round 1 (pre-registered; primary FCg = Sonnet generic pre-mortem, zero-shot), 1 - mean success as target
| benchmark | FCg (Sonnet P1) | GG (Gemini P1) | FCg+GG | LLM human time | length |
|---|---|---|---|---|---|
| TheAgentCompany, 175 tasks, 17 agents | **0.458 [0.329, 0.579]** | 0.252 | 0.399 | 0.444 | 0.179 |
| MCPMark, 126 tasks, 14 agents | **0.272 [0.089, 0.432]** | 0.347 | 0.355 | -0.059 | -0.101 |
FCg beats length on both (TAC +0.279 [+0.132, +0.434], MCPMark +0.373 [+0.180, +0.564]); vs LLM human time: tie on TAC
(+0.014), clearly better on MCPMark (+0.331 [+0.148, +0.509]). Within TAC categories 0.453, within MCPMark services 0.325.
DrafterBench sample still running.

## Zero-shot S1 under the exact protocol cells (rawproto.py): sop .012 / .057, tau2 .411 / .517, pool4 0.249.
The trained best f2 has tau2 .421 / .532: one Sonnet call per task with nothing fitted matches the trained stack on tau2.

## fresh/ DrafterBench (240 sampled cases, 30 agents): FCg 0.231 [0.116, 0.344], GG 0.269, FCg+GG 0.279, HT 0.130,
length 0.026. FCg beats length on all three fresh benchmarks.

## newpop/eval_trained.py: trained out-of-fold predictions vs the independent 30-agent population (seed 0 cells)
- new_domain: f2 0.678 [0.375, 0.790] vs baseline 0.265 (+0.413 [+0.044, +0.677]); vs g1 +0.106 [+0.025, +0.170].
- new_procedures: f2 0.408, g1 0.476, baseline 0.205 (telecom cells rest on home vs abroad).

## f2 against the protocol rule (fc/brier.py): criteria 1, 2, 4 pass (Brier skill 0.069 / 0.067 / 0.050 / 0.076 vs
baseline 0.015 / 0.005 / 0.003 / -0.021); criterion 3 only via the universal proxy L_doc_fc (no held-out benchmark
with an interval below zero vs baseline); criterion 5 (blind HANDBOOK) open.

## fcx harness (fcx_score_main.txt): f2x = f2 with FCX in place of FC: pooled 0.354 [0.236, 0.395], vs f2 +0.007
[-0.011, +0.022], vs g1 +0.043 [+0.008, +0.059]; tau2 .446 / .515; SOPBench procedure level 0.444.

## fctype (fctype/PREREG.md): failure-mode clusters (K=12) on top of f2: 0.334, -0.013 [-0.052, +0.022]. NEGATIVE.
Cluster list in fctype/clusters.txt (skipped checks, wrong order, date checks, extra calls, telecom chains, etc.).

## fresh/ round 2 (pre-registered after round 1): ENS3 = mean z of Sonnet P1, Sonnet P2, Gemini P2; OP1 = Opus P1
| benchmark | FCg Sonnet P1 | SP2 | GP2 | ENS3 | Opus P1 | HT | length |
|---|---|---|---|---|---|---|---|
| TAC | 0.458 | 0.479 | 0.385 | 0.482 | **0.511 [0.393, 0.616]** | 0.444 | 0.179 |
| MCPMark | 0.250 | 0.343 | 0.253 | 0.307 | **0.371 [0.203, 0.528]** | -0.077 | -0.116 |
| DrafterBench | 0.231 | 0.233 | 0.290 | **0.299** | not run | 0.130 | 0.026 |
ENS3 vs FCg: +0.024 n.s., +0.058 n.s., +0.068 [+0.021, +0.117]. Opus vs FCg: +0.054 n.s., +0.121 [+0.010, +0.244].
Exploratory combinations (post hoc) stay at about 0.50 on TAC and 0.36 on MCPMark. No variant reaches 0.6 on fresh data.

## fco/ Opus 5.5 pre-mortem, prompt P1, tau2 + banking (exploratory, after Opus looked good on TAC)
- tau2 within-domain: Opus 0.407 [0.104, 0.658] vs Sonnet 0.517; rank average Opus + Sonnet 0.570 [0.363, 0.673].
- banking: Opus 0.619 [0.494, 0.710], Sonnet 0.596, average 0.633.
- new 30-agent population on tau2: Opus 0.577, Sonnet 0.629.
- A stronger forecaster is not uniformly better (worse on tau2, better on TAC and MCPMark); averaging two models helps.
- Partial correlation on fresh data (fresh/fresh_scores2.csv): pre-mortem given LLM human time and length keeps
  0.22 (TAC), 0.28 (MCPMark), 0.20 (DrafterBench); human time given pre-mortem 0.19, -0.07, 0.06.

## ablate/ (pre-registered): direct rating D vs pre-mortem P1, retest R
| benchmark | P1 | D (just p_success) | R (retest) | mean P1+R | retest rho |
|---|---|---|---|---|---|
| tau2 | 0.517 | **0.574** (+0.057 [-0.002, +0.116]) | 0.448 | 0.544 | 0.85 |
| banking | 0.596 | 0.531 (-0.066 n.s.) | 0.584 | 0.620 | 0.87 |
| TAC | 0.458 | 0.470 | 0.512 | 0.491 | 0.90 |
| MCPMark | 0.250 | 0.260 | 0.226 | 0.247 | 0.88 |
Hypothesis P1 > D NOT supported: the failure-listing framing does not matter on average; what matters is a strong model
asked about agents of a given generation under strict grading. D on tau2 under the protocol cells: new procedures 0.497,
new domain 0.574 (trained f2: 0.421 / 0.532). D against the new 30-agent population: 0.682 (airline .69, retail .49,
telecom .88). One Sonnet call, nothing fitted.

## fcd/ + zeroshot/: direct rating D on all benchmarks (added to ablate PREREG before the SOPBench calls)
- D zero-shot under the protocol (zeroshot/zero_score_main.txt, one Sonnet call per case, nothing fitted):
  pool4 0.294 [0.180, 0.350]; vs baseline +0.163 [+0.045, +0.238]; vs c_lad +0.036 [-0.064, +0.116]; vs c_lad + GAP
  -0.017 n.s. By scenario .022 .084 .497 .574; tau2 vs c_lad + GAP: +0.115 [+0.014, +0.212], +0.153 [+0.015, +0.318].
- Harness: f6 = f2 + D 0.351 (vs f2 +0.004 [-0.000, +0.012], primary not met); f7 = c_lad + GAP + D 0.333 (vs g1 +0.022
  [+0.008, +0.037]).
- DrafterBench D: see ablate cache (not evaluated separately).

## panel/ (pre-registered): direct question D to six models, equal-weight mean of z-scores
| model | tau2 | new 30 agents | banking | TAC | MCPMark | DrafterBench |
|---|---|---|---|---|---|---|
| Sonnet 5.5 | **0.574** | 0.682 | 0.531 | 0.470 | 0.260 | 0.215 |
| Opus 5.5 | 0.468 | 0.648 | **0.595** | 0.484 | **0.371** | not run |
| Gemini 2.5 Pro | 0.455 | 0.580 | 0.345 | 0.479 | 0.071 | 0.200 |
| Gemini 3.8 Flash | 0.493 | 0.639 | 0.557 | 0.344 | 0.266 | **0.311** |
| Gemini 3.5 Flash | -0.065 | -0.083 | 0.406 | 0.339 | 0.228 | 0.150 |
| Haiku 4.5 | 0.250 | 0.184 | 0.353 | 0.381 | 0.175 | 0.049 |
| PANEL (all) | 0.561 | **0.688** | **0.635** | **0.500** | 0.310 | 0.287 |
Primary (PANEL > Sonnet on TAC and MCPMark): +0.030 [-0.059, +0.123], +0.050 [-0.071, +0.176], NOT met. Weak forecasters
(Haiku, Gemini 3.5 Flash) dilute the panel; under the tau2 protocol cells the panel is 0.364 / 0.561 vs Sonnet 0.497 / 0.574.
The forecaster has to be strong; model size and family matter more than averaging.
