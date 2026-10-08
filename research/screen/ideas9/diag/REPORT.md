# Swarm agent `diag`, round 9 (6-7 Oct 2026)

Saved by the main session from the agent's final message.

# Round 9, agent `diag`: free diagnostic of every case-level feature table

No LLM calls, no GPU, no harness run. Pure CPU under `cpu_lock("diag")`, seeds fixed. Code and outputs are in `prototype/screen/ideas9/diag/`.

## 0. Findings

1. **Most of tau2 has no within-procedure variation.** Airline (50) and retail (114) cases are each their own procedure. Only telecom (114 cases) has repeated procedures, and only 3 of them (29, 36 and 49 cases). Banking has 97 cases in 97 procedures.
   - So "within-procedure" is a real test on SOPBench only (7 domains, 70 procedures, 830 cases).
   - On tau2 the within-procedure test is one domain with 3 procedures. For airline, retail and banking I report only raw within-domain rho.
2. **SOPBench within-procedure signal exists but is small, and it is one factor.**
   - Across the 361 SOPBench features with within-procedure variance, 144 beat the 95th percentile of the permutation null for the mean within-procedure rho. Chance would give about 18.
   - Per domain, 24% of tests beat the null, against 5% expected.
   - The best features reach |mean within rho| of 0.18 to 0.30. The noise ceiling is about 0.076 for the 7-domain mean and 0.14 to 0.30 for a single domain.
   - Almost all of it is "how likely is the hidden case to be refused or performed". A feature that holds up after centring on procedure × should_succeed label counts as non-proxy.
3. **No feature is a clean, cross-domain, label-independent winner.** Only three features beat the null in 5 or more of 7 SOPBench domains: `dry.refuse_share_sampled`, `emu.mix_pre`, `pers.n_act_wo_check`.
   - `emu.mix_pre` falls to |rho| 0.007 once label is controlled.
   - `dry.refuse_share_sampled` falls from -0.178 to -0.150.
   - `pers.n_act_wo_check` is the only one that keeps its size after label control (0.185 goes to 0.208), and it comes from an emulated-persona run.
4. **ADeLe carries essentially no within-procedure signal on SOPBench or tau2.**
   - SOPBench: the largest of 13 live scales is ad.QLq at +0.076, and it beats the null in 1 of 7 domains. Zero of the 13 ADeLe scales beat the mean-null. The grouped-CV ridge on all ADeLe scales gives within rho 0.016.
   - tau2: ADeLe is also about 0 in the ridge diagnostic (0.016 raw, 0.010 within).
   - This confirms the round-8 psychometric screen (|rho| 0.039).
5. **PLAN carries most of the usable signal, on both benchmarks.**
   - SOPBench within CV rho 0.145.
   - tau2 raw CV rho 0.237, and its single features reach 0.42 to 0.51.
6. **RUBRIC carries within-procedure signal on SOPBench, concentrated in two columns.** CV rho 0.154. It is driven by `p_refusal` and `needs_hidden_data`. On tau2 it is dead (-0.015).
7. **CMP is weak on SOPBench and absent on tau2.**
   - Within CV rho 0.011 for the two c_lad CMP features (`len_mean`, `branch_mean`), and 0.074 after label control.
   - The wider cmpp and cp code-structure features have a small positive tilt of +0.05 to +0.12. They are mostly under the null in individual domains, except `cmpp_depth` at 0.124 (0.189 within label).
8. **Removing ADeLe from c_lad improves its within-procedure CV rho.** c_lad-ADELE gets 0.169 against c_lad at 0.114 on SOPBench. On tau2 raw it gets 0.229 against 0.216. This is a cheap ridge with fixed alpha 10, not the harness. It suggests ADeLe is dilution on the case-level target, not information.

## 1. Tables used and skipped

**Included.** 475 numeric features were tested, 427 on SOPBench, 312 on tau2 and 71 on banking. Per-benchmark counts are `summary2.csv`.

| Source | Prefix |
|---|---|
| repo `adele.csv`, `length.csv`, `human_time.csv` | ad, len, ht |
| `rubric/features/rub_proc`, `rub_lara`, `rub_rpa`, `rub_len` | rubp, lara, rpa, rlen |
| `ideas3/plan_tools` plan3, plan3sd | plan3, plan3sd |
| `ideas3/plan_tau` | plan1 |
| `ideas6/planners` avg, son, int, dis | plAvg, plSon, plInt, plDis |
| `ideas3/compile_consistency` cons and `unit_features` (the latter averaged to procedure level) | cons, unit |
| `ideas4/code_plan` cmpplus, cp | cmpp, cp |
| `ideas4/user_scenario` scn_tau2 | scn |
| `ideas7/entropy_combo`, `small_model` | ent, sm |
| `ideas8/textcount`, `ideas8/surface` | txt, surf |
| `ideas7/adele_plus` A, B, C | adA, adB, adC |
| `ideas/dryrun`, `ideas/fmea`, `ideas/pairwise` | dry, fmea, bt |
| `ideas3/paper_pilot` pf_both | pilot |
| `ideas6/emu_feat`, `msg_facts`, `personas`, `synth_label` | emu, msg, pers, synl |
| `ideas7/case_signals` act, nli, ret, bnd | act, nli, ret, bnd |
| `ideas7/tau_graph` graph_avg (tau2 only) | grph |
| `ideas7/evo_check` bank_plan, bank_dry (banking only) | plan3, dry |

Notes on what was loaded:
- `ideas4/code_plan/cp_tau2.csv` is the only tau2 code-plan table.
- Banking has no `plan3_tauk_banking.csv`, so its plan features come from `evo_check/bank_plan.csv` and its dryrun features from `evo_check/bank_dry.csv`.
- Banking also has rub_proc, rub_lara, rub_rpa, rub_len, txt, adele, length, human_time (these are in the 71 banking features).

**Skipped.**
- `ideas/lupi` privileged tables, because they use gold information.
- `ideas7/evo` feat_f1/f2, because they are search-selected composites.
- `ideas3/conditions` conds and cond_feats, because they are condition-level, not case-level.
- `adele16`, `A0` (duplicates of adele), the `concat` folders (duplicates), and the `ideas*/out*/betas_*` result files (not features).
- The `ideas2/*` tables (steps, risk, outcomes), because they are agent outcomes, not features.
- `ideas6/personas/pers_t07_*`, `msg_features_v1` and `msg_features_partial*`, because they are partial subsets or older versions.
- The `.npz` embeddings, because they are not scalar features.

**Track tags.** `emu` (emulated agent runs) and `pers` (persona runs) are tagged L1+. They are mentioned as diagnostics but cannot be accepted as L0/L1 features.

## 2. Method

- **Target.** The full-data 1PL `b` (higher = harder), plus `procedure_id` and `domain` from `tasks.csv`. The `should_succeed` label is used for analysis only.
- **(a) Raw within-domain Spearman**, per domain and averaged (SOPBench 7, tau2 3, banking 1).
- **(b) Within-procedure Spearman.** Centre the feature and `b` by procedure mean, drop singleton procedures, then Spearman per domain and average. For SOPBench I also centre by procedure × label.
- **(c) Within-procedure share of feature variance**, on rows in procedures with at least 2 cases.
- **(d) Sign stability.** The number of domains with the same sign as the mean, and the worst-domain signed value.
- **(e) Two-level decomposition.** Statsmodels is not installed. I used the fixed-effects (within) estimator, which is the within component of the random-intercept model `b ~ x + (1 | procedure)`, with a cluster-robust z by procedure and x standardised.
  - For the between component I regress procedure means of `b` on procedure means of x.
  - Caveat: with 70 clusters of very unequal size the z is dominated by the large procedures (n up to 99), so it sometimes disagrees with the mean per-domain rho (the `txt.dig_*` features have mean within rho +0.12 and z near 0).
- **Zero-variance flag.** Within-procedure variance share below 1e-8 is flagged. Of 427 SOPBench features, 66 are flagged:
  - the 5 constant ADeLe scales ad.CL/KNn/KNs/MS/SNs, and the matching adB/adC columns,
  - rpa standardization, maturity, determinism, interfaces and stability, plus lara.D1,
  - `plan_transfer` in all planner variants,
  - the `emu.*` raw and fp/fr columns, `act.act_n_args`/`n_secret`/`n_enum`, `msg.no_facts`, and all 22 `unit.*` features (procedure-level by construction),
  - and others (`dry.ambiguous_policy`, `synl.sl_p_procmean`, and some `pilot.*` columns).
- **Permutation null.** `b` shuffled within procedure, 200 permutations, within-procedure rho recomputed per domain and as the mean over domains.
  - Median 95th percentile of |rho|: bank 0.167, dmv 0.200, healthcare 0.174, hotel 0.135, library 0.232, online_market 0.144, university 0.295. For the 7-domain mean it is 0.076 (`sop_nullm`). For telecom (tau2, 3 procedures) it is 0.159.
  - A feature is "beating the null" in a domain when |within rho| exceeds that feature's own 95th percentile there.
  - The null controls the size of a single test, not the 361 features tested. About 18 features would beat the null by chance at the mean level.

## 3. Case-level candidates (real within-procedure signal and stable sign)

**Rule** (fixed after seeing the data, applied once):
- at least 5 testable SOPBench domains;
- within-procedure variance share at least 0.05;
- |mean within rho| above the feature's mean-null 95th percentile;
- |cluster-robust z| at least 3.5;
- beats the per-domain null in at least 3 domains;
- sign agrees in all but one domain;
- worst domain better than -0.05.

Twelve rows pass, and they collapse to 4 distinct signals because the planner variants are near duplicates. Ranked by worst-domain signed rho:

| Rank | Feature (family, track) | mean within rho | within x label | FE z | agree / domains | beat null | worst dom. | var within | tau2 raw (sign) | tau2 telecom within | banking raw |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | emu.p_pre (EMULATION, L1+) | -0.300 | -0.027 | -6.5 | 7/7 | 4 | +0.056 | 0.82 | n/a | n/a | n/a |
| 2 | rubp.p_refusal = emu.p_rub (RUBRIC, L1) | -0.172 | -0.107 | -4.9 | 7/7 | 4 | +0.026 | 0.76 | -0.284 (same) | -0.017 | -0.128 (same) |
| 3 | pers.n_act_wo_check (PERSONAS, L1+) | +0.185 | +0.208 | +4.2 | 6/7 | 5 | -0.023 | 0.40 | +0.390 (same) | +0.342 | n/a |
| 4 | plan3.plan_refuse = plan3sd = plan1 = emu.p_plan (PLAN, L1) | -0.180 | -0.148 | -4.5 | 5/6 | 4 | -0.033 | 0.83 | -0.259 (same) | n/a | +0.019 (different) |
| 5 | plAvg.plan_refuse, plInt.plan_refuse | -0.198 | -0.134 | -4.8 | 5/6 | 4 | -0.036 | 0.83 | -0.27, -0.24 (same) | n/a | n/a |
| 6 | pers.dec_disagree (PERSONAS, L1+) | -0.195 | -0.151 | -5.1 | 5/6 | 4 | -0.046 | 0.84 | -0.042 (same) | n/a | n/a |
| 7 | emu.mix_plan (EMULATION, L1+) | +0.114 | +0.075 | +3.5 | 5/6 | 3 | -0.032 | 0.07 | n/a | n/a | n/a |

How to read the table:
- `emu.*` and `pers.*` come from emulated or persona agent runs, not documentation only. Treat them as reference, not candidates.
- Rows 2, 4 and 5 are one signal, the predicted refusal share. They keep roughly 60% to 80% of their size after label control (-0.172 to -0.107, -0.180 to -0.148). That is partial label proxying, not the full story.
- The only feature that keeps its full size after label control is `pers.n_act_wo_check`. It also has the same sign on all three tau2 domains (+0.248, +0.367, +0.554) and in the telecom within-procedure test, and it passes in 5 of 7 SOPBench domains. It is from emulated personas, so it is not documentation only.
- The only candidate that is L1 documentation-only, label-independent and has the same sign on tau2 and banking is rubp.p_refusal, and it only keeps 62% of its size after label control. Its tau2 within-procedure value is about 0 (-0.017), so the tau2 sign agreement comes from raw cross-procedure variation.
- Banking sign agreement is available only for a few features. Banking is one case per procedure, so banking raw is a between-procedure number and not a within-procedure test.

**Near misses** (L0 or L1, pass most criteria but fail one):
- `txt.cred_maxent` (L0): within -0.240, within x label -0.210, beats the null in 4 of 7 domains, 6/7 sign agreement, worst -0.019. Fails on the cluster z (-2.0).
- `sm.sm_msg_nll` (L1-local): within -0.212, within x label -0.211, z -4.7, 6/7 sign agreement, but only 3 domains beat the null. Fails the worst-domain rule (-0.074). It has the opposite sign on tau2 telecom (+0.353).
- `rubp.needs_hidden_data` (L1): within +0.213, within x label +0.198, z 4.3, but only 4 testable domains and 2 beat the null.
- `dry.refuse_share_sampled` (L1): within -0.178, within x label -0.150, z -5.4, beats the null in 5 of 7 domains, but worst domain -0.141.
- `cmpp.cmpp_depth` (L1): within +0.124, within x label +0.189, z 5.7, 6/7 sign agreement, beats the null in 2 domains only.
- `plan3.plan_reads`/`plAvg.plan_reads` (L1): within +0.137 to +0.161, within x label +0.170 to +0.211, 6/7 sign agreement, and tau2 raw +0.30 to +0.26. They beat the null in 3 domains.

## 4. Strong only between procedures (procedure-level candidates)

Rule: |mean per-domain rho on procedure means| at least 0.25, sign agrees in at least all but 2 of at least 5 domains, and not a case-level candidate. 41 features pass.

Procedure sample sizes per domain are only 6 to 14, so these rhos are noisy. The top of the list, strongest first:

| Feature | family, track | between rho | agree / domains | within rho | raw rho | tau2 raw | banking raw |
|---|---|---|---|---|---|---|---|
| pers.n_act_nocheck | PERSONAS, L1+ | -0.562 | 6/6 | -0.005 | -0.096 | -0.128 | n/a |
| adC.KNc | ADELE_PLUS, L1 | -0.468 | 5/5 | -0.055 | -0.070 | n/a | n/a |
| emu.mix_pre | EMULATION, L1+ | +0.448 | 5/6 | +0.251 | +0.253 | n/a | n/a |
| sm.sm_msg_nll | SMALLMODEL, L1-local | -0.433 | 6/6 | -0.212 | -0.238 | -0.232 | n/a |
| pilot.l_n_extra | PAPERPILOT, L1 | +0.412 | 5/6 | +0.062 | +0.098 | n/a | n/a |
| ad.KNf | ADELE, L1 | +0.358 | 5/6 | +0.070 | +0.104 | -0.017 | -0.129 |
| adA.KNs | ADELE_PLUS, L1 | -0.354 | 5/6 | -0.052 | -0.053 | +0.221 | n/a |
| ad.VO | ADELE, L1 | +0.324 | 6/6 | +0.036 | +0.075 | +0.071 | +0.471 |
| cmpp.cmpp_lines / cons.len_mean / cp.cp_lines | CMP, L1 | +0.299 to +0.328 | 4/6 | +0.048 to +0.088 | +0.156 to +0.172 | n/a or +0.125 | n/a |
| plan3.plan_reads, plan3.plan_calls | PLAN, L1 | +0.279 | 4/6 | +0.137 | +0.138 | +0.303 | +0.381 |
| rubp.n_lookup_document | RUBRIC, L1 | +0.280 | 4/6 | +0.135 | +0.123 | +0.010 | +0.396 |

- The ADeLe features carry between-procedure signal and no within-procedure signal. ad.VO, ad.KNf and the A, B and C variants are the examples, with within rho 0.04 to 0.07 and between rho 0.32 to 0.47. This matches the c_lad procedure-level jump (0.231 to 0.486).
- `cmpp_lines`, `cons.len_mean` and `cp_lines` are program-size features. They are procedure-level at 0.30 to 0.33 and nothing within.
- The `unit.*` features are procedure-level by construction and are not ranked here.

**tau2 raw, strongest first** (airline, retail and telecom are all cross-procedure, so this is a procedure-level measure):

| Feature | tau2 raw | air / ret / tel | SOPBench raw | SOPBench within |
|---|---|---|---|---|
| scn.n_clauses (SCN) | +0.547 | +0.62 / +0.33 / +0.69 | n/a | n/a |
| plInt.plan_facts (PLAN) | +0.507 | +0.61 / +0.27 / +0.64 | -0.158 | -0.177 |
| plAvg.plan_distinct_tools | +0.447 | +0.48 / +0.32 / +0.55 | +0.134 | +0.136 |
| plan3.plan_distinct_tools | +0.423 | +0.40 / +0.32 / +0.55 | +0.120 | +0.124 |
| plan3.plan_writes | +0.415 | +0.43 / +0.27 / +0.55 | +0.100 | +0.088 |
| grph.g_dw_writes (TAUGRAPH) | +0.411 | +0.49 / +0.21 / +0.53 | n/a | n/a |
| adA.QLl | +0.411 | +0.37 / +0.16 / +0.71 | +0.143 | +0.096 |

Several PLAN features agree on sign across all three tau2 domains and across SOPBench within-procedure (distinct_tools, writes, calls, reads), at 0.09 to 0.14 on SOPBench. `plan_facts` flips: positive on tau2 (+0.38 to +0.51) and negative on SOPBench (-0.12 to -0.18).

## 5. Dead on both

89 of the 475 features have |within rho| below the null on SOPBench, |between rho| below 0.15 and |tau2 raw| below 0.15 (or no tau2 value). By family:
- ADeLe: CEc, KNa, MCu.
- ADELE_PLUS: AT, KNn, MCr, MS, SNs and more.
- ENT (entropy) and FMEA: nearly all.
- PLAN_DIS (disagreement between planners): all 10 features.
- CMP: tool_pair, n_fields, cond_pair and more.
- MSGFACTS, SMALLMODEL (sm_tgt_lp, sm_margin, sm_ent), SYNTHLABEL (`sl_p_x_depth`), SURFACE (`comp_req`).
- RUBRIC_RPA: failure_rate, structuredness, resources and more. RUBRIC_LARA: D3, D4.
- PERSONAS: skip_mean, skip_gap.

The full per-feature list is in `summary2.csv` and `classify2.out`.

## 6. c_lad families: within-procedure signal

Grouped 5-fold ridge (alpha 10, no tuning, folds by procedure, 3 seeds averaged) on the features of each family, features and `b` centred by procedure. This is a diagnostic, not the harness. Rho is the mean over domains.

| Family (features) | SOPBench raw | SOPBench within | SOPBench within x label | tau2 raw | tau2 within (telecom) |
|---|---|---|---|---|---|
| ADELE (18) | 0.087 | 0.016 | -0.011 | 0.016 | 0.010 |
| RUBRIC (12) | 0.089 | 0.154 | 0.179 | -0.015 | 0.063 |
| PLAN (9) | 0.117 | 0.145 | 0.166 | 0.237 | 0.363 |
| CMP (2: len_mean, branch_mean) | 0.130 | 0.011 | 0.074 | n/a | n/a |
| c_lad (41 on SOP, 39 on tau2) | 0.181 | 0.114 | 0.106 | 0.216 | 0.137 |
| c_lad minus ADELE | 0.174 | 0.169 | 0.197 | 0.229 | 0.298 |
| c_lad minus RUBRIC | 0.191 | 0.117 | 0.090 | 0.138 | 0.088 |
| c_lad minus PLAN | 0.191 | 0.122 | 0.111 | 0.064 | 0.042 |
| c_lad minus CMP | 0.122 | 0.098 | 0.087 | 0.216 | 0.137 |

Other families for reference: SMALLMODEL (13) SOPBench within 0.186 but tau2 raw -0.228; TEXTCOUNT (5) 0.140 within and -0.314 raw on tau2; DRYRUN (11) 0.107 within and 0.363 on telecom; ENT 0.028.

Reading the table:
- **SOPBench within-procedure:** RUBRIC (0.154) and PLAN (0.145) carry it. ADeLe (0.016) and CMP (0.011) carry none. The within x label column does not shrink for RUBRIC (0.179), PLAN (0.166) and c_lad-ADELE (0.197), so those survive centring on label in this ridge, even though several of their single best features are label proxies.
- **tau2:** PLAN carries it (raw 0.237, telecom within 0.363). Removing PLAN from c_lad drops tau2 raw from 0.216 to 0.064. ADeLe and RUBRIC are near zero.
- **Caveat on tau2 within numbers.** The CV for telecom has 3 procedures spread across folds, so the training set for the test procedure is the other 2 procedures. Treat 0.363 and 0.298 as indicative, not estimates.
- **CMP in c_lad.** It has only two features in c_lad, neither is within-procedure informative, and its removal costs 0.06 raw on SOPBench (0.181 to 0.122), so its contribution is between-procedure.
- **The harness c_lad numbers (0.258 pooled) are not comparable to this ridge.**

## 7. Sign stability and transfer across domains

- **Within-procedure signs on SOPBench.** The 12 features that pass the rule agree in 5 to 7 of the 6 to 7 testable domains.
- **Cross-benchmark sign.** The refusal-share features (`plan_refuse`, `p_refusal`) share the SOPBench sign on tau2 raw (-0.24 to -0.28), but on banking `plan3.plan_refuse` has the opposite sign (+0.019, near zero).
- **PLAN counts transfer in sign.** `plan_calls`, `plan_reads`, `plan_writes` and `plan_distinct_tools` are positive on SOPBench within-procedure (+0.09 to +0.14), tau2 raw (+0.35 to +0.43) and banking raw (+0.33 to +0.48). They are modest within procedure (worst domain -0.006 to -0.095) and strong between procedures, so in a pooled model they help mostly by ranking procedures.
- **Features that do not transfer.** `sm.sm_msg_nll` flips on tau2 telecom (+0.353) and has the opposite sign on tau2 raw to what a pure SOPBench reading would predict. `txt.cred_maxent` is -0.22 on SOPBench and -0.11 on tau2 raw (same sign) but weak, and `txt.dig_*` are +0.12 to +0.19 SOPBench and +0.17 to +0.24 tau2 but only +0.12 within. `plan_facts` flips between SOPBench and tau2.

## 8. Verdict

1. **Case-level signal on SOPBench is a refusal-share signal.** The planner and rubric features that win are proxies for the hidden perform/refuse label, with 60% to 80% of their size surviving a label control. No documentation-only feature gives a clean, label-independent, cross-domain within-procedure signal larger than about 0.2.
2. **ADeLe cannot help the case-level metric.** Its within-procedure signal is at noise level (CV 0.016), and removing it improves the diagnostic fit.
3. **For case-level gains, look at PLAN and RUBRIC-style predicted-refusal and read/call counts.** For procedure-level gains, ADeLe, CMP size and PLAN counts all carry between-procedure signal. Round 8's `textcount` result (c_lad plus regex counts +0.005, n.s.) is consistent with this: those counts are largely between-procedure.

## 9. Limitations

- Within-procedure testing exists only on SOPBench. tau2 within is 3 procedures in telecom, and banking has none.
- 475 features were screened. The permutation null is per test, not family-wise, so about 18 of the 144 "beats the null" features at the mean level would be false positives by chance.
- Rho per domain uses 42 to 195 cases, so single-domain noise is wide (null 95th percentile up to 0.295 in university).
- The fixed-effects z is a cluster-robust approximation of the random-intercept slope, and statsmodels MixedLM was not available.
- Feature selection here used all data. Nothing was fed into a harness. If a harness run follows, pre-register one variant (c_lad minus ADELE, or c_lad plus `plan_refuse` and `p_refusal` at the procedure level), fix the rule above, and keep the selection out of the test folds.
- Several features share values (plan_refuse in 4 tables, rubp.p_refusal equals emu.p_rub). Counts of "candidates" overstate the number of independent signals.

## 10. Paths

All in `research/screen/ideas9/diag/`:
- `diag.py`: loader, within and between rho, permutation null, fixed-effects z (`res.pkl`, `log.pkl`).
- `summarize2.py` writes `summary2.csv` and `summary2.pkl`, the per-feature table with every column used above.
- `classify2.py` writes `classify2.out`, the candidate, procedure-level and dead lists.
- `extra.py`: family and label tables.
- `famcv.py` writes `famcv.pkl`, the family cross-validation.
