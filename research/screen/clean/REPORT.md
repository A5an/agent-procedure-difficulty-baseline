# SOPBench with empty-reply configurations removed: does anything change?

Date 4 Oct 2026. No LLM calls. Everything is in this folder; the repository and other screen folders were not written.
"Old" = the original 28 agents. A = three configurations dropped (25 agents). B = 28 agents, every mostly-empty case removed.

## 1. Cleaning

Per-case empty shares recomputed from the raw run files (`s01_empty.py`, `empty_cases.csv`; assistant message = dict without `tool_call_id`, sender not "user"; empty = content in ("", "None") and tool_calls in ("None","[]",""); mostly empty = at least half of the assistant messages empty). Shares reproduce the report: gpt-5 89% (738 of 830 cases), gpt-5-mini 47% (390), gemini-2.5-pro 28% (231). The other agents with such cases: gemma-4-e2b 36, gemma-4-e4b 33, qwen3.5-4b 11, qwen3.5-2b 10, gemini-2.0-flash-thinking 1 (all under 5%).

| variant | agents | cells removed | cells left (of 23,235) |
|---|---|---|---|
| A | 25 | 2,490 (3 x 830) | 20,745 |
| B | 28 | 1,450 (1,359 from the three reasoning models, 91 from the five others) | 21,785 |

Layout: `data_A/sopbench/` and `data_B/sopbench/` (responses.jsonl changed; tasks.csv, statements.jsonl, features/, adele_responses.jsonl, human_time.jsonl, label_check.json copied unchanged; `irt/1d_1pl/` refit). `tau2` and `tauk_banking` are symlinks to the repository data. Note: `tasks.csv` keeps its `success_rate_28` column (analysis only, not used by any method).

## 2. Full-data 1PL IRT (same call as `src/fit_irt.py`, seed 0, `s03_irt_cmp.py`)

| | A | B |
|---|---|---|
| Spearman, new vs old task difficulty (830 tasks) | 0.990 | 0.995 |
| Spearman, new vs old agent ability (agents in common) | 0.998 | 0.993 |
| within-domain Spearman of difficulties, range over 7 domains | 0.977 to 0.994 | 0.985 to 0.999 |

The target the protocol evaluates against barely moves. Whatever the empty replies did, they did not reshuffle task difficulty.

## 3. Baseline protocol on the cleaned data

`run_clean.py` redirects `run_baseline.DATA`, `evaluate.DATA` and `run_baseline.RESULTS`; fold IRT is refit and cached in `results_<v>/sopbench/irt_splits`. tau2 obs are unchanged and merged column-wise from the repository results and from the earlier screening runs (rows checked identical). The "old" rows are re-evaluated with the same code on the original obs (they reproduce 0.131, 0.128, 0.172, 0.182 of the earlier reports). Methods: baseline `adele`; `length`, `human_time`; shares target (`shares_adele`, `bb_adele`, from `targets/shares.py`); `lad2_adele` and `grp_proc_adele` (code and cached features from `rubric/`). Variant A: all. Variant B: baseline, length, human_time only (the rest not run; B is cheap, but the brief asked for baseline only).

Pooled over the four new-process scenarios. The paired CI is against the baseline of the same data version. No cleaned or old row passes criterion 1 of the improvement rule (the paired interval must lie above zero): every method is "not better" in every version.

| method | data | pooled rho | 95% CI | worst | sop-proc / sop-dom / tau2-proc / tau2-dom | paired CI vs baseline of the same data | Brier skill | log-loss skill |
|---|---|---|---|---|---|---|---|---|
| adele | 28 agents | +0.131 | [+0.027, +0.255] | +0.120 | +0.120 / +0.125 / +0.128 / +0.150 | - | +0.000 | -0.003 |
| adele | A: 25 agents | +0.134 | [+0.027, +0.258] | +0.127 | +0.127 / +0.131 / +0.128 / +0.150 | - | +0.003 | -0.001 |
| adele | B: 28, cells removed | +0.131 | [+0.020, +0.250] | +0.113 | +0.113 / +0.134 / +0.128 / +0.150 | - | +0.000 | -0.003 |
| length | 28 agents | +0.043 | [-0.187, +0.230] | -0.051 | -0.020 / +0.032 / -0.051 / +0.211 | [-0.318, +0.046] | -0.048 | -0.038 |
| length | A: 25 agents | +0.042 | [-0.187, +0.227] | -0.051 | -0.018 / +0.026 / -0.051 / +0.211 | [-0.319, +0.042] | -0.044 | -0.034 |
| length | B: 28, cells removed | +0.052 | [-0.184, +0.223] | -0.051 | +0.008 / +0.040 / -0.051 / +0.211 | [-0.311, +0.047] | -0.047 | -0.037 |
| human_time | 28 agents | +0.107 | [-0.002, +0.283] | +0.079 | +0.093 / +0.131 / +0.079 / +0.124 | [-0.131, +0.108] | -0.006 | -0.007 |
| human_time | A: 25 agents | +0.105 | [-0.002, +0.277] | +0.079 | +0.089 / +0.126 / +0.079 / +0.124 | [-0.136, +0.103] | -0.003 | -0.003 |
| human_time | B: 28, cells removed | +0.087 | [-0.017, +0.245] | +0.062 | +0.083 / +0.062 / +0.079 / +0.124 | [-0.139, +0.078] | -0.006 | -0.007 |
| shares_adele | 28 agents | +0.128 | [+0.025, +0.251] | +0.112 | +0.120 / +0.123 / +0.112 / +0.155 | [-0.009, +0.005] | +0.033 | +0.046 |
| shares_adele | A: 25 agents | +0.131 | [+0.030, +0.255] | +0.112 | +0.129 / +0.128 / +0.112 / +0.155 | [-0.009, +0.007] | +0.034 | +0.047 |
| bb_adele | 28 agents | +0.127 | [+0.029, +0.251] | +0.108 | +0.120 / +0.126 / +0.108 / +0.153 | [-0.008, +0.008] | +0.035 | +0.050 |
| bb_adele | A: 25 agents | +0.129 | [+0.030, +0.254] | +0.108 | +0.129 / +0.125 / +0.108 / +0.153 | [-0.012, +0.007] | +0.035 | +0.050 |
| lad2_adele | 28 agents | +0.182 | [+0.046, +0.270] | +0.168 | +0.181 / +0.179 / +0.168 / +0.199 | [-0.061, +0.101] | +0.059 | +0.076 |
| lad2_adele | A: 25 agents | +0.177 | [+0.040, +0.269] | +0.167 | +0.173 / +0.167 / +0.168 / +0.199 | [-0.072, +0.097] | +0.059 | +0.075 |
| grp_proc_adele | 28 agents | +0.172 | [+0.053, +0.269] | +0.133 | +0.194 / +0.179 / +0.181 / +0.133 | [-0.060, +0.091] | -0.002 | -0.020 |
| grp_proc_adele | A: 25 agents | +0.162 | [+0.044, +0.258] | +0.133 | +0.179 / +0.155 / +0.181 / +0.133 | [-0.074, +0.083] | -0.003 | -0.020 |
| shares_adele+adele | 28 agents | +0.129 | [+0.029, +0.253] | +0.122 | +0.122 / +0.124 / +0.123 / +0.149 | [-0.004, +0.005] | +0.019 | +0.026 |
| shares_adele+adele | A: 25 agents | +0.133 | [+0.031, +0.257] | +0.123 | +0.128 / +0.132 / +0.123 / +0.149 | [-0.004, +0.005] | +0.021 | +0.028 |
| lad2_adele+adele | 28 agents | +0.162 | [+0.058, +0.269] | +0.140 | +0.173 / +0.182 / +0.153 / +0.140 | [-0.036, +0.077] | +0.046 | +0.060 |
| lad2_adele+adele | A: 25 agents | +0.158 | [+0.052, +0.265] | +0.140 | +0.168 / +0.170 / +0.153 / +0.140 | [-0.045, +0.070] | +0.046 | +0.060 |
| grp_proc_adele+adele | 28 agents | +0.165 | [+0.054, +0.269] | +0.123 | +0.189 / +0.184 / +0.166 / +0.123 | [-0.046, +0.087] | +0.013 | +0.005 |
| grp_proc_adele+adele | A: 25 agents | +0.157 | [+0.045, +0.265] | +0.123 | +0.178 / +0.163 / +0.166 / +0.123 | [-0.059, +0.080] | +0.012 | +0.005 |

Reading: the baseline moves from 0.131 to 0.134 (A) and 0.131 (B); the SOPBench scenarios of the baseline go 0.120/0.125 to 0.127/0.131 (A) and 0.113/0.134 (B). All CI limits shift by 0.00 to 0.01. The best rubric variants lose a little on SOPBench in A (lad2_adele 0.181/0.179 to 0.173/0.167, grp_proc_adele 0.194/0.179 to 0.179/0.155) and their gain over the baseline shrinks from +0.051/+0.041 to +0.043/+0.028, intervals still containing zero. The shares target equals the baseline in both versions (differences within +-0.012). `human_time` falls a bit in B (0.107 to 0.087), `length` is unchanged.

## 4. Rechecks

### (i) carry-out vs refuse (`s04_checks.py`, `checks_i_ii.json`)
Spearman between agents' success rate on carry-out cases (should_succeed = 1) and on refuse cases:

| | old (28) | A (25) | B (28) |
|---|---|---|---|
| rank correlation | 0.821 | 0.914 | 0.889 |
| mean success, carry-out / refuse | 0.307 / 0.549 | 0.302 / 0.509 | 0.332 / 0.533 |

The expected 0.91 is the cleaned value (A). With the empty-reply agents in, it is 0.82. The empty replies are what made the two halves disagree; mean refuse-case success drops by 0.04 when they are removed.

### (ii) between-domain agent rank correlations (7 domains, 21 pairs)

| | old | A | B |
|---|---|---|---|
| mean | 0.732 | 0.696 | 0.704 |
| min / max | 0.490 / 0.908 | 0.384 / 0.902 | 0.393 / 0.894 |

Between-domain agreement does not improve by cleaning; it goes down slightly (mean -0.04). I did not investigate why. Pair-level values in `checks_i_ii.json`.

### (iii) held-out-cell check, section A4 of `checks/E_irt_ideas.md` (SOPBench only, `s05_a4.py`, copy of the scratchpad script with the data root as argument; same folds seed, 5-fold random cell holdout, L-BFGS, one seed). The old column reproduces the table of the plan exactly.

| model | old logloss / AUC | A | B |
|---|---|---|---|
| rates | 0.4713 / 0.8603 | 0.4739 / 0.8553 | 0.4734 / 0.8569 |
| 1PL | 0.4675 / 0.8605 | 0.4712 / 0.8558 | 0.4704 / 0.8575 |
| 2PL | 0.4670 / 0.8613 | 0.4711 / 0.8561 | 0.4694 / 0.8582 |
| agent slope | 0.4627 / 0.8658 | 0.4706 / 0.8615 | 0.4689 / 0.8626 |
| rank-2 interaction | 0.4460 / 0.8749 | 0.4541 / 0.8684 | 0.4489 / 0.8716 |
| rank-2 minus 1PL | -0.0215 / +0.0144 | -0.0171 / +0.0126 | -0.0215 / +0.0141 |

The interaction gain survives: rank-2 beats 1PL by 0.017 logloss and 0.013 AUC in A (0.022 and 0.014 in old and B). So agent x task structure beyond one ability is not an artifact of the empty replies. The agent-slope gain shrinks to almost nothing in A (0.0006 logloss). Single seed, no intervals, as before.

### (iv) offline routing (`s06_pertype_clean.py`, reusing `pertype/pertype_models.py` and `route()` of `pertype/s06_routing.py`; SOPBench only; logit_all is the declared per-type model)
Mean realised success of the routed choice, bootstrap over procedures; "adele" = one ability per agent.

| scenario | strategy | old | A |
|---|---|---|---|
| new procedures | oracle (per task) | 0.952 | 0.947 |
| | one ability per agent (adele) | 0.735 | 0.753 |
| | best agent per domain from training | 0.759 | 0.753 |
| | logit_all | 0.766 | 0.776 |
| | logit_all minus adele, paired CI | +0.031 [+0.001, +0.058] | +0.023 [-0.004, +0.050] |
| | logit_all minus best per domain, paired CI | +0.007 [-0.025, +0.037] | +0.022 [-0.009, +0.051] |
| new domain | oracle | 0.952 | 0.947 |
| | one ability per agent (adele) | 0.659 | 0.673 |
| | logit_all | 0.695 | 0.730 |
| | logit_all minus adele, paired CI | +0.036 [-0.011, +0.082] | +0.057 [-0.003, +0.131] |

(The old logit_all new-domain value 0.695 is from `pertype/routing.json`; the 0.728 quoted in the pertype headline is the `logit_all_l30` variant, not rerun here.) Best agent per domain is undefined in the new-domain scheme. In A, one ability per agent always picks the same agent (the single best one), so it equals "single best from training". Routing with the per-type model is +0.02 to +0.06 above that, with every interval touching zero; in old data the new-procedures interval for logit_all minus adele just excluded zero, in A it does not. `share_dom` (a domain-share model): new procedures 0.775 (+0.022 over adele [-0.006, +0.050]) in A, old 0.790 (+0.055 [+0.019, +0.092]); its significance is lost too. The oracle gap stays about 0.17 to 0.27.

## 5. Verdict

1. The cleaning changes the data much more than the results. Task difficulties correlate 0.99 with the old ones; the baseline's pooled rho moves 0.131 to 0.134 (A) / 0.131 (B), with intervals shifting by at most 0.01.
2. No conclusion about methods flips. Under the improvement rule every method remains "not better" in old, A, and (for the three run) B. The rubric variants keep a point gain of +0.03 to +0.04 with paired intervals around [-0.07, +0.10]. The shares target stays equal to the baseline.
3. The check that changes is (i): carry-out and refuse rankings agree at 0.91 once the empty-reply agents are out (0.82 before), so the weaker carry-out versus refuse agreement seen with all 28 agents was partly produced by those three runs (I did not re-test the other pertype findings on flags).
4. The held-out-cell interaction (rank-2 over 1PL) survives with the same size. Per-type routing gains over one ability per agent shrink on new procedures (+0.031 to +0.023) and lose their significance there; on new domains the point gain grows (+0.036 to +0.057) with an interval that still reaches zero.
5. Choice between A and B: they agree within noise on every number. B keeps 28 agents but loses the cases where the three agents did anything at all (gpt-5 keeps only 92 cases), so its per-agent comparisons rest on a different case mix per agent; A is the cleaner comparison. I would report A and mention B as a robustness check.

## 6. Not done / caveats
- B was run for the baseline, length and human_time only; the per-type routing and the other recheck items except (i), (ii), (iii) were not run on B.
- Routing was run for SOPBench only (tau2 has no per-type tables), as in the earlier report; routing for the logit_all_l30 variant was not rerun.
- Seed 0 for the cleaned full-data IRT, as in the repository; fold IRT seeds and the 5 split seeds as in the baseline.

## 7. Files
`s01_empty.py` (empty shares, `empty_cases.csv`), `s02_build.py` (data_A, data_B), `s03_irt_cmp.py`, `run_clean.py` (run A|B / eval, old eval), `s04_checks.py`, `s05_a4.py` + `run_a4.sh` (`a4_{old,A,B}.json`), `s06_pertype_clean.py` (`routing_A.json`), `s07_table.py` (`table_main.md`). Evaluations: `results_old/eval.json`, `results_A/eval.json`, `results_B/eval.json` (and `.txt`). Logs: `runA.log`, `runB.log`, `pertypeA.log`, `a4.log`.
