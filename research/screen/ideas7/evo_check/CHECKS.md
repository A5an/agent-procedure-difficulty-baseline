# Clean checks of evo finalist F1 (agent evo_check, 6 Oct 2026). Written BEFORE any check was run.

Claim to test: c_lad + F1 (8 features) lifts the full protocol from 0.258 to 0.353, +0.095 [+0.034, +0.123] vs c_lad.
Known problems: about 70% of cases are in the search domains; most_diff_check uses the condition skip model, fitted on outcomes of other domains.
evo's files are not modified. All outputs go to this folder.

## Check 1. Leak check (no LLM calls)
Columns, all "c_lad ladder + F1 features, domain-centred" exactly as run_protocol.py builds evo_f1, same harness, same folds (sopbench and tau2, new_procedures and new_domain):
- f1_same: F1 with the stored feat_f1.csv. Sanity column, must reproduce evo_f1 in out_protocol.
- f1_nomdc: F1 with most_diff_check removed (7 features).
- f1_refit: F1 with most_diff_check recomputed inside each harness fold. The skip model (ideas3/conditions CondModel) is fitted only on outcomes of the training procedures of that fold. Test cases get the model fitted on all training cases of the fold. Training cases get out-of-fold values (inner 5-fold grouped by procedure), so train and test feature distributions are comparable. Same formula as in F1 (max over conditions of p_skip x type weight x position weight). Domain-centred on the union of train and test cases.
tau2 has no conditions, so most_diff_check is constant 0 there and tau2 columns should equal f1_same up to ridge noise (they are still rerun).
Score: score_protocol.py machinery (ideas7/evo), c_lad from ideas3/combo/out_main. Report pooled (4 scenarios), per scenario, and the paired bootstrap CI vs c_lad (bootstrap over procedures), for all, plus SOPBench procedure level as in score_protocol.
Pass rule fixed now: F1 survives the leak check if the pooled paired CI vs c_lad of f1_refit and of f1_nomdc stays above 0 and the point estimate keeps at least half of +0.095.

## Check 2. Fresh benchmark, tau-Knowledge banking (97 tasks, 12 models x 4 attempts, never used in the search)
Inputs built from documentation only, gemini-3.8-flash on Vertex, at most 600 calls, everything cached:
- scenario text = the banking statement (the simulated customer script, as for tau2).
- knowledge: policy_header.md + a compact index of all 698 knowledge-base documents (titles) + the 6 documents most similar to the scenario by TF-IDF (stand-in for what KB_search returns; no gold document lists from tasks.json are used).
- tools: the real agent toolkit from the tau2-bench clone (ideas3/plan_tools/src_tau2) base tools, schema dumped by code; discoverable tools are reached through unlock/call_discoverable_agent_tool, with their names visible in the documents.
- 3 plans per task (t0, s1, s2, prompt of ideas3/plan_tools/gen2.py adapted to this policy block) -> PLAN features by the same code as plan3_tau2.csv (feats3.py).
- scenario features (prompt of ideas4/user_scenario/gen.py; clause list replaced by the document-title index) -> scenario dict and clauses touched.
- dry-run JSON (main_prompt of ideas/dryrun/lib.py with the same status definitions and format; the case block carries the policy block above) -> dry-run columns by ideas/dryrun/build_features.py rules (decision samples are not made; entropy is not used by F1).
Budget plan: 97 x (3 plans + 1 scenario + 1 dry) = 485 calls plus retries, hard cap 600.
Features with SOPBench-only inputs (program, conditions, skip scores) are empty for banking, exactly as for tau2 records, so those features take the values the evo code gives to empty lists (zero), which after domain-centring is the banking mean.
Transfer test: c_lad and c_lad + F1 (same Ladder class, level 2) are trained on SOPBench + tau2 cells (agents prefixed per benchmark) and predict banking task difficulty. Common feature columns only: rubric (12), ADeLe columns shared by the three benchmarks, PLAN (9); the SOPBench-only CMP columns are dropped in this check for both c_lad and c_lad+F1 (said openly: this is c_lad without CMP). F1 columns are domain-centred per domain (banking by its own mean).
Metric: Spearman of predicted difficulty vs banking IRT difficulty b (data/tauk_banking/irt/1d_1pl/items.csv), 97 tasks, bootstrap over tasks (2000 draws), paired difference CI between c_lad+F1 and c_lad. Secondary, fixed now: same two columns trained on tau2 only; and the F1 features alone (no c_lad) as a diagnostic.
Pass rule fixed now: survives if the paired CI of c_lad+F1 minus c_lad is above 0. Point estimate with a CI through 0 is reported as not confirmed.
Note: banking has no IRT scheme inside the harness (single domain, one procedure per task), so no harness run is possible; this custom transfer is the test.

## Check 3. Smart-agent target (evaluation only)
Target = per SOPBench case mean failure of Gemini 3.8 Flash and Sonnet 5.5 (ideas5/smart_gemini and smart_sonnet real_results.csv, 1 - success). Predictions: out-of-fold beta_hat of evo_f1 (out_protocol) and c_lad (combo/out_main), averaged over seeds, both schemes separately. Case level: within-domain Spearman averaged over domains; procedure level: within-domain Spearman of procedure means. Reported for search domains (bank, dmv, healthcare, hotel) and held-out domains (library, online_market, university) separately, with and without the two broken bank procedures (main numbers without them), paired bootstrap over procedures vs c_lad.
Pass rule fixed now: survives if the held-out-domain difference to c_lad (case level) has a CI above 0, otherwise reported as flat or negative.

## Verdict format
Per check: survives / does not survive / inconclusive, with numbers.
