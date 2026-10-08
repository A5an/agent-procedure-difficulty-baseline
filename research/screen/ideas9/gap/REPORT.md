# Swarm agent `gap`, round 9 (6-7 Oct 2026)

Saved by the main session from the agent's final message.

# Report: agent `gap`, round 9 (data gap, L1)

## Idea and level
- **Idea.** One cheap LLM call per case labels where each needed value comes from. The target action and the check tools get a source for every required argument: GIVEN, DERIVABLE, FETCH, ASK or MISSING. Every gating condition of the policy gets a source for its inputs: GIVEN, FETCH, ASK or UNKNOWN.
- **Level.** L1.
- **Origin.** Round-8 R1, R6, S2, A3, N08 and F1.
- **Result.** The primary hypothesis holds, but only through tau2. The gain on SOPBench is zero.

## Variants as pre-registered
`PREREG.md` was written before the full LLM run and before any harness run. The only look at outputs beforehand was the 10-case test.

- `g1_clad_gap`: c_lad plus all 13 GAP features. This is the primary hypothesis (g1 > c_lad on the pooled metric). I wrote "14" in PREREG.md, but the list has 13 features: 7 argument features and 6 condition features.
- `g2_adele_gap`: ADeLe plus the 13 GAP features.
- `g3_clad_cond`: c_lad plus the 6 condition-source features.
- `g4_clad_3`: c_lad plus n_fetch, frac_not_given and n_cond_fetch.
- Reference: `t0_clad`, built exactly like `ideas8/textcount/run.py`. It reproduced 0.258.

Setup:
- Model gemini-3.8-flash on Vertex, temperature 0, thinking low, JSON output.
- One prompt template for all benchmarks (`gap.py`, `INSTR`).
- Argument rows are de-duplicated by (tool, argument).
- One harness run, no reruns. Benches sopbench and tau2, schemes new_procedures and new_domain. The scorer is a copy of the textcount `score.py`.

Inputs per benchmark:
- **SOPBench.** Policy and customer message come from `statements.jsonl`. Tool schemas are the domain `actions` lists in the SOPBench source (`env/domains/<dom>/<dom>_assistant.py`), filtered to the statement's tool list.
- **tau2.** The request is the statement. The policy is the domain policy files; telecom gets main policy plus tech support manual. Tools come from `ideas3/plan_tools/tool_text.json`.
- **Banking.** The agent policy header and additional instructions, plus tool names only. The knowledge base is 2.8 MB (about 700 documents), so it was not given. Banking labels are therefore weaker. The prompt text did not change.

## Cost
- 1,089 LLM calls: 10 test plus 1,079 full run. The budget was 1,500.
- 1,205 cases gave 1,089 distinct prompts. There were 0 failed calls and 0 parse failures.
- No debugging reruns were needed. The only failure was a code bug (a missing `description` key in two online_market tools), fixed before any LLM call.

## Main table
All numbers are case-level rho. Intervals in brackets are 95% bootstrap intervals over procedures.

| variant | pooled4 | sop-proc | sop-dom | tau2-proc | tau2-dom | pooled vs adele | pooled vs c_lad |
|---|---|---|---|---|---|---|---|
| adele | 0.131 [.059,.206] | .120 | .125 | .128 | .150 | | -0.127 |
| t0_clad | 0.258 [.158,.335] | .237 | .212 | .298 | .285 | +0.127 [+.071,+.172] | |
| **g1_clad_gap** | **0.311** [.210,.363] | .248 | .194 | .382 | .421 | +0.180 [+.099,+.212] | **+0.053 [+.007,+.089]** |
| g2_adele_gap | 0.260 [.131,.311] | .212 | .166 | .312 | .351 | +0.129 [+.020,+.176] | +0.002 [-.087,+.062] |
| g3_clad_cond | 0.300 [.206,.361] | .235 | .201 | .368 | .394 | +0.169 [+.100,+.202] | +0.042 [+.011,+.070] |
| g4_clad_3 | 0.260 [.158,.332] | .242 | .205 | .307 | .284 | +0.129 [+.069,+.170] | +0.001 [-.009,+.010] |

Paired intervals vs c_lad per scenario:

| scenario | g1 | g3 |
|---|---|---|
| SOPBench new procedures | +0.011 [-.022,+.046] | -0.002 [-.010,+.013] |
| SOPBench new domain | -0.018 [-.064,+.030] | -0.012 [-.032,+.013] |
| tau2 new procedures | +0.084 [+.010,+.126] | +0.070 [+.024,+.099] |
| tau2 new domain | +0.136 [+.004,+.276] | +0.109 [+.012,+.210] |

SOPBench procedure level (new domain, mean per procedure):

| variant | rho | vs c_lad |
|---|---|---|
| adele | 0.231 | |
| t0_clad | 0.486 | |
| g1 | 0.482 | -0.004 [-.133,+.144] |
| g2 | 0.487 | +0.001 [-.123,+.150] |
| g3 | 0.526 | +0.040 [-.047,+.143] |
| g4 | 0.452 | -0.034 [-.095,+.041] |

Files: `gap_score_main.txt` and `score.log`. I did not run `metrics_extra`.

## Diagnostics
Files: `diag.txt`, `diag2.txt`.

**SOPBench within-procedure Spearman with b (higher b = harder):**
- n_cond_fetch +0.001, against the L2 diagnostic of +0.17.
- frac_not_given -0.025.
- n_cond_given +0.014, frac_cond_not_given -0.025.
- Best signals: n_cond_ask -0.172 (n=135 cases with variation) and n_ask -0.134. The sign is opposite to the hypothesis: more ASK labels means easier.
- Why the L2 signal disappears:
  - The LLM labels almost every gating condition FETCH (mean 3.54 of 3.81 conditions) and almost every argument GIVEN (6.71 of 7.28).
  - Both vary little within a procedure.
  - The L2 count of +0.17 came from the hidden record plus `user_known`. An L1 reading of the text does not recover it.
- Domain-mean rho for n_cond_fetch is +0.089 and for n_args +0.031.

**tau2 domain-mean rho with b:**
- n_args +0.39.
- n_fetch +0.36.
- n_cond_given +0.22.
- frac_cond_not_given -0.17.
- n_cond_fetch only +0.04. The signal is in the argument counts and in condition-given, not in the condition-fetch count that was the L2 hypothesis.

**tau2 per domain:** the gain is not carried by telecom alone, but telecom has the strongest correlations, as in earlier rounds.

| domain | n | n_args | n_fetch | n_given |
|---|---|---|---|---|
| airline | 50 | +0.30 | +0.32 | +0.05 |
| retail | 114 | +0.34 | +0.26 | -0.01 |
| telecom | 114 | +0.55 | +0.50 | +0.71 |

n_cond_given in telecom is +0.59, and frac_cond_not_given there is -0.45. Airline and retail show n_args and n_fetch around +0.3. Statement length gives +0.40 (airline), +0.44 (retail) and -0.06 (telecom), so the GAP features are not simple length copies.

**Overlap with PLAN features:**
- SOPBench:
  - n_cond_fetch correlates 0.72 with plan_calls and 0.89 with plan_rules.
  - n_cond correlates 0.89 with plan_rules.
  - n_args correlates 0.50 with plan_calls.
  - n_ask correlates 0.55 with plan_facts and plan_ambig.
  - So the GAP condition and argument counts are largely a re-count of the plan.
- tau2:
  - n_args correlates 0.63 with plan_distinct_tools and 0.61 with plan_facts.
  - n_fetch correlates 0.57 with plan_distinct_tools.
  - n_cond_ask correlates 0.54 with plan_facts.
  - frac_not_given correlates -0.37 with plan_refuse.
  - The overlap is real but only partial. The GAP features add about +0.08 to +0.14 on tau2 on top of PLAN.

**Banking within-domain Spearman with b (97 cases, never used for selection):**
- n_cond_fetch +0.218.
- n_cond +0.176.
- n_args +0.194.
- n_fetch +0.146.
- frac_cond_not_given +0.129.
- frac_not_given -0.029.
- n_cond_given -0.073.
- All signs of the main features match tau2 except the given counts. n=97, so none of this is significant on its own.
- I did not run the harness on banking. The labels were produced without the knowledge base, so they are weaker than for the other benchmarks.

## Verdict
1. The primary hypothesis holds: g1 beats c_lad pooled by +0.053 [+0.007, +0.089], 0.311 against 0.258. The whole gain comes from tau2 (+0.084 and +0.136). SOPBench is flat (+0.011 and -0.018, procedure level -0.004). The cap in the briefs, that SOPBench case difficulty needs the hidden record, was not broken by the L1 labels.
2. The condition-source features carry the gain. g3 reaches +0.042 [+0.011, +0.070] and g1 +0.053, while g2 (no PLAN) is +0.002 and g4 is +0.001. So the useful part is the broad set of counts, not the three round-8 diagnostic features alone. The effect needs PLAN already in the model.
3. This is post hoc and exploratory, because the idea came from round-8 screens on the same cases. The pooled lower bound is +0.007, so it is barely above zero. Tau2 has 278 cases, 3 domains and about 3 domains' worth of procedures, and the banking sign check is only suggestive. It needs a blind test (HANDBOOK, ThinkingBox) before it counts.

## Limitations
- The SOPBench labels are nearly constant, as the numbers show. A request that always supplies the arguments cannot show a data gap. Only the n_ask and n_cond_ask counts vary, and with the opposite sign.
- The labels may be partly a re-count of the plan, with 0.72 to 0.89 correlation to PLAN on SOPBench. The independent contribution is established only on tau2.
- I did not compare against statement length directly in the harness. The feature correlations with length are low to moderate. A length-matched control would need a new run.
- Telecom has few procedures. The strongest correlations come from telecom.
- Banking labels used tool names only and no knowledge base, and banking was checked only by Spearman. The prompt was tested on 10 cases for format and sense only.
- I did not run `metrics_extra`, so there is no hard-case recall, selective lift or label-conditional rho for these variants.

## Paths
Folder: `research/screen/ideas9/gap/`
- `PREREG.md`, `gap.py` (prompt and loader), `feats.py`
- `run.py`, `score.py`, `diag.py`
- `llm_cache.jsonl` (cache, 1,089 entries), `gap_raw.json`, `test10.json`
- `gap_sopbench.csv`, `gap_tau2.csv`, `gap_tauk_banking.csv`
- `out/` (predictions), `gap_score_main.txt`, `gap_score_main.pkl`, `score.log`, `harness.log`
- `diag.txt`, `diag2.txt`

I did not write a REPORT.md file. This message is the full report.
