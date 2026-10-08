# Swarm agent `psychometric`, round 8 (6 Oct 2026)

Saved by the main session from the agent's final message.

# psychometric: ADeLe rating fixes and item-difficulty models for agent tasks (round 8)

**Headline.** Job 1: three of the five rating fixes were already run in round 7, and measurement noise was not the bottleneck. My pure-code screen shows why. On SOPBench the ADeLe levels differ a lot between cases of one procedure, but that variation does not track difficulty (|rho| about 0.04). Job 2: LLTM explains 77% of difficulty only for items built from controlled cognitive operations. For naturally written verbal items the verified R2 is 0.12, so expect the low end here.

Nothing was fetched or run beyond what is stated. The screen script is saved at `prototype/screen/ideas8/psychometric/screen1.py`. No other files were written.

## Verified literature facts used below

- **ADeLe** (Zhou et al., arXiv 2503.06378, 26 authors):
  - 18 rubrics, levels 0 to 5+, each level with three anchor instances. The authors call the anchors "very instrumental" for the LLM rater. Anchoring is therefore already built in.
  - GPT-4o versus the human Delphi consensus: Spearman 0.75 to 0.94, mean 0.86. Five human raters: rWG 0.70 to 0.91, mean 0.83.
  - The authors state "there is no ground truth in the levels", so validity is judged by predictive power.
- **Round 7 `adele_plus`** (our own report `ideas7/adele_plus/REPORT.md` and `reliability.json`):
  - Retest on 40 shared tasks, mean per-scale rho: original 0.85, probabilities (A) 0.73, checklists (B) 0.80, relative-to-anchors (C) 0.92.
  - Within-domain retest for the original is only 0.63, and 0.64 for B.
  - Pooled case-level rho: baseline 0.131; with B 0.176 (+0.045, interval [-.018, +.123]); c_lad 0.258; c_lad plus A 0.256.
- **G-Eval** (arXiv 2303.16634): Spearman 0.514 with humans on summarization. Probability-weighted scoring is the method. The ablation figure "removing probabilities costs 0.012" came from a secondary blog, so it is unverified.
- **CheckEval** (arXiv 2403.18771, EMNLP 2025): binary checklists raise average agreement across evaluator models by 0.45. The abstract gives no other figure.
- **PoLL** (Verga et al., Cohere, arXiv 2404.18796): a panel of small models from different families has the highest kappa with humans and is more than 7 times cheaper than a single large judge.
- **LLM comparative assessment** (Liusie et al., arXiv 2307.07889, EACL 2024): pairwise beats absolute scoring for FlanT5 and Llama2-chat, but position bias is strong. A product-of-experts method (arXiv 2405.05894) matches full-comparison performance with about 2% of the comparisons.
- **LLM compare** (Ballon et al., arXiv 2512.14220): pairwise LLM difficulty judgements fed into Bradley-Terry reach r >= 0.80 with human annotations on n = 1876 problems. This is agreement with human labels, not with agent outcomes.
- **Scoring bias in LLM judges** (arXiv 2608.05726): judges favour particular numbers regardless of input. I read only the search snippet.
- **Many-facet Rasch with LLM raters:**
  - Jiao, Song and Lee (arXiv 2505.18486) found ten LLMs show different severity and leniency patterns, with ChatGPT 4o, Gemini 1.5 Pro and Claude 3.5 Sonnet best.
  - Sachdeva and Boudol (arXiv 2608.27463, EMNLP 2026) apply Rasch measurement to nine LLM raters, finding differences in severity, scale usage and question-order sensitivity.
- **Item-difficulty modeling:**
  - Fischer 1973, Acta Psychologica 37:359-374: citation checked, but I could not verify any variance figure from it.
  - Embretson 1998, Psychological Methods 3:380-396: citation checked, numbers unverified.
  - Sun, Liu and Luo 2019, Frontiers in Psychology, number-series generator: LLTM explained 77.4% of difficulty variance (r = 0.88).
  - A Frontiers in Psychology 2017 paper on Q-matrix validation (authors not checked) reports Rasch-versus-LLTM correlations of 0.85 (listening) and 0.72 (reading). It says published values range from 0.75 to 0.98, citing Sonnleitner 2008 at 0.98 and Baghaei and Ravand 2015 at 0.75.
  - Hornke and Habon's three cognitive dimensions explained about 40% (secondary, from the review arXiv 2201.08450).
  - Bejar et al. 2017 (GRE sentence equivalence, 800 operational items, linear regression): R2 = 0.12. This comes from an ETS slide deck (Bejar and Flor, Maryland Conference 2017), not the paper.
  - Gorin and Embretson 2006 (GRE paragraph comprehension): about 0.34 from a search snippet only. Unverified.
- **LLM difficulty prediction for exams:**
  - Hoyl (arXiv 2602.00034): LLM-extracted features (solution steps, cognitive complexity, misconceptions) give Pearson about 0.78 on unseen math questions. The abstract gives no setup beyond a student-response simulator.
  - Wang et al. (arXiv 2607.28634): GPT-4.1 reaches only quadratic weighted kappa 0.578 and underestimates hard items.
- **Other citations:**
  - Junker and Sijtsma 2001, Applied Psychological Measurement 25:258-272 (DINA): citation checked.
  - Janssen, Schepers and Peres 2004, a chapter in De Boeck and Wilson, *Explanatory Item Response Models*: citation checked.
  - Liu, Xu and Gu (arXiv 2603.14676): text-embedding-informed Q-matrix priors for LLM cognitive diagnosis.
  - KLM root-mean-square error of 21%: Wikipedia only, unverified against Card, Moran and Newell.
  - Cognitive load theory: element interactivity is counted as the number of simultaneously processed elements, and it depends on the learner's schemas. I found no hard difficulty-correlation number.

## Job 1: the ADeLe rating problem

### Verdict

Four of the user's six candidate fixes are untried or only partly tried. They have a weak premise, shown by the screen below. The 18 rubrics already have three anchors per level. Round 7 already ran probability elicitation, checklists, and domain-matched anchors, and none moved c_lad (0.258 versus 0.256, 0.225 and 0.215).

The retest numbers say the rater is repeatable across the whole pool (0.85). Within a domain it is weaker (0.63), and checklists did not fix that.

### Quick screen (pure code, exploratory screen, not the protocol)

I split each ADeLe scale's variance into between-procedure and within-procedure parts. I then took the Spearman correlation between within-procedure-centred levels and within-procedure-centred IRT difficulty. Source: `data/<bench>/features/adele.csv` and `irt/1d_1pl/items.csv`, under the lock.

| benchmark | n cases / procedures | within-procedure share of IRT difficulty variance | within-procedure share of ADeLe scale variance | mean abs. within-procedure rho |
|---|---|---|---|---|
| SOPBench | 830 / 70 | 0.74 | 0.54 | 0.039 (range -0.073 to 0.061) |
| tau2 | 278 / 167 | 0.13 | 0.08 | 0.132 (best -0.263, MCr) |
| banking | 97 / 97 | 0.00 | 0.00 | not defined |

- **SOPBench:** the levels do vary across cases of the same procedure, but they do not track within-procedure difficulty. Cleaner rating of these scales cannot help. The gap is a lack of case-level signal in the scales, not rating noise.
- **tau2:** "procedure" is essentially the task, so the numbers are weak evidence either way.
- **Banking:** one case per procedure, so no within-procedure test is possible.

A second screen was not run.

### Ideas for the rating problem

**P1. Many-facet ordinal rater model with several raters** (tier 3, new)
- **Measures and targets:** removes rater severity and scale-use bias from the 18 levels. It does not target any agent failure directly.
- **Inputs:** the case request plus the rubrics, from three LLMs of different families (black-box API).
- **Recipe:**
  - Rate each case on each scale with raters r1 to r3, using the existing 3-anchor prompt.
  - Fit a many-facet partial-credit or ordinal logistic model, level ~ case measure + rater severity + rater-by-scale threshold. Do this per scale on training data plus the new benchmark. Use any ordinal many-facet routine, or statsmodels OrderedModel with rater dummies.
  - Take the case measure and its standard error as the feature. Optionally weight the feature by 1/SE.
- **Level and cost:** L1, three calls per case.
- **Varies by case:** yes (this is still ADeLe).
- **Universality:** yes on all four; it needs only the rubrics and the request.
- **Status:** new. The round-7 raters were all one model family (Gemini), and `adele_plus` tested single-call variants only.
- **Evidence:** PoLL, Jiao et al. 2025, Sachdeva and Boudol 2026.
- **Expected effect:** small at case level (at most +0.02 to +0.05, my guess). The screen shows noise is not the bottleneck. Most likely it yields the same 0.13 on its own and no gain over c_lad.

**P2. Per-scale within-procedure comparative judgement with Bradley-Terry** (tier 3, variant of "pairwise comparisons over c_lad")
- **How it differs from the failed version:**
  - It compares two cases of the same procedure on one demand scale, for example "which case needs more checks of this type", not "which is harder overall" over the c_lad score. Ties are allowed.
  - It runs only inside a procedure, so it removes the procedure effect by construction, which is the case-level target.
  - Pairs are adaptive and sparse, 2% to 10% of all pairs under a product-of-experts or Gaussian aggregation, and each pair is judged in both orders to cancel position bias.
- **Recipe:**
  - For each procedure with k cases, draw about 4k pairs.
  - Ask the LLM per scale which case demands more of it, in both orders, and record 1, 0 or tie.
  - Fit Bradley-Terry per scale and procedure, then centre the scores within the procedure.
  - Feed the 18 centred scores to the existing ADeLe feature block.
- **Level and cost:** L1, 18 scales x 4k pairs x 2 orders, about 144k calls per 1000 cases. Restrict to the 6 or 8 scales that vary most within procedure to cut this.
- **Varies by case:** yes, and by construction only within a procedure.
- **Universality:** SOPBench and tau2 yes. Banking and a new company process have one case per procedure in the benchmarks, so it needs several cases of the same process.
- **Evidence:** Liusie et al.; Ballon et al. (r >= 0.80 with human labels).
- **Expected effect:** the screen predicts nothing for SOPBench, because the scales do not track difficulty even when varied. The most that can be hoped for is a different, more reliable estimate of the same scales. Risk of failure: high.

**P3. Magnitude estimation versus a reference case** (tier 4, new, no source)
- **Recipe:**
  - Fix one reference case per scale in the same procedure, say "level 100".
  - Ask the LLM for a ratio, for example "how many times more of this demand does the case need; give a positive number".
  - Take the log.
  - Rater number bias is a known risk (arXiv 2608.05726, snippet only).
- **Level and cost:** L1, 18 calls per case.
- **Varies by case:** yes.
- **Universality:** needs a same-procedure reference case, so it fails for a single-case new process.
- **Status:** new, own reasoning.
- **Expected effect:** about the same as P2. High risk.

## Job 2: item-difficulty models (concrete recipes)

### Calibration of expectations from the classics

LLTM-type models explain 77% to 98% (as correlation) of difficulty when items are built from controlled cognitive operations: matrices, number series, generated reading items. They explain only 12% to 34% for naturally written verbal items.

Our cases are naturally written, written by a benchmark builder and not generated from operations. Expect the low end.

### Ideas

**M1. LLTM with learned positive operation weights, log-additive** (tier 1, variant of PLAN and `conditions`)
- **Measures and targets:** difficulty as a weighted sum of operation counts in the case's plan. It targets skipped checks and read steps.
- **Inputs:** policy, tools, request; black-box LLM.
- **Recipe:**
  - One LLM call writes the plan for the case and tags each step with an operation type: authenticate, lookup, compare datetime, compare relation, compute, write call, refuse. This matches the known step-reliability types.
  - Build the count matrix Q (cases x types).
  - Fit a non-negative or LLTM-style regression of IRT difficulty on Q, using logistic weights.
  - Predicted difficulty = Q times eta, with eta pooled across benchmarks.
- **Level and cost:** L1, one call per case.
- **Varies by case:** yes, if the plan depends on the request; no, if the plan comes from the policy only.
- **Universality:** yes on all four. The operation inventory is closed and domain-free.
- **Status:** variant of PLAN. The difference is a small, closed, interpretable vocabulary with sparse, constrained weights, so it cannot overfit.
- **Evidence:** Fischer 1973; Embretson; Sun et al. 77.4% (generated items).
- **Expected effect:** about 0.2 to 0.3 on tau2, procedure level; SOPBench near 0.2 (cap). Risk: it duplicates PLAN, since the planner already writes calls and reads.

**M2. Radicals versus incidentals invariance filter** (tier 3, new)
- **Measures and targets:** keeps only features that depend on a task's cognitive structure (radicals: number of checks, conditions, dependencies) and discards those driven by surface wording or naming (incidentals). It targets the overfitting that produced the encoder's tau2-only gain.
- **Recipe:**
  - For each training case, have a cheap LLM make two or three rewrites that change only incidentals: names, order of sentences, synonyms, field labels. A second rewrite changes radicals: adds or removes one condition.
  - Re-extract every candidate feature (ADeLe scales, plan counts, length, embeddings).
  - Keep a feature only if it moves little under incidental rewrites and moves in the right direction under radical rewrites (an item-model validity check).
  - Fit the final model on the survivors.
- **Level and cost:** L1, 2 to 3 rewrites plus feature recalculation per case. This is a filter run once on the training benchmarks.
- **Varies by case:** yes.
- **Universality:** yes on all four. The rewrites need only the text.
- **Status:** new as a filter. The idea comes from automatic item generation: Embretson's radicals and incidentals, in Hornke and Habon style item models.
- **Evidence:** own reasoning; the radical/incidental distinction is classic AIG (citation not individually checked).
- **Expected effect:** modest. At best, a more robust transfer, say +0.02 to +0.04. Risk: moderate. Rewriting may alter the semantics.

**M3. Explanatory IRT with a procedure random effect** (tier 2, new as a training recipe)
- **Measures and targets:** estimates feature weights from within-procedure variation only. This is the LLTM-plus-error model (Janssen, Schepers and Peres 2004; De Boeck and Wilson 2004) with the procedure as the item group.
- **Recipe:**
  - Fit difficulty = sum(eta_k x feature_k) + u_procedure + e_case, with u_procedure random.
  - Report the within-procedure R2 separately from the between-procedure R2.
  - For prediction on new procedures, set u to 0 and use the fixed part, then take rho within domain.
- **Level and cost:** no extra calls. L0 or L1 depending on the features.
- **Varies by case:** yes. It tells you which features can ever work at the case level.
- **Universality:** SOPBench and tau2 yes; banking has one case per procedure so no random effect; a new company process needs several cases.
- **Status:** new as a diagnostic and training recipe. The group-mean-centring trick is what I did in the screen.
- **Evidence:** Janssen et al. 2004 (citation only).
- **Expected effect:** it may not raise rho, but it will tell the team which candidate features have within-procedure signal. My screen already shows ADeLe has none on SOPBench.

**M4. DINA-style conjunctive model over check types (Q-matrix)** (tier 2, variant of `conditions`)
- **Measures and targets:** a case succeeds only if every required step type is done. Difficulty is 1 minus the product of the per-type step reliabilities, which matches 77% of must-perform failures being a skipped check.
- **Inputs:** policy, tools, request.
- **Recipe:**
  - Build the Q-matrix by LLM: for each case and each step type, 1 if required.
  - Estimate per-type reliability from training data. DINA-style slip and guess parameters are fitted by the CDM package, or use the shortcut of logistic regression on log-reliability.
  - Predict difficulty as minus the sum of the log reliabilities of the required types.
  - An LLM-informed Q-matrix prior is tested in Liu, Xu and Gu (arXiv 2603.14676).
- **Level and cost:** L1, one call per case for the Q row.
- **Varies by case:** only if the step set differs by case.
- **Universality:** yes on all four, since the type list is closed. DINO (disjunctive) is the wrong form here because skipped steps compound.
- **Status:** variant of `conditions` and of the written-condition skip model. The difference is the conjunctive product form with a few shared reliability parameters.
- **Evidence:** Junker and Sijtsma 2001 (DINA); the step reliabilities (88%, 42%, 46%) are our own data.
- **Expected effect:** about equal to M1, on top of c_lad +0.01 to +0.03. Risk: it duplicates M1.

**M5. Element interactivity of the decisive check** (tier 3, variant of "simultaneous load")
- **Measures and targets:** how many information elements must be held together to evaluate the one check that decides the outcome. Cognitive load theory counts interacting elements that cannot be processed in isolation.
- **Recipe:**
  - Ask an LLM to write the decisive check as a formula.
  - Count its distinct inputs, and mark which come from the request, the tool result or the policy.
  - Count only the elements that must be combined.
- **Level and cost:** L1, one call per case.
- **Varies by case:** yes.
- **Universality:** yes on all four.
- **Status:** variant of the already-listed simultaneous-load and compiled-program ideas. It differs by counting only the decisive check's interacting inputs.
- **Evidence:** cognitive load theory (secondary sources); I did not verify a difficulty-correlation number.
- **Expected effect:** low. Risk: it duplicates existing features.

**M6. GOMS and KLM-style operator time with HTA depth** (tier 2, variant of PLAN)
- **Measures and targets:** unit-time operator costs summed over the plan (read, compare, compute, call, write), plus the hierarchical task analysis depth of the goal tree.
- **Recipe:**
  - Use the M1 plan.
  - Assign each operator a fixed time or cost from a small table, and sum it. Calibrate the table on training data.
  - Compute the goal-tree depth from a nested plan.
- **Level and cost:** L1, same plan call as M1.
- **Varies by case:** yes if the plan does.
- **Universality:** yes on all four.
- **Status:** variant of PLAN and AST features. The new part is the calibrated cost table.
- **Evidence:** KLM root-mean-square error 21% (Wikipedia; original unverified). The repo already has `features/human_time.csv`, which could validate the cost table.
- **Expected effect:** low, close to M1.

**M7. Two-parameter targets: difficulty for strong versus weak agents** (tier 4, new)
- **Measures and targets:** fact 4 says strong and weak agents find different things hard (rho 0.41). Fit item discrimination and difficulty separately, or fit b_strong and b_weak.
- **Recipe:** fit M1 to the top-5 and bottom-5 agents' difficulty separately, then average with weights tuned on training data. Use the difference as a diagnostic.
- **Level and cost:** no extra calls.
- **Varies by case:** yes.
- **Universality:** all four, since the targets come from the IRT data.
- **Status:** new. Own reasoning.
- **Expected effect:** small at best. Risk: the disagreement may be pure noise.

**M8. Item-family hierarchical model (case within procedure)** (tier 4, variant of M3)
- **Measures and targets:** treat the cases of one procedure as clones of an item model. Difficulty = family mean + clone deviation; predict the family mean from L0 documentation (where rho is 0.5 to 0.7) and the deviation from case-specific features.
- **Level and cost:** L0 or L1.
- **Varies by case:** the deviation part varies by case; the family part does not.
- **Universality:** same limits as M3.
- **Status:** new as a two-stage decomposition. Own reasoning.
- **Expected effect:** the family mean gives 0.5 to 0.7 at procedure level, so the combination is capped by the within-procedure part. Not a gain by itself.

## Top 3

1. **M3, explanatory IRT with a procedure random effect.** It is the only item that changes what we learn, not just what we compute. The screen shows the SOPBench within-procedure share of difficulty variance is 74% while ADeLe has zero within-procedure signal there. M3 turns every candidate feature of this round into a pass or fail test at the case level. I recommend running it first, for free.
2. **M1 with M4, LLTM and conjunctive form over a closed step vocabulary.** It is the classic psychometric recipe applied to the failure mechanism we know (skipped checks), with few shared parameters. Expect a modest gain over c_lad at best, and it overlaps PLAN.
3. **M2, the radical and incidental invariance filter.** It addresses the actual transfer failure (the encoder's tau2-only gain, F1 not holding on a fresh benchmark). It does not add a signal. It removes features that look good on training but do not carry cognitive structure.

The rating fixes (P1 to P3) rank below these. The user's hypothesis, that ADeLe is hard to rate and rating better will fix it, is not supported by round 7 plus my screen.

## Files

- `research/screen/ideas8/psychometric/screen1.py`
