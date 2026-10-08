# Fresh view: how to get unstuck (5 Oct 2026)

Basis: only CONTEXT.md. No searches were run, so every "nearest work" entry is from memory and unverified.
Numbers quoted below are copied from CONTEXT.md.

## Diagnosis in four lines
1. The target is wrong for the data. 60% of case difficulty is procedure x perform/refuse, which lives in the database, so documentation alone cannot reach it. A ceiling of about 0.2 is probably real, not a method failure.
2. The metric (within-domain case-level rho, MDD 0.05 to 0.18) cannot separate methods. More variants will not fix that.
3. Positive signals exist at other units: procedure level (0.41 to 0.52 vs 0.23) and pilot run (+0.134).
4. A manager does not ask "rank cases by IRT difficulty". They ask which process to automate, with which agent, with what human fallback, and how much testing is needed. Redefine the question around those decisions.

Marked [NEW QUESTION/UNIT/DATA] = changes question, unit or data. Marked [SUP] = uses the supervisor's background naturally.

---

## D1. Portfolio ranking of procedures [NEW UNIT]
- Question: from documentation alone, can we rank whole procedures by expected agent success well enough to choose which to automate first?
- Why it could work: procedure-level rho is already 0.41 to 0.52 for policy-reading methods vs 0.23 baseline. Procedure variance is 26% and domain 14%, so the unit has signal. Also it is the unit a manager decides on.
- Minimal decisive experiment: SOPBench, about 70 procedures, mean success across the 28 agents (after cleaning the empty-answer agents, see D5). Leave-one-procedure-out and leave-one-domain-out. Methods: baseline, best 2 policy-reading methods, a trivial predictor (policy length). Metrics: Spearman rho and top-quartile precision. Inference: paired bootstrap by procedure, plus a permutation test. Add tau2 domains and tau-Knowledge as a second dataset only if procedures can be defined there (tau2 telecom has 5 distinct texts, so probably not).
- Kills it: rho CI includes the trivial predictor, or top-quartile precision under 0.5 on leave-domain-out.
- Nearest work (unverified): benchmark-difficulty prediction from prompts (various "prompt difficulty" papers), Krsteski and Meyer protocol (given in context), Ge et al. agent psychometrics.
- Partner gets: a ranked automation backlog from SOP text only, with honest intervals.

## D2. Information ladder with non-sensitive aggregate side information [NEW DATA]
- Question: which cheapest piece of non-sensitive side information closes the gap between documentation-only (0.13) and the case-label oracle (0.52)?
- Why: difficulty is roughly p*b_perform + (1-p)*b_refuse per procedure, and p is the share of cases that end in "perform". A company can give p (approval rate, rejection rate) without releasing any record. This fits the course wording "without access to sensitive operational data". Rungs: doc only, + p, + p and per-condition failure rates, + 1 pilot, + 5 pilots.
- Experiment: SOPBench tasks give p per procedure (action_should_succeed). Procedure-level and case-level rho for each rung, using the existing pilot numbers for the top rungs. Cost per rung in dollars or in "questions asked of the company". Pure existing data.
- Kills it: procedure-level rho given p is under 0.35, or p adds nothing over doc features (paired difference CI includes 0).
- Nearest work (unverified): value-of-information framing in active evaluation, "tinyBenchmarks" style few-sample estimation.
- Partner gets: a short data-request list ("send us these 3 aggregate numbers per process") instead of a vague promise.

## D3. Step-level reliability from existing trajectories, then the router [SUP] [NEW UNIT]
- Question: can per-step-type success rates for each model be estimated from public trajectories well enough that a series-reliability model predicts whole-case success better than a whole-case baseline?
- Why: CONTEXT says the public data has only whole-task success, but the SOPBench trajectories carry dirgraph_satisfied, action_called_correctly, constraint_not_violated, database_match per record, and the full tool-call sequence. That is step-level evidence for 28 agents at zero cost. The supervisor's router needs exactly this table. E8 ($29) can then test the hand-off instead of discovering the table.
- Experiment: parse trajectories into steps (look-up, check, irreversible action, final answer). Fit p(model, step type). Predict case success as a product over the required steps of the case (series system). Held-out: leave-procedure-out. Metric: AUC and Brier for case success against a model-by-procedure baseline. Then a simulated router: cheapest model per step with fallback to the best model, evaluated on cost vs success using logged trajectories only as a plug-in estimate.
- Kills it: step-type success varies by under 5 points across step types within a model, or the series model does not beat the baseline Brier score.
- Nearest work (unverified): FrugalGPT and cascades, RouteLLM, learning to defer (Madras et al.), process reward models.
- Partner gets: a concrete "which steps stay with a person" map for each process.

## D4. Certificate length: AND/OR hardness of the decision [SUP] [NEW UNIT]
- Question: is perform-vs-refuse hardness explained by certificate size, i.e. performing needs all k constraints verified (conjunction) while refusing needs one violated constraint (disjunction), and does the number of required look-ups predict within-label difficulty?
- Why: it explains the +1.64 vs -0.62 gap mechanically and is the combinatorial-optimisation reading of the finding. Within the perform class, difficulty should rise with k and with the dependency depth; within refuse, with the position of the first failing check and the number of decoys. Features come free from user constraints, initial_database and directed_action_graph in the SOPBench release (symbolic, no LLM).
- Experiment: compute k, minimal certificate length, number of satisfied-but-irrelevant constraints, depth of directed_action_graph, per case. Regress IRT b within label (rho_given_label as metric) with leave-procedure-out, compared with the baseline given label.
- Kills it: rho_given_label gain under 0.05 over the baseline with a CI including 0.
- Nearest work (unverified): reasoning-depth scaling in planning benchmarks, SAT/CSP phase-transition hardness, FOLIO or ProofWriter depth effects.
- Partner gets: a "decision complexity" score per process branch computable from rules, no data needed.
- Caveat: the features need the database, so this is not documentation-only. Say so openly. It answers the question "once you know the case, what makes it hard".

## D5. Noise ceiling, label cleaning and variance decomposition across benchmarks [NEW DATA]
- Question: what is the highest rho any predictor could reach (split-half reliability of IRT difficulty), and does cleaning artefact agents (gpt-5, gpt-5-mini, gemini-2.5-pro, empty answers scored as refusal) raise or lower it?
- Why: if split-half reliability of within-domain difficulty is about 0.5 and documentation can see only 26% of variance, then 0.13 to 0.19 is near the ceiling and the paper's honest result is "the ceiling is low and here is why". This is publishable as a measurement result and costs nothing. It also redefines the 0.131 baseline against its ceiling.
- Experiment: random split of the agents into two halves, fit IRT on each, correlate difficulties within domain (many splits). Repeat after dropping the three artefact agents, and per tau2 simulator. Report attainable rho. Repeat variance decomposition on tau2 and tau-Knowledge.
- Kills it: split-half reliability is near 0.9 (then the ceiling is not the issue and the stuck state is a feature problem).
- Nearest work (unverified): reliability of benchmark scores, psychometric reliability and attenuation correction, Krsteski and Meyer.
- Partner gets: knowing in advance how much of an automation outcome is unknowable from paperwork.

## D6. Decision metric: triage with abstention and cost [NEW QUESTION]
- Question: at a fixed budget of human review, how many agent failures can documentation plus case-visible information catch, compared with the label-only rule?
- Why: rho over all cases rewards sorting the easy middle. The manager wants a flag list. Selective-prediction curves (risk at coverage) have more power for the top and bottom of the ranking where signal lives. Expected-cost metric: cost = review cost * flagged + failure cost * missed.
- Experiment: per case, per agent (not mean difficulty): predicted failure probability from baseline, label, one pilot (cheap model plus Gemini Flash judge), their combinations. Metrics: area under the risk-coverage curve, cost saved at 80% recall of failures, paired bootstrap by procedure. Existing data only.
- Kills it: no method beats the label-only rule on cost saved, or CIs overlap everywhere.
- Nearest work (unverified): selective classification (Geifman and El-Yaniv), learning to defer, conformal risk control.
- Partner gets: a review-load number ("review 20% of cases to catch 70% of failures").

## D7. Sequential pilot allocation as best-arm identification [SUP] [NEW QUESTION]
- Question: given a budget of B pilot runs, how should a company allocate runs across procedures (and across candidate agents) to find which procedures are automatable at a target reliability, and does a documentation prior cut the needed runs?
- Why: pilot outcome is the only large positive result. The next question is how to buy the information. This is a bandit and stopping problem, the supervisor's home ground (stochastic decision making, RL). The documentation prior is useful here as a Bayesian prior even if rho is poor, because a prior only needs to shrink variance.
- Experiment: offline replay on the 830x28 matrix. Policies: uniform, successive halving, Thompson with an uninformative prior, Thompson with a doc-based or kNN-procedure prior. Metric: probability of correctly labelling procedures above/below a 0.8 success threshold vs budget, and runs to reach 90% correct. Existing data, nothing new to run.
- Kills it: the informed prior saves under 15% of runs vs uninformative at equal accuracy, and adaptivity does not beat uniform.
- Nearest work (unverified): best-arm identification (Even-Dar et al., Audibert and Bubeck), active testing (Kossen et al.), efficient benchmarking by adaptive sampling.
- Partner gets: a testing plan: how many trial runs per process before the go/no-go decision.

## D8. Controlled documentation edits, causal effect on failure [NEW DATA]
- Question: which documentation properties actually cause agent failures, measured by editing the text and keeping cases fixed?
- Why: all 60 variants are correlational and confounded by the procedure. Editing the same procedure (paraphrase, reorder conditions, make an implicit condition explicit, add a distractor condition, split a compound sentence) removes the confound and the case state. The effect is the manager's lever: rewrite the SOP.
- Experiment: take 150 SOPBench cases (the E8 sample can be reused), 4 edits per procedure, one cheap model (Gemini Flash class), 2 repeats, graded by the SOPBench evaluator. Cost near $20 to $40 by E8's scale of $29 for 150 cases and two phases. Metric: paired success difference per edit type with a McNemar or cluster bootstrap by procedure.
- Kills it: all edit effects under 3 points with CIs including 0 (then wording hardly matters and the DB state is everything, which is itself a clean finding).
- Nearest work (unverified): prompt sensitivity and format-sensitivity studies (Sclar et al.), SOP clarity studies.
- Partner gets: a checklist "rewrite your SOP like this before automation" with measured effect sizes.

## D9. Documentation to executable rules: predict feasibility of the case, not the difficulty [NEW QUESTION] [SUP]
- Question: can an LLM compile SOP text into a decision table or constraint list that a symbolic checker runs on case data, and how accurate is the resulting perform/refuse label compared with the agent itself?
- Why: the label is the biggest single factor (rho 0.52). If compiled rules reproduce it, then (a) process managers get a deterministic pre-check and (b) the doc-only gap closes once data is plugged in. Compiled rules also expose under-specified policies, where compilation fails or disagrees across two compile runs.
- Experiment: one Gemini Flash call per procedure (about 70 procedures, a few dollars) to emit constraints in a fixed schema. Execute on initial_database of each case. Metric: label accuracy and the resulting rho_given_label. Also compile twice and measure rule-level agreement as an "under-specification" score, tested against procedure-level difficulty.
- Kills it: label accuracy under 85%, or agreement score unrelated to procedure difficulty (rho under 0.2).
- Nearest work (unverified): DMN decision tables from text, SOPBench's own symbolic verification, semantic parsing to rules, LLM-to-PDDL translation.
- Partner gets: an automatic "does this SOP compile into rules" audit, pointing to the clauses that are ambiguous.

## D10. Make pilots reliable: a state-aware judge [NEW DATA]
- Question: does giving the cheap judge the final database or state diff cut its tau2 error from 33% to near the SOPBench 10%, and does that restore the pilot's gain on tau2?
- Why: the pilot is the strongest known signal but it works poorly where the judge cannot see state. This is a targeted fix with a clear metric. If the pilot is the recommended product, its measurement quality is the weak link.
- Experiment: tau2 tasks with logged trajectories. Reconstruct the final environment state by replaying tool calls through the tau2 domain code where possible. Judge prompt with and without the state. Metrics: judge accuracy vs gold on a 300-trajectory sample ($5 to $15), then pilot rho gain on tau2 vs the 0.13 baseline.
- Kills it: state replay not possible for most tasks, or judge error still above 25%.
- Nearest work (unverified): LLM-as-judge on agent trajectories, state-based reward in tau-bench (the benchmark's own gold check), outcome verifiers.
- Partner gets: a trustworthy cheap dry-run protocol on their own process.

---

## Ranking

1. **D3 then D7 as one story, ranked first as D3.** Step-level table from existing trajectories feeds the supervisor's router idea at zero cost and yields the paper's solution-design half. It also de-risks E8 (the $29 pilot), which currently has to discover the table.
2. **D2 (information ladder, with D1 as its procedure-level readout).** Turns the failure into the main result: documentation gives X, an aggregate gives Y, a pilot gives Z, with prices. It matches the course topic (feasibility assessment without sensitive data) and needs only existing data.
3. **D5 (noise ceiling).** Cheapest, fastest and decides how to frame everything else. If the ceiling is low, the paper becomes "what is knowable at each information level", and D2 is the structure of that paper. Run it first in time even though it ranks third by upside.

Why not D4, D9: strong ideas but they use the database, so they only partly answer the accepted gap. Keep D4 as a mechanism section and D9 as a stretch. D7 is the best contribution from the supervisor's side if D3 stalls. D8 is the best "business value" fallback if all data-only ideas fail, and fits the $50 cap.

Suggested order: D5 (1 day), D2 and D1 (2 days), D3 (about 1 week), D7 replay (3 days), E8 only after D3.

## Three things to stop doing

1. Stop screening more feature variants on case-level within-domain rho. Sixty variants all at 0.06 to 0.19 with MDD 0.05 to 0.18 means the test cannot answer, and new features are the same question.
2. Stop reporting the 0.131 baseline without a ceiling and without cluster-by-procedure intervals. Cases of the same procedure are not independent, so current CIs are probably too narrow, and a number with no ceiling cannot be called weak or strong.
3. Stop treating the benchmark scores as clean truth. The empty-answer agents scored as correct refusals, the mixed tau2 simulators, and telecom's 5 distinct texts distort difficulty. Clean or drop them before any method comparison.
