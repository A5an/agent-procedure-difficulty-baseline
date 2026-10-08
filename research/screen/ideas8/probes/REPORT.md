# Swarm agent `probes`, round 8 (6 Oct 2026)

Saved by the main session from the agent's final message.

# Catalog of cheap documentation experiments (agent `probes`, round 8)

I wrote no files and ran no quick screen, because every idea needs LLM calls. The catalog below is the full return.

## Shared setup

- **Answer key.** Almost every idea is scored against a "compiled program". This is the policy turned into code by CMP, which already exists. Compile it once per procedure with a strong model.
- **Circularity.**
  - The key is built from the policy only, and the target is IRT difficulty over 28 agents. So the key never sees the evaluation labels.
  - Two further risks remain. The probe model and the compiler may share blind spots, and compiler errors become noise in the score.
  - Controls: use a different model family for the probe than for the compiler. Compile twice with different models and keep only the items where the two programs agree.
  - Never use SOPBench rule trees or tau2 user-simulator fields as the key. That would be L2 and not universal.
- **The case-level cap.** The SOPBench perform/refuse state lives in the hidden database record. Message-only probes cannot see it.
  - For SOPBench, any probe that reads only the request will mostly vary by procedure. Expect case-level rho there to stay near 0.
  - On tau2, banking and new processes, the request carries more of the case. Those are the places where per-case probes can help.
- **Probe model.** A cheap black-box API model, no tools. Call counts below are per case unless stated otherwise.

## Literature, all checked against the source pages

- **ContextCite** (Cohen-Wang, Shah, Georgiev, Madry, arXiv 2409.00729, Sept 2024) is the context-attribution method.
  - Confirmed: it works with any LLM, and the applications are verifying statements, pruning context and detecting poisoning.
  - The surrogate-model details and speed-up numbers were not on the page I read. Those numbers are unverified and I quote none.
- **ProSA** (Zhuo et al., EMNLP 2024 Findings, arXiv 2410.12405). It defines PromptSensiScore.
  - Confirmed: higher model confidence goes with higher robustness to paraphrase, and larger models are more robust.
  - The "70% variation" figure came only from a search snippet. I did not verify it in the paper.
- **Premise order** (Chen, Chi, Wang, Zhou, ICML 2024, arXiv 2402.08939). Confirmed: permuting premise order can drop accuracy by more than 30%.
- **Irrelevant context** (Shi et al., ICML 2023, arXiv 2302.00093). Confirmed: performance drops sharply when irrelevant context is added (GSM-IC). No exact figure was extracted.
- **Counterfactual simulatability** (Chen et al., arXiv 2307.08678, 2023). Confirmed: explanation precision is low and does not correlate with plausibility.
  - It concerns explanations, not decisions. I use it only for the flip-test logic.
- **Kadavath et al.** (arXiv 2207.05221, 2022). Confirmed: models predict P(IK) well in distribution but calibrate poorly on new tasks.
- **Binder et al.** (arXiv 2410.13787, 2024). Confirmed: introspection works on simple tasks and degrades on complex or out-of-distribution ones.
- **RuLES** (Mu et al., arXiv 2311.04235). Confirmed: almost all models fail simple rule-following scenarios.
- **Huang et al.** (ICLR 2024, arXiv 2310.01798). Confirmed: models cannot self-correct reasoning without external feedback.
- **Farquhar et al.** (Nature 630, 2024). Confirmed: semantic entropy detects confabulations. I did not get the AUROC numbers.
- **MILE** (Wei, Zhang, Sun, arXiv 2409.04831). Mutation testing for in-context-learning demonstrations. It mutates demos, not policies.
- **Jia and Harman** (IEEE TSE 37(5), 649-678, 2011). The mutation testing survey, confirmed.
- **Student-simulation papers.** "SMART" (arXiv 2507.05129) and "Take Out Your Calculators" (ACL 2026) are student-simulation difficulty predictors. Both are known only from a search snippet. The snippet says LLMs are poor direct judges of difficulty but simulation helps. Treat as unverified.
- **Where nothing was found.** I found no published work that applies mutation testing to policy documents to measure whether a model reads each clause. I cannot prove it does not exist.

## Catalog (tier 1 is most obvious, tier 5 least)

Every idea below is L1. Ideas marked "seed" come from the brief's seed list. The rest go beyond it: 7 are clearly beyond the seeds (Q2, C1, B1, F1, A1, TW1, O1) and T1 is a further non-seed, making 8 of 15.

### Tier 1

**Q1. Policy quiz and teach-back (seed)**
- **Measures:** how well the cheap model parses the policy. Targets skipped checks caused by misreading.
- **Inputs:** policy, plus the compiled program as key. Probe model is a black-box API model.
- **Recipe:**
  1. From the compiled program, code-generate 10 to 20 multiple-choice or true/false items per procedure. Examples: "Is X required before Y?", "At what threshold is Z refused?", "Which condition makes this action fail?".
  2. Ask the cheap model with the policy in context. Score by exact match against the program.
  3. Teach-back variant: the model lists the required checks in 8 bullets. An LLM-free matcher (condition variable names plus operators) scores recall and precision against the program's conditions.
  4. Report accuracy by condition type (threshold, status, datetime, relation).
- **Level and cost:** L1. 1 to 2 calls per procedure.
- **Varies:** by procedure only. Case-level use would need quiz items conditioned on the request, which is D1.
- **Universality:**
  - SOPBench: yes.
  - tau2: yes.
  - Banking: yes, with retrieval.
  - New process: yes.
  - All four need a compiler.
- **Status:** new as a black-box recipe. It overlaps with PRI (prior surprisal), which is a procedure-only signal at 0.689, so expect high correlation with it.
- **Evidence:** own reasoning. RuLES shows models fail simple rules, which supports the premise.
- **Expected effect:** procedure level roughly 0.4 to 0.6. Case level about 0. Main risk is redundancy with PRI and CMP.

**D1. Per-case checklist recall (seed: decomposition)**
- **Measures:** whether the model identifies which checks this request triggers. Targets the 77% of must-perform failures that skip a check.
- **Inputs:** policy, request, compiled program.
- **Recipe:**
  1. Ask: "List every condition that must be verified before acting on this request. One per line, using the policy's wording."
  2. Code-match each line to program conditions by variable name or token overlap.
  3. The program's set of conditions reachable from the request's action is the truth.
  4. Features: recall, precision, and the number of conditions the model never mentions.
- **Level and cost:** L1. 1 call per case.
- **Varies:** by case through the requested action, though SOPBench has few actions per procedure.
- **Universality:** same as Q1. The case-conditioned truth needs program branches keyed by action, so the new-process case depends on compile quality.
- **Status:** variant of PLAN and CMP.
  - It differs because it scores the cheap model against the program (an error rate) instead of counting plan size.
  - It does not use the same prompt as PLAN.
- **Evidence:** own reasoning.
- **Expected effect:** procedure level moderate. Case level small on SOPBench, plausibly +0.02 to +0.05 on tau2. Risk: it is largely a function of the procedure.

**S1. Perturbation battery sensitivity (seed: paraphrase, plus order and distractors)**
- **Measures:** how fragile the dry-run decision is to wording. Targets agent brittleness that produces random skipped checks.
- **Inputs:** policy, request.
- **Recipe:**
  1. Build 4 perturbations of the (policy, request) pair:
     - paraphrase the request;
     - paraphrase the policy;
     - shuffle clause order, which is pure code;
     - insert one irrelevant clause or chatter sentence.
  2. For each, ask for decision (perform or refuse) and the first check, in a fixed schema.
  3. Feature: share of perturbations whose (decision, first check) differs from the baseline. Optionally cluster free-text answers semantically in the Farquhar style.
- **Level and cost:** L1. 5 calls per case (baseline plus 4).
- **Varies:** by case through request and chatter. Order and policy paraphrase vary only by procedure.
- **Universality:** all four. It needs no compiler, which is its advantage.
- **Status:** new as a perturbation battery.
  - It is related to cross-model disagreement and compile disagreement (failed). The difference is that this varies the input, not the model.
  - It is also related to "position of gating checks", which is a static measure.
  - The brittleness is a behavioural measurement, not a position statistic.
- **Evidence:** ProSA (EMNLP 2024): confidence tracks robustness. Chen et al. (ICML 2024): over 30% drop from order alone. Shi et al. (ICML 2023): distractors hurt.
- **Expected effect:** small. The dry-run decision with no database record is often near-constant, so flips will be mostly noise. Risk of near-zero variance on SOPBench is high.

### Tier 2

**Q2. Refusal-example quiz (negative space)**
- **Measures:** whether the model imagines the refusal boundary correctly. Targets the 78% of must-refuse failures that act despite a failed condition.
- **Inputs:** policy, compiled program.
- **Recipe:**
  1. Ask: "Write 5 requests that must be refused and 5 that must be performed."
  2. Extract facts from each with a second call.
  3. Run the compiled program to get the true decisions.
  4. Features: precision and recall of the model's refuse label. Also the share of its "refuse" examples refused for the right reason.
- **Level and cost:** L1. 2 calls per procedure.
- **Varies:** by procedure only.
- **Universality:** SOPBench, tau2 and banking yes. New process yes. All need a compiler. SOPBench is the cleanest since conditions are explicit.
- **Status:** new.
  - It is close to the failed synthetic-case classifiers. Those trained a classifier. This scores the model's own generated examples against the program, so no classifier is trained.
- **Evidence:** own reasoning.
- **Expected effect:** procedure level plausibly 0.3 to 0.5. Case level about 0. Risk: the compiled program is noisy, so the score is noisy.

**C1. Schema-valid call construction**
- **Measures:** whether the model can build a correct tool call from the request. Targets wrong-argument and missing-argument failures.
- **Inputs:** tool schemas, request. No policy needed.
- **Recipe:**
  1. Ask for the JSON call(s) for the request, without executing.
  2. Validate each with `jsonschema` against the tool schema, in pure code.
  3. Features: invalid-call rate, missing required fields, enum violations, and the number of fields the request does not supply.
- **Level and cost:** L1. 1 call per case.
- **Varies:** by case through arguments.
- **Universality:** all four, because tool schemas exist everywhere. It does not depend on a compiler.
- **Status:** new. Related to "tool-schema complexity" and "argument provenance", which are static. This one is behavioural and code-validated.
- **Evidence:** own reasoning.
- **Expected effect:** small to moderate on tau2, where tool calls carry IDs. Near 0 on SOPBench, where the arguments are mostly just the user ID. Risk: modern models validate almost always, leaving a floor effect.

**B1. Closed-book versus open-book decision gap**
- **Measures:** how far the policy departs from the model's prior. Targets agents following common sense instead of reading the document.
- **Inputs:** request, policy. The closed-book run drops the policy.
- **Recipe:**
  1. Ask for the decision on the request with the policy removed ("this is a bank, what would a reasonable assistant do?").
  2. Ask again with the policy.
  3. Feature: binary disagreement per case. Per procedure, the share of cases that disagree.
  4. Optional: the policy clauses are "counter-prior" if removing them changes the decision.
- **Level and cost:** L1. 2 calls per case.
- **Varies:** by case.
- **Universality:** all four, no compiler needed.
- **Status:** new. Related to PRI (prior surprisal under Qwen3-8B-Base), which measures token surprise on rule text and is procedure-only. B1 measures behavioural reliance on the text per case.
- **Evidence:** own reasoning. I did not verify the knowledge-conflict literature in this round.
- **Expected effect:** procedure level plausibly informative. Case level small, since with no database record both calls often give the same cautious default. Risk of constant output.

**F1. Fact provenance coverage**
- **Measures:** for every condition variable, where its value comes from. Targets checks skipped because the fact is not at hand.
- **Inputs:** compiled program variables, tool schemas, request.
- **Recipe:**
  1. List the variables used in gating conditions (code, from the program).
  2. For each, classify the source: (a) stated in the request, (b) returned by some read tool (match field names in the tool output schema), (c) must be asked of the user, (d) nowhere.
  3. The request-side classification (a vs not) is one call per case.
  4. Features: counts of (a), (b), (c), (d) and the share of gating variables with no tool source.
- **Level and cost:** L1. 1 call per case, or code only if tool schemas include output fields.
- **Varies:** by case through (a), by procedure otherwise.
- **Universality:** all four. It needs tool output schemas. SOPBench has them in code. For a new process the tool documentation must list returned fields.
- **Status:** a variant of "argument provenance" (council list). It differs by targeting the gating condition variables, not tool arguments.
- **Evidence:** own reasoning.
- **Expected effect:** small. Risk: it correlates with PLAN's "reads" count.

**A1. Silent-assumption listing**
- **Measures:** how many facts the model would have to assume. Targets acting without verifying.
- **Inputs:** policy, request.
- **Recipe:**
  1. Ask: "List every fact you would be assuming that the request does not state."
  2. Match each assumed fact to a program variable.
  3. Features: the count of assumed facts that are gating variables, and the ratio to total gating variables.
- **Level and cost:** L1. 1 call per case.
- **Varies:** by case.
- **Universality:** all four. It needs a compiler for the variable match. A looser version can match to policy nouns.
- **Status:** new. It is close to F1 and to the "ambiguity" field of PLAN. The assumed-gating-fact count is the new part.
- **Evidence:** own reasoning.
- **Expected effect:** small. Risk: duplicates PLAN's ambiguity and facts fields.

### Tier 3

**CF1. Counterfactual flip margin and planted-violation detection (seed)**
- **Measures:** whether the model's decision responds to the right facts. Targets acting despite a failed condition.
- **Inputs:** request plus extracted facts, compiled program.
- **Recipe:**
  1. Extract the facts from the request (1 call).
  2. Build single-fact edits, in code, by swapping one gating variable (status, amount across the threshold, date).
  3. Run the program on the original and edited facts to get the true decision. Ask the cheap model the same edited request.
  4. Features: agreement between model and program on edits that truly flip, and the number of single edits that flip the true decision (the flip margin).
- **Level and cost:** L1. 1 extraction call, plus about 3 to 5 edit calls per case. Cache per procedure.
- **Varies:** by case. The number of flipping edits depends on the case's facts.
- **Universality:**
  - tau2: yes.
  - SOPBench: only for the facts stated in the message, which are few. The hidden record is exactly what is missing.
  - Banking: yes.
  - New process: yes.
  - All need a compiler.
- **Status:** new.
  - The failed idea "facts from the message plus running the compiled rule" tried to predict the label. This idea does not predict the label. It measures the model's sensitivity to edits.
- **Evidence:** Chen et al. (arXiv 2307.08678) use the flip logic on explanations. Own reasoning for the rest.
- **Expected effect:** plausible on tau2, near 0 on SOPBench. Risk: fact extraction errors dominate.

**M1. Mutation testing of the policy, with ablation variant (seed)**
- **Measures:** whether the model really reads each gating clause. Targets skipped checks directly.
- **Inputs:** policy text, compiled program.
- **Recipe:**
  1. For each gating clause, build 2 mutants: flip its operator or threshold (pure regex on numbers and negations, or by LLM), and delete it.
  2. For a small battery of 3 to 5 synthetic requests per procedure, generated from the program so that each clause matters, ask the cheap model for the decision under the original and each mutant.
  3. A mutant is "killed" if the decision changes.
  4. Mutation score is the share of mutants killed, but count only mutants where the program says the decision should change, which removes equivalent mutants.
  5. Per condition type, this gives a "read rate".
  6. ContextCite-style attribution is the cheaper variant: sample about 32 random clause subsets, fit a linear surrogate on the decision, and read the weights. The number of subsets is my choice, not a verified ContextCite setting.
- **Level and cost:**
  - Per procedure: n clauses x 2 mutants x 4 requests, about 8n calls. For a 15-clause policy that is about 120 calls.
  - Per case, restrict to the clauses relevant to the case's action: about 4 to 6 calls.
- **Varies:** by procedure. A per-case version via the relevant clauses is possible but small.
- **Universality:** all four, given a compiler. Mutation by regex works without one but cannot check equivalence.
- **Status:** new for policies as far as I could find. Jia and Harman (2011) is the software-testing origin. MILE (arXiv 2409.04831) mutates ICL demos only.
- **Evidence:** the concept is from Jia and Harman. No policy-reading number exists, so this is own reasoning.
- **Expected effect:** procedure level plausibly 0.3 to 0.6, and it gives a per-type read rate that complements the written-condition skip model. Case level near 0. Risk: a high kill rate everywhere, since a synthetic request built around one clause makes the mutation obvious.

**V1. Omission-recognition gap (seed: verification vs generation)**
- **Measures:** whether the model recognises a check it would not itself produce. Targets skipped checks specifically.
- **Inputs:** policy, request, compiled program.
- **Recipe:**
  1. Ask the model to write the ordered steps for the request (generation, 1 call).
  2. Take the program's required steps that the generation omits.
  3. For each omitted step, ask a yes/no: "Is this step required before acting?" (verification).
  4. Feature: omitted steps count (generation weakness) and the share of omitted steps the model confirms as required (recognition).
  5. The gap is "omitted but would accept", the exact pattern of a skipped check.
- **Level and cost:** L1. 1 call plus about 3 verify calls per case, batchable into one call.
- **Varies:** by case.
- **Universality:** all four given a compiler. The step list must come from the program.
- **Status:** new. Related to the PLAN features, which count the plan but do not compare against recognition.
- **Evidence:** Huang et al. (ICLR 2024) show weak intrinsic self-correction, which is consistent with the gap being real. Nothing verified on the gap itself, so own reasoning.
- **Expected effect:** small to moderate. Risk: the omission count alone is D1 under a different name.

**PM1. Pre-mortem: the most likely mistake (seed)**
- **Measures:** how many distinct plausible violations exist. Targets acting despite a failed condition.
- **Inputs:** policy, request, compiled program.
- **Recipe:**
  1. Ask: "Give the 3 most likely mistakes an assistant makes on this request, each with a 0 to 10 plausibility."
  2. Code-check each: does it violate a program condition (a real mistake), or is it harmless?
  3. Features: the number of real mistakes, the plausibility-weighted sum, and the share of mistakes on the condition types known to be weak (datetime 42%, relation 46%).
- **Level and cost:** L1. 1 call per case plus code.
- **Varies:** by case.
- **Universality:** all four given a compiler.
- **Status:** new as scored. It overlaps with the written-condition skip model, since the weak-type share is the same signal. The new part is the model's own proposal of the mistake.
- **Evidence:** Kadavath et al. suggest self-prediction is shaky out of distribution. Binder et al. say introspection degrades on complex tasks. Both point to a risk that the plausibility scores carry little signal.
- **Expected effect:** small. Risk: high, since LLM self-assessed difficulty is known to be weak.

### Tier 4

**TW1. Twin discrimination**
- **Measures:** how legible the decision boundary is from text. Targets the perform/refuse confusion.
- **Inputs:** request, compiled program.
- **Recipe:**
  1. For each case, build a minimal twin by editing one gating fact so the program's decision flips (as in CF1).
  2. Show the cheap model both requests with random order and ask: "Which one must be refused?"
  3. Feature: whether the model picks right, with a few random orderings. The forced choice removes the default-refuse bias that sinks single-case classification.
- **Level and cost:** L1. About 2 calls per case. The twin is generated once.
- **Varies:** by case.
- **Universality:** same as CF1, with the same SOPBench limit.
- **Status:** new. It differs from the failed message-to-label classifiers by comparing a pair, which controls the procedure-level prior.
- **Evidence:** own reasoning. The pairwise-comparison idea over c_lad hurt, but that compared cases, not twins.
- **Expected effect:** plausible case-level signal on tau2, near 0 on SOPBench. Risk: fact extraction and twin quality.

**O1. Step-order reconstruction**
- **Measures:** whether the model sees the dependencies between steps. Targets wrong-order and delayed-check failures.
- **Inputs:** compiled program order or data flow.
- **Recipe:**
  1. Shuffle the program's relevant steps for the case.
  2. Ask the model to put them in a valid order, with the policy in context.
  3. Score the Kendall tau against the program's dependency partial order, counting only dependent pairs.
- **Level and cost:** L1. 1 call per procedure or per case.
- **Varies:** by procedure mostly. Per case if only the steps of the case's action are used.
- **Universality:** all four given a compiler that exposes data flow.
- **Status:** new as behaviour. Data-flow metrics are static and already known.
- **Evidence:** own reasoning.
- **Expected effect:** small. Risk: steps are often so few that tau is trivially 1.

**T1. Request typicality**
- **Measures:** how unusual the request is for its procedure. Targets off-distribution requests.
- **Inputs:** policy, request.
- **Recipe:**
  1. Ask the model to write 5 typical requests for the procedure, from the documentation only.
  2. Embed them and the actual request, and compute the distance from the request to the typical set.
  3. Alternative: ask a 0 to 10 typicality score and use the logprob.
- **Level and cost:** L1. 1 call per procedure plus code.
- **Varies:** by case.
- **Universality:** all four, with no compiler.
- **Status:** new, though it leans on embeddings (tried and mostly a tau2 effect) and personas (tried).
- **Evidence:** own reasoning. The student-simulation papers (SMART, arXiv 2507.05129; ACL 2026) are from search snippets only and unverified.
- **Expected effect:** small. SOPBench requests are templated, so I expect little variance there. Risk: high.

### Tier 5 (not separately listed)

No tier 5 idea survived. Self-predicted failure ("where will you fail?") belongs here but I do not recommend it, for the reasons in PM1.

## Top 3

1. **M1, mutation testing.**
   - It is the only idea here that measures what the agent actually fails at, which is reading a gating clause.
   - It gives a per-condition-type read rate, which complements the written-condition skip model.
   - It is universal given a compiler. Calls are cheap per procedure.
   - It does not touch the case-level cap, so the realistic payoff is procedure level (0.3 to 0.6) and perhaps a small pooled gain.
2. **CF1 together with TW1, counterfactual flips and twin discrimination.**
   - They are the best candidates for real case-level signal on tau2, banking and new processes.
   - On SOPBench the hidden record limits them, and I would say so up front.
   - TW1's forced choice avoids the default-refuse bias that sank the earlier label classifiers.
3. **B1, closed-book versus open-book gap.**
   - It is the cheapest idea at 2 calls per case. It needs no compiler, so no circularity from the key.
   - It targets a distinct mechanism, the model leaning on its prior instead of the text.
   - The risk is that both calls often agree because no record is visible. Run it first on tau2 and banking.

A shared warning for all of them: the case-level SOPBench cap cannot be beaten by any message-only probe. The honest expectation is procedure-level gains in the 0.3 to 0.6 range, plus small case-level gains on tau2 only. Most of these probes also correlate with PLAN, CMP and PRI, so test each for incremental gain over c_lad before trusting it.

## Quick screen

Not run. Every idea needs LLM calls, and the quick screen was limited to pure-code features.

Files: none written. The brief is at `research/screen/ideas8/BRIEF.md`.
