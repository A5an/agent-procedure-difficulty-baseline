# Swarm agent `structure`, round 8 (6 Oct 2026)

Saved by the main session from the agent's final message.

I did not run the quick screen on the two best ideas, because both need an LLM call or compiled-program parsing. I screened a cruder pure-code proxy instead, and its results are below. The citations are checked (below). Your instruction to return the catalog as text is why I'm giving it all here. The only file I wrote is `research/screen/ideas8/structure/screen.py`.

## Catalog (tier 1 = obvious, tier 5 = least obvious)

Every idea below is its own recipe unless the status line says it is a variant of a tried idea. Mark "not universal" is used only where it applies.

### S1. Typed-argument fit of the request against the tool schema (tier 1)
- **Measures:** how many of the target tool's required arguments the request supplies, in which formats, and which ones are not stated literally. It targets wrong or invented arguments, and ToolSandbox's "insufficient information" and "canonicalization" categories.
- **Inputs:** tools and request. No model.
- **Recipe:**
  1. Parse the target tool's required arguments and their types from the schema.
  2. Extract typed literals from the request with regexes, for example numbers, snake_case ids, upper-case codes, dates and times.
  3. Per case, compute `n_required_filled`, `n_required_missing`, `n_format_mismatch` and `n_distinct_formats`.
  4. Format mismatch means the value is given as "next Friday" while the schema wants ISO date, or as "$4,500" while it wants float.
- **Level:** L0, near-zero cost.
- **Varies by case:** yes. This is the main reason to try it.
- **Universality:** SOPBench yes (the message states the parameters). tau2 partly (the scenario states ids in prose). Banking yes. New process yes (any schema with types).
- **Status:** variant of "argument provenance" and "tool-schema complexity" from the 6 Oct councils. The difference is that it scores the gap between the message and the schema per case. The councils scored the schema or where arguments come from.
- **Evidence:** ToolSandbox (arXiv 2408.04682) names the three scenario categories; I did not extract numbers from it.
- **Expected effect:** small, case rho about 0.05 to 0.15. See the screen below.

### S2. Slot-fill gap as the number of tool lookups before the action (tier 1)
- **Measures:** variables the policy needs that the request does not state, so the agent must fetch them. It targets skipped checks.
- **Inputs:** policy and tools, plus the request. No model; an LLM can do the extraction step at L1.
- **Recipe:**
  1. List the variables the compiled program reads. In `programs_t0.json` these are `params.get(...)` and `tools.internal_*` calls.
  2. Subtract the variables the request fixes.
  3. The remainder is `U`, the number of free variables that need a read.
  4. Report `U` and `U / total`.
- **Level:** L0 if the variables are parsed from the compiled code, L1 if an LLM lists them.
- **Varies by case:** weakly. The request fixes different variables across cases of one procedure. Mostly procedure-level.
- **Universality:** yes on all four, since it only needs the text, tool names and request.
- **Status:** variant of "partial evaluation of the compiled program". The difference is that it counts only variables, with no evaluation.
- **Evidence:** the Schema-Guided Dialogue dataset (arXiv 1909.05855, AAAI 2020) models services as slots and intents and reports zero-shot dialogue state tracking on new APIs. I did not check any per-slot difficulty number.
- **Expected effect:** procedure level 0.2 to 0.3, case level near 0.05.

### S3. Policy as a Boolean function: sensitivity and certificate size at the case's known inputs (tier 3)
- **Measures:** how many condition variables an agent must check correctly to reach the right verdict at this case. It targets "78% of must-refuse failures act despite a failed condition".
- **Inputs:** compiled policy program. No model after compilation.
- **Recipe:**
  1. Extract the program's boolean variables `x1..xn`, with comparisons treated as one variable each.
  2. Build the truth table by brute force for `n <= 20`, or use a BDD or SAT enumeration for larger `n`.
  3. The request fixes some variables (the known set `K`). Fix `K` to the values the request implies.
  4. At each completion of the free variables, compute local sensitivity (how many single flips change the output).
  5. Report mean local sensitivity over completions, the mean certificate size (smallest fixed subset that forces the output), and the expected decision-tree depth under uniform completions.
- **Level:** L0 after one compile (already paid).
- **Varies by case:** yes, through `K`. The request fixes different variables per case.
- **Universality:** SOPBench yes (the compiled programs exist for 415 procedure variants). tau2 only if policy text is compiled to code (needs the compile step, so L1). Banking yes after compile. New process yes after compile.
- **Status:** new in this form. "Decision-table metrics" and "partial evaluation" are on the tried list; this is different because it uses sensitivity and certificate complexity, not table size or rule count.
- **Evidence:** Buhrman and de Wolf (TCS 288(1), 2002, pp. 21-43, checked) relate certificate complexity, sensitivity, block sensitivity and decision tree depth. Hahn and Rofin (ACL 2024, arXiv 2402.09963, checked) show transformers are biased against high-sensitivity functions. That paper is about training dynamics, not agents reading policies, so the link is by analogy only.
- **Expected effect:** procedure level plausible 0.3 or more. Case level small, since SOPBench difficulty is mostly a hidden state, but this is the cleanest per-case structural measure. Risk: compiled programs are wrong or ambiguous.

### S4. Model-counting prior of "refuse" given what the request fixes (tier 4)
- **Measures:** the share of completions of the free variables that lead to refuse. It targets the label-driven part of SOPBench difficulty without reading the label.
- **Inputs:** compiled policy and request. No model.
- **Recipe:**
  1. Use the same boolean encoding as S3.
  2. Fix the variables the request implies.
  3. Count satisfying assignments of the "perform" outcome and the "refuse" outcome by enumeration. For `n > 25` use a model counter.
  4. Report `p_refuse = #refuse / (#perform + #refuse)` and also `-log2` of the smaller count.
  5. Weight completions uniformly, and also with per-variable priors from the pooled training set if available.
- **Level:** L0 after compile.
- **Varies by case:** yes, through the fixed variables.
- **Universality:** same as S3.
- **Status:** new. "Facts from message plus running the compiled rule" failed (it tried to read the label). This does not read the label, it only counts how many completions lead where.
- **Evidence:** own reasoning. Model counting is standard (Gomes, Sabharwal and Selman in the Handbook of Satisfiability; unverified).
- **Expected effect:** the large risk is that a uniform prior is unrelated to how the benchmark samples DB records. The oracle label gives rho 0.52, so a good prior could carry a fraction of it; I guess 0.05 to 0.1 at best.

### S5. Backdoor size of the policy (tier 4)
- **Measures:** the smallest set of variables that, once fixed, makes the rest follow by unit propagation. It targets simultaneous load, reframed as a structural quantity.
- **Inputs:** compiled policy. No model.
- **Recipe:** for each variable subset up to size 3, fix it and check whether the output is determined. Report the smallest size, per case with `K` fixed.
- **Level:** L0. Cheap when `n` is small.
- **Varies by case:** yes, through `K`.
- **Universality:** same as S3.
- **Status:** new in this form. Related to "condition packing density" but computes a different object.
- **Evidence:** Williams, Gomes and Selman, "Backdoors to Typical Case Complexity", IJCAI 2003, pp. 1173-1178 (checked, venue and pages only). SATBench (arXiv 2505.14615, checked) reports o4-mini at only 65.0% on hard UNSAT puzzles and names "condition omission" as a failure mode, which matches the skipped-check mechanism.
- **Expected effect:** highly correlated with S3 in small policies. Expect 0.0 to 0.1 beyond S3.

### S6. Data-flow graph between tool outputs and inputs, case-restricted (tier 2)
- **Measures:** longest dependency chain and width of the plan's call graph. It targets multi-step reliability.
- **Inputs:** tools plus policy. No model, or an LLM plan at L1.
- **Recipe:**
  1. Nodes are tool calls the plan needs.
  2. An edge exists when one tool's output type feeds another's required argument.
  3. Restrict to the nodes the request does not already satisfy (see S2).
  4. Report depth, width, and treewidth of the restricted graph.
- **Level:** L0 or L1.
- **Varies by case:** yes through the restriction.
- **Universality:** yes on all four.
- **Status:** variant of "data-flow metrics" and "tau2 tool-to-ID graph (+0.016)". The difference is that the graph is restricted per case. Expect a small gain, as the full-graph version gave +0.016.
- **Evidence:** Faith and Fate (arXiv 2305.18654, checked) uses computation graphs to measure compositional complexity and reports degrading performance as complexity rises. I did not verify the specific depth or width numbers.
- **Expected effect:** procedure level modest, case level near zero.

### S7. PDDL/STRIPS-style plan length and branching (tier 3)
- **Measures:** shortest plan length and branching factor of the encoded procedure. It targets step count, which is the tau2 signal (gold actions reach 0.40 to 0.47 as an oracle).
- **Inputs:** tools plus policy. An LLM does the encoding at L1.
- **Recipe:**
  1. An LLM encodes the tools as operators with preconditions and effects, and the request as the initial state.
  2. A pure-code solver (BFS) gets the plan length, the number of applicable operators per step, and a relaxed-plan size (ignore delete effects).
  3. Use the initial state from the request, so these vary by case.
- **Level:** L1 (one encoding call per case), cheap.
- **Varies by case:** yes. The initial state is case-specific.
- **Universality:** tau2 and new processes yes. SOPBench and banking yes, though plans are short, so little spread.
- **Status:** variant of "LLM plan with real tools". The difference is a formal solver, not an LLM-written plan, so plan length is the true shortest, not the model's guess.
- **Evidence:** PlanBench (arXiv 2206.10498, NeurIPS 2023 D&B, checked) reports that LLM plan generation falls short on classical planning domains. I did not check its numbers.
- **Expected effect:** tau2 case level could reach 0.2 to 0.3. Risk: encoding errors, and the LLM encoder reproduces the LLM-plan correlation, so a gain over c_lad is uncertain.

### S8. Dialogue state machine path count (tier 3)
- **Measures:** the number of distinct accepting paths, and the number of paths through rejecting states.
- **Inputs:** compiled program or plan. No model.
- **Recipe:**
  1. Build the control-flow graph of the compiled program.
  2. Count paths from entry to each terminal (perform, refuse, transfer).
  3. Per case, count only paths consistent with the request-fixed variables.
- **Level:** L0.
- **Varies by case:** yes through the fixed variables.
- **Universality:** same as S3.
- **Status:** variant of "compiled program length and branches (CMP+)". The difference is counting paths consistent with the request, not total branches.
- **Evidence:** own reasoning.
- **Expected effect:** highly collinear with S3 and S4.

### S9. Compression length of the compiled program under a local model (tier 4)
- **Measures:** how well the compiled program compresses, as a proxy for description length. It targets "algorithmic surprise".
- **Inputs:** compiled program. Local open model.
- **Recipe:** compute bits per token of the program text under a local base model. Subtract the same measure for a shuffled-statement version as a control.
- **Level:** L0 (local inference, no case reading).
- **Varies by case:** only by procedure.
- **Universality:** yes on all four after compile.
- **Status:** variant of "prior surprisal of rules" (PRI, 0.689 at procedure level only). This one scores the compiled code, not the policy text.
- **Evidence:** Delétang et al., "Language Modeling Is Compression" (arXiv 2309.10668, ICLR 2024 per memory, venue unverified, checked as arXiv paper) argues that LMs are general compressors.
- **Expected effect:** procedure level only. It adds nothing at the case level.

### S10. Type-system style check of the plan (tier 2)
- **Measures:** number of unresolved typed holes when the plan is type-checked. It targets wrong-tool and argument errors.
- **Inputs:** tools plus LLM plan.
- **Recipe:** treat each tool as a typed function with named input and output types. Run a type-checker over the plan and count unfilled arguments and type mismatches.
- **Level:** L1.
- **Varies by case:** yes.
- **Universality:** yes on all four.
- **Status:** variant of S1 and "argument provenance". Mostly redundant with S1 plus S6.
- **Evidence:** own reasoning.
- **Expected effect:** small.

### S11. Rule-chain depth from the data-flow of conditions (tier 3)
- **Measures:** depth of the chain "condition A needs the result of condition B".
- **Inputs:** policy and tools.
- **Recipe:** from the compiled program, link a condition to the earlier conditions or calls whose results it uses. Report the longest chain and the number of chain starts with `K` fixed.
- **Level:** L0 after compile.
- **Varies by case:** yes, through `K`.
- **Universality:** same as S3.
- **Status:** variant of "verification chain depth". The difference is that chain depth is computed from variable data dependencies in code, with `K` removing the ones the request already answers.
- **Evidence:** RuleTaker (Clark et al., arXiv 2002.05867, IJCAI 2020, checked) is a reasoning-depth benchmark. I did not verify the accuracy-by-depth numbers from the abstract page, which only gave 99% overall.
- **Expected effect:** small.

### S12. Request-to-condition coverage as a set cover (tier 3)
- **Measures:** how many of the policy's conditions the request mentions at all, as a share of all conditions.
- **Inputs:** policy plus request. Black-box LLM at L1 for the matching, or a local embedding.
- **Recipe:** one LLM call per case lists which conditions the message addresses. Report the covered fraction and the largest uncovered gating condition.
- **Level:** L1.
- **Varies by case:** yes.
- **Universality:** yes on all four.
- **Status:** variant of "clause retrieval margin" and "user assertion and action readiness". The difference is a coverage count, with no margins.
- **Evidence:** own reasoning.
- **Expected effect:** the best chance of a case-level gain on SOPBench. The risk is that the message says nothing about the hidden DB state, which is the issue behind the five failed label-reading attempts.

### S13. Noise sensitivity of the verdict to numeric thresholds (tier 5)
- **Measures:** how often the verdict flips if the request's numbers are perturbed within a small relative range, over the compiled program.
- **Inputs:** compiled program plus request numbers. No model.
- **Recipe:** perturb each numeric literal in the request by plus or minus 5% and 20%, run the compiled program with stubbed tool returns for all other variables, and report the flip rate.
- **Level:** L0.
- **Varies by case:** yes.
- **Universality:** SOPBench and banking yes (numeric thresholds). tau2 weaker. New process yes if numeric.
- **Status:** variant of "numeric boundary margin". The difference is a flip rate across perturbations, not a single margin.
- **Evidence:** own reasoning; noise sensitivity is a standard Boolean-function measure (O'Donnell, Analysis of Boolean Functions, 2014; unverified).
- **Expected effect:** small. Only matters where the thresholds are close.

### S14. Schema-conformance entropy of IDs across the tool set (tier 2)
- **Measures:** how many different ID formats exist among the tools, and how close two formats are (confusable IDs, for example user_id versus reservation_id).
- **Inputs:** tools.
- **Recipe:** extract ID-like argument names, regex-classify the example values, and report the number of formats and the mean edit distance between format templates.
- **Level:** L0.
- **Varies by case:** only by procedure and domain.
- **Universality:** yes on all four.
- **Status:** variant of "tool-schema complexity" and "wrong-tool confusion". Procedure-level only, so it cannot help the case metric.
- **Evidence:** own reasoning.
- **Expected effect:** near zero at case level.

## Quick screen (exploratory, not the protocol)

I could not do the two best ideas by pure code, so I ran a crude pure-code proxy of S1. It counts typed literals in the customer message or scenario (numbers, snake_case ids, upper-case codes, distinct formats) and correlates each count with the full-data IRT difficulty per domain. I ran it under the CPU lock. The code is in `research/screen/ideas8/structure/screen.py`.

| Benchmark | Feature | Mean within-domain Spearman |
|---|---|---|
| SOPBench | numeric literals | 0.157 |
| SOPBench | upper-case codes | 0.189 |
| SOPBench | distinct formats | 0.155 |
| SOPBench | typed literals total | 0.146 |
| SOPBench | message length (words) | 0.029 |
| tau2 | numeric literals | 0.223 |
| tau2 | typed literals total | 0.185 |
| tau2 | message length (words) | 0.267 |

- **SOPBench domains are inconsistent.** Numeric literals range from -0.107 (healthcare) to 0.372 (university). Library and university have only 66 and 42 cases, and two domains have a constant code count, so those cells are NaN.
- **tau2 is carried by telecom.** Telecom reaches 0.44 for numeric literals, and I would not trust that. The brief says telecom has only 3 procedures, so this may be the procedure effect.
- **This is not S1 itself.** I did not compare against the tool schema. The result says only that typed-literal counts are weak and domain-dependent.
- **No procedure-level control.** I did not regress out procedure identity, so some of these correlations may be procedure effects.

## Top 3

1. **S3 (sensitivity and certificate size at the known inputs) with S4 (counting prior) as a companion.** These are the only ideas in the list that vary by case through the request's fixed variables and carry a formal argument. SOPBench case difficulty is mostly a hidden perform/refuse state. S4 estimates how likely refuse is from the request-fixed variables, and S3 says how many checks the agent must get right. The compiled programs already exist, so the cost is near zero. The risk is that the benchmark samples DB records in a way a uniform prior ignores.
2. **S12 (request-to-condition coverage).** One cheap LLM call per case, universal, and it varies by case. It is the likeliest to add something over c_lad on tau2, where the oracle shows case-level structure exists.
3. **S7 (formal plan length and branching from a PDDL-style encoding).** The only idea that uses a solver, so it gives the true shortest plan, not a model-written guess. tau2 has no label-only cap, and its gold step count alone reaches 0.40 to 0.47. Plan length from a solver could recover part of that without gold data. The risk is that the LLM encoding errors reproduce what the LLM-plan feature already gives.

## Citations

| Source | Status |
|---|---|
| Hahn and Rofin, ACL 2024, arXiv 2402.09963 | Checked: title, authors, venue, claim |
| Faith and Fate, arXiv 2305.18654 | Checked: title and claim only. Venue and numbers not verified |
| ToolSandbox, arXiv 2408.04682 | Checked: the three scenario categories. Numbers not verified |
| SATBench, arXiv 2505.14615 | Checked: o4-mini 65.0% on hard UNSAT puzzles, failure modes named |
| Schema-Guided Dialogue, arXiv 1909.05855, AAAI 2020 | Checked: 16k conversations, 16 domains, zero-shot DST claim |
| Language Modeling Is Compression, arXiv 2309.10668 | Checked as arXiv paper. ICLR 2024 venue not checked |
| PlanBench, arXiv 2206.10498, NeurIPS 2023 D&B | Checked: venue and qualitative claim. No numbers extracted |
| RuleTaker, arXiv 2002.05867, IJCAI 2020 | Checked: title, venue, 99% on synthetic test data. Depth numbers not verified |
| Buhrman and de Wolf, TCS 288(1), 2002, pp. 21-43 | Checked via search |
| Williams, Gomes and Selman, IJCAI 2003, pp. 1173-1178 | Checked via search |
| O'Donnell, Analysis of Boolean Functions (2014); Handbook of Satisfiability model-counting chapter | Unverified |

Of the 14 ideas, 7 go beyond the seeds (S3, S4, S5, S8 to S10, S13, partly S12); the others are tier 1 to 2 seed items.
