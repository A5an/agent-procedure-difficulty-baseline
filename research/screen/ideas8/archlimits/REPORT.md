# Swarm agent `archlimits`, round 8 (6 Oct 2026)

Saved by the main session from the agent's final message.

## Catalog: failure mechanisms of LLMs and agents, turned into case features (agent `archlimits`)

I could not write files, so the whole catalog is in this message. The only file I created is the quick-screen script at `research/screen/ideas8/archlimits/screen.py`.

**Sign convention.** In the repo, `items.csv` column `b` is higher for easier cases (SOPBench should_succeed cases have mean b +1.64). A hazard feature should therefore correlate NEGATIVELY with `b`.

**Citation status.** I checked each paper below against its arXiv abstract page (or the paper's own abstract). Per-model numbers inside the papers I did not check. Those are marked unverified.

### Verified curves (the evidence base)

| Mechanism | Paper | Curve shape and quoted numbers |
|---|---|---|
| Position | Liu et al., Lost in the Middle, TACL 2023, arXiv 2307.03172 | U-shape in the position of the relevant document. Best at the start or end, "significantly degrades" in the middle. Exact percentages unverified. |
| Long context and no literal match | Modarressi et al., NoLiMa, ICML 2025, arXiv 2502.05167 | Good below 1K tokens. At 32K, 11 models fall below 50% of their short-context baseline. GPT-4o goes from 99.3% to 69.7%. The test removes literal word overlap between question and answer. |
| Number of instructions | Jaroslawicz et al., IFScale, arXiv 2507.11538 | 500 keyword instructions, 20 models. The best model reaches 68% at the maximum density. Three decay shapes were reported, tied to model size and reasoning. There is a bias toward earlier instructions. The shape names and thresholds are unverified. |
| Number of instructions, multiplicative | Harada et al., When Instructions Multiply, EMNLP 2025, arXiv 2509.21051 | Accuracy falls with the instruction count (up to 10 in text, up to 6 in code). A logistic regression on the count predicts unseen multi-instruction performance with about 10% error. |
| Same, the p^n form | "Curse of Instructions", ICLR 2025 (same group) | P(all) is about p^n. The p=0.9, n=10 example gives about 0.35. Source: a web summary only. I did not open the paper and have no arXiv id, so this is unverified. |
| Multi-turn | Laban et al., arXiv 2505.06120 | Average drop of 39% across six tasks from single-turn to multi-turn. Most of the extra failure is unreliability, not lost capability. Early wrong assumptions are never revised. |
| Proactive interference | Wang and Sun, arXiv 2506.08184 | Accuracy on the latest value falls log-linearly toward zero as the number of interfering overwritten items grows. The effect is not due to context length. |
| Multi-hop | Press et al., arXiv 2210.03350 | Single-hop accuracy improves faster with scale than multi-hop. The gap between "all sub-answers right" and "composed answer right" does not shrink with scale. |
| Composition depth | Dziri et al., Faith and Fate, arXiv 2305.18654 | Accuracy decays with computation-graph depth and width and with the number of digits. |
| Reversal | Berglund et al., Reversal Curse, arXiv 2309.12288 | GPT-4 gets 79% in the direction learned in training and 33% in the reverse. This is about training, not in-context use. |
| Entity tracking | Kim and Schuster, ACL 2023, arXiv 2305.02363 | Models infer an entity's final state after a sequence of operations. Only code-pretrained GPT-3.5 models do it without fine-tuning. The accuracy-per-operation numbers are unverified. |
| Horizon | Sinha et al., ICLR 2026, arXiv 2509.09677 | Small gains in per-step accuracy compound exponentially in the length of task a model can finish. Visible earlier errors raise the error rate (self-conditioning). |
| Tool count | Gan and Sun, RAG-MCP, arXiv 2505.03275 | With a large tool pool, selection accuracy is 13.62% for the baseline and 43.13% with retrieval. A secondary web summary quotes 78% at 10 tools. That figure is unverified. |
| Consistency | τ-bench, arXiv 2406.12045 | GPT-4o succeeds on under 50% of tasks. pass^8 is below 25% in retail. |
| Negation | Truong et al., arXiv 2306.08189 | Models are insensitive to negation, and scale does not fix it. No usable curve. |
| Sycophancy | Sharma et al., arXiv 2310.13548 | Present in five assistants. Preference data favours agreement. No dose-response curve. |
| Date arithmetic | Fatemi et al., Test of Time, arXiv 2406.09170 | Accuracy depends on problem structure, size, question type and fact order. Numbers unverified. |

I found no verified accuracy curve against a measurable quantity for three seeds: counting, digit copying and tokenization of IDs, and over-compliance. Those hazard terms have to be fitted on training data. They cannot be taken from the literature.

### The catalog (tier 1 is most obvious, tier 5 least)

**A1. Active-obligation count with a p^n survival (tier 1).**
- Measures: how many atomic obligations are active in THIS case, not in the whole policy. Targets skipped checks under simultaneous load.
- Inputs: policy, request. Model: black-box API.
- Recipe:
  - One L1 call: "list the atomic checks and actions that must be done for this request, one per line".
  - Count them as n_active.
  - Predicted survival is p^n_active, with p fitted on training folds.
- Level: L1, one cheap call per case.
- Varies by case: yes, because different requests activate different branches. Only partly in SOPBench, where the request names one action.
- Universality: all four, since it needs only policy and request.
- Status: variant of the tried "simultaneous load". It differs by counting only active obligations and using a fixed multiplicative form.
- Evidence: Harada et al. 2509.21051 and the unverified Curse-of-Instructions p^n form.
- Expected effect: case level small positive, procedure level moderate. Risk: it correlates with the PLAN counts already tried.

**A2. Visible-context length at decision time (tier 1).**
- Measures: tokens of dialogue, tool outputs and policy before the first decisive action. Targets long-context and multi-turn degradation.
- Recipe: code only. Count tokens of policy + schemas + case text. For tau2, the expected tool-output volume comes from the L1 plan.
- Level: L0.
- Varies by case: tau2 yes. SOPBench barely, since the message is short and the policy is fixed per procedure.
- Universality: all four.
- Status: variant of the tried text length. It excludes the policy and counts only what lies between the rule and the decision.
- Evidence: NoLiMa 2502.05167 and Laban 2505.06120.
- Expected effect: small. The policies here are far below 32K tokens, so the curve is flat. Risk: high.

**A3. Slot-fill gap at the first message (tier 2).**
- Measures: required tool arguments that the request does not supply. Targets the premature answer or guess found by Laban (wrong early assumption, then no recovery).
- Inputs: tool schemas, request. Model: black-box API.
- Recipe:
  - Take the required parameters of every tool in the L1 plan from the schemas.
  - Ask one L1 call: "for each required argument, quote the span in the request that gives it, or write MISSING".
  - Feature = number of MISSING plus the number of arguments that need a lookup.
- Level: L1.
- Varies by case: yes, because users give different details.
- Universality: all four.
- Status: new. Argument provenance is on the known list. This is the first-message gap per case, not the provenance of values.
- Evidence: Laban 2505.06120 (39% drop under underspecification). The link to this benchmark is own reasoning.
- Expected effect: case level 0.05 to 0.15 on tau2. Risk: SOPBench messages are almost fully specified.

**A4. Literal-overlap hazard between request and governing clause (tier 2).**
- Measures: low word overlap between the request's words and the clause that gates it. Targets NoLiMa-style latent retrieval.
- Recipe:
  - Take the rare content words of the request.
  - For each policy clause compute BM25 or Jaccard overlap.
  - Feature = max overlap, or the overlap with the L1-identified gating clause. Low overlap means more hazard.
- Level: L0, pure code.
- Varies by case: yes.
- Universality: all four.
- Status: variant of the "clause retrieval margin" idea. It uses literal rather than embedding similarity, which is what NoLiMa found to matter.
- Evidence: NoLiMa.
- Expected effect: my screen found 0.14 on SOPBench but the sign is wrong for a hazard (see the quick screen).

**A5. Identifier and digit load (tier 2).**
- Measures: digits, ID-like tokens and distinct numbers in the request, and tokens-per-ID under the agent's tokenizer. Targets copy and transcription errors and weak digit handling.
- Recipe: regex plus a tokenizer, no inference. Count digits, ID tokens (5 or more characters with a digit), their total length, and distinct numbers.
- Level: L0.
- Varies by case: yes.
- Universality: all four.
- Status: new as a feature.
- Evidence: Faith and Fate 2305.18654 supports digit count only for multiplication. Copying curves unverified.
- Result: tested, see the quick screen. It is a proxy for the perform label, not a hazard.

**A6. Calendar-hardness count (tier 2).**
- Measures: date and time operations the case needs, and how hard each is. Targets the step type with 42% reliability.
- Inputs: policy, request. Model: black-box API.
- Recipe:
  - One L1 call extracts each date comparison (for example "within 24 hours", "30 days after", business days, timezone).
  - Code scores each operation as hard if it crosses a month or year boundary or involves a leap day, a timezone, or business days.
  - Feature = the weighted count.
- Level: L1.
- Varies by case: yes. Dates come from the case and the database.
- Universality: all four.
- Status: variant of "numeric boundary margin". The margin is the distance to the threshold. This is the calendar structure of the computation.
- Evidence: the 42% datetime step reliability is from the brief. Test of Time 2406.09170 supports "structure matters", with no number.
- Expected effect: case level 0.05 to 0.1 where the procedure has date rules. Risk: few cases involve dates.

**A7. Entity and state-tracking load (tier 3).**
- Measures: how many entities are simultaneously alive and how many state changes occur during the interaction. The scoring is typed (entity, state, update).
- Recipe: one L1 call listing entities, their states and the operations on them. Feature = entities × updates, or the longest chain of updates to one entity.
- Level: L1.
- Varies by case: tau2 yes (modifications, multi-item orders). SOPBench no.
- Universality: tau2 and banking yes. SOPBench weak. A new process yes.
- Status: new.
- Evidence: Kim and Schuster 2305.02363 supports the mechanism. I have no curve.
- Expected effect: moderate on tau2. Risk: it correlates with write counts.

**A8. Interference among superseded values (tier 3).**
- Measures: how many same-type values in the case context are overwritten, corrected, or sit near the relevant one. Targets stale-value retrieval. Examples: a user who changes the date, a tool output that lists 6 orders, 3 card numbers in the dialogue.
- Recipe: code only. Run a regex NER for the types date, money, ID, name and count. For each type, interference = the number of distinct values. Add 1 for each correction phrase ("actually", "instead", "no wait").
- Level: L0.
- Varies by case: yes.
- Universality: tau2 and banking yes. SOPBench weak. A new process yes.
- Status: new. It is not the same as the known "interference among similar conditions", which concerns policy clauses. This one concerns the values in the case.
- Evidence: Wang and Sun 2506.08184 (log-linear decline). I did not verify the effect size.
- Expected effect: small to moderate on tau2 only.

**A9. Typed survival product over the planned trajectory (tier 3).**
- Measures: the L1 plan broken into steps, each tagged with a step type. Predicted survival is the product of per-type reliabilities. The brief gives authentication 88%, datetime 42% and relation 46%.
- Recipe:
  - Take the PLAN output already available.
  - Tag each step as authentication, lookup, write, datetime, relation, calculation, or refuse-check.
  - Compute log S = sum over steps of log r_type.
  - Fit r_type on training folds only. The typed product is the feature.
- Level: L1.
- Varies by case: yes, whenever the plan differs.
- Universality: all four.
- Status: variant of PLAN counts. PLAN used raw counts, not typed survival with fitted per-type reliabilities.
- Evidence: Sinha et al. 2509.09677. Also own reasoning, since the 88/42/46 figures come from our data.
- Expected effect: case level likely 0.25 to 0.3 on tau2. SOPBench is capped near 0.2. Risk: it overlaps with c_lad.

**A10. Hop depth of dependent lookups (tier 3).**
- Measures: the longest chain in which one tool's output feeds another's input. Targets the compositionality gap.
- Recipe: from the L1 plan, build a dependency graph over tool calls. Feature = the longest path and the number of hops that need information not in the request.
- Level: L1.
- Varies by case: yes.
- Universality: all four.
- Status: variant of the known data-flow metrics. It is a per-case longest path, not a per-procedure metric.
- Evidence: Press et al. 2210.03350 (gap does not shrink with scale).
- Expected effect: small to moderate.

**A11. In-context reversal direction (tier 4).**
- Measures: whether the case enters a rule from the side opposite to the one the policy states. Example: the policy says "Gold members may X" and the case gives X and asks which tier applies.
- Recipe:
  - One L1 call classifies each gating clause as forward (the case's known fact matches the clause's subject) or reverse (the case matches the clause's object).
  - Count the reverse clauses.
- Level: L1.
- Varies by case: yes.
- Universality: all four.
- Status: new.
- Evidence: Berglund et al. 2309.12288 (79% vs 33%) is about training. In-context reversal is much weaker (own reasoning, unverified). Expect a small effect. Risk: high.

**A12. Tool-set confusability around the needed tools (tier 4).**
- Measures: for each tool the plan needs, the number of other tools whose schema descriptions are nearly the same.
- Recipe: embed or lexically compare tool descriptions. Feature = the mean number of near-duplicates of the needed tools.
- Level: L0 plus the plan.
- Varies by case: yes, via which tools are needed.
- Universality: all four.
- Status: variant of the known "wrong-tool confusion". It is weighted by the tools this case actually needs.
- Evidence: RAG-MCP 2505.03275 (13.62% to 43.13%), on a pool of far more tools than these domains hold.
- Expected effect: small. Risk: the domains have 10 to 30 tools, so the curve is flat.

**A13. Mechanistic hazard model (tier 5, the main idea).**
- Measures: a predicted failure probability per case, assembled from the mechanism features above using shapes from the published curves.
- Recipe:
  1. For each case compute the quantities q_m (see the model below).
  2. Per mechanism the survival is s_m(q) = exp(−λ_m · g_m(q)). The shapes g_m come from the literature: g = n for the instruction count (p^n form), g = log(1 + k) for interference (log-linear form), g = depth for composition (decay with depth).
  3. Predicted failure = 1 − Π s_m. The case score is the rank of that failure probability.
  4. Zero-training version: all λ_m = 1 after scaling each q_m to [0, 1] across the training folds.
  5. Calibrated version: fit the non-negative λ_m (about 6 to 8 numbers) by non-negative logistic regression on the training folds only, using IRT difficulty as the target. Folds are leave-one-procedure-out or leave-one-domain-out, matching the protocol. Constrain each λ_m ≥ 0 so a mechanism cannot flip sign and fit the label.
- Level: L1, with 2 to 3 cheap calls per case.
- Varies by case: yes, per case through the quantities.
- Universality: all four, because each quantity comes from policy, schemas and request.
- Status: new as a combined model. Its ingredients A1, A6, A8, A9 and A10 are mostly variants of tried ideas.
- Evidence: the literature table above for the shapes. The combination itself is own reasoning.
- Expected effect: below.

### Quantities that vary between cases of the same procedure

These are the only inputs that can help at the case level on SOPBench, where the procedure-level cap is near 0.2.

- Digits and numbers in the request (A5).
- Which branch of the policy the request activates (A1).
- Dates in the request and in the database (A6).
- The number of listed tool outputs in tau2 (A8).
- The number of distinct values in the request.
- Which tools the plan needs (A12).

Quantities that are constant within a procedure and therefore useless for SOPBench case ranking: policy length, total rule count, tool pool size, and tool-description similarity of the full set.

### Expected effect of the hazard model

- tau2: possibly 0.25 to 0.35, mostly from A9 and A10 on plan structure. It will correlate strongly with c_lad.
- SOPBench: stays near 0.2, bounded by the facts. The hidden perform/refuse label is not readable from the message (five earlier attempts failed).
- Main risk: the typed-survival and hop features are plan features in new clothes. The incremental gain over c_lad may be near zero.
- Second risk: calibration with only 3 telecom procedures can overfit. Fit λ with a ridge penalty and report the zero-training variant alongside.

### Optional quick screen (pure code, exploratory, not the protocol)

Features were computed on SOPBench (7 domains, 830 cases) and tau2 (3 domains). I computed each feature's Spearman correlation with full-data IRT `b` per domain, then averaged. Results for A5 and A4:

| Feature | SOPBench raw, mean over domains | SOPBench within-procedure | tau2 raw, mean |
|---|---|---|---|
| digit count (A5) | +0.188 | +0.105 | +0.240 (airline 0.25, retail 0.03, telecom 0.44) |
| number of distinct numbers | +0.196 | +0.159 | +0.166 |
| ID-token count | +0.179 | +0.103 | -0.057 |
| ID-token length | +0.160 | +0.099 | -0.147 |
| lexical overlap request vs policy (A4) | +0.143 | +0.064 | -0.099 |
| request word count | -0.042 | +0.030 | +0.076 (airline 0.41, retail 0.44, telecom -0.63) |

How to read this:
- The sign is OPPOSITE to the hazard hypothesis on SOPBench. More digits, more numbers and more overlap go with higher `b`, meaning easier cases.
- The likely cause is the perform label. Cases that must be performed carry full parameters (amounts, usernames), so numbers mark should_succeed cases. These features are proxies for the hidden label, not mechanistic hazards.
- The within-procedure column keeps about half the effect (0.105 and 0.159), so part of the signal varies within a procedure.
- Per-domain values are unstable (library 0.37, hotel and healthcare about 0 or negative).
- On tau2 the signal is mixed. Within-procedure values are not defined, because each tau2 task is its own procedure. Telecom has only 3 procedures, so do not trust it.
- This does NOT count as evidence for a hazard model. If anything it shows that surface text features on SOPBench mostly track the perform label. I did not use A5 or A4 as predictors of failure, and I did not try other signs or variants after seeing the numbers.

### Top 3

1. **A9, the typed survival product.** It is the best-grounded idea: Sinha et al. support compounding per-step error, and our own data give typed per-step reliability (88/42/46). It varies by case wherever the plan varies. It reuses the PLAN output already paid for, so the added cost is a tagging step. It should help on tau2 and does little for SOPBench, where the hidden label dominates.
2. **A13, the hazard model as a whole.** It lets several weak mechanism features combine under a monotone, non-negative form, which limits overfitting on 4 scenarios. The zero-training variant (all weights equal, shapes from the literature) is a clean negative or positive control. The calibrated variant fits under 8 numbers on training folds.
3. **A3, the slot-fill gap.** It is the only mechanism here that is new relative to the tried list, universal, and tied to a measured effect (Laban's 39% drop under underspecification). It needs one L1 call against the tool schemas. Its main limit is that SOPBench messages are fully specified, so it may only help on tau2.

### Caveats

- I read only abstracts, via a fetch tool that summarises pages. Curve numbers beyond those quoted above are not verified.
- The p^n form comes from a secondary summary of the ICLR 2025 paper. I did not find or open the paper itself.
- Over-compliance, counting and ID-copy curves have no verified source. Do not cite them.
