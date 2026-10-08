# Swarm agent `outdist`, round 8 (6 Oct 2026)

Saved by the main session from the agent's final message.

# Output-distribution signals for predicting case difficulty: catalog of 15 ideas (agent `outdist`, round 8)

I wrote no files except `ncd_screen.py` in `prototype/screen/ideas8/outdist/`; the catalog is in this message. I ran one quick screen, a pure-code compression feature (results at the end).

## Map of the family, for the user

"Token entropy" is the average uncertainty of the model's next-word distribution. The whole family answers one question: what does the model's own probability distribution say about a case? The same question can be asked at five levels.

1. **One token.** Surprisal is -log p(token) and entropy is the spread of the distribution. Perplexity is the exponentiated mean surprisal of a text. Tried here and weak.
2. **A text.** Perplexity of the policy (the PRI idea, tried) or of the request (new, ideas 1 and 9).
3. **A sampled answer.** Draw k answers and measure how much they disagree (self-consistency, semantic entropy). This needs no logprobs.
4. **The model's opinion of itself.** Verbalized confidence, P(True), P(IK).
5. **The model's reaction to edits.** Paraphrase, delete a clause, swap a model. This is where I put the least obvious ideas.

### API reality as of today (6 Oct 2026)

| Source | Logprobs | Evidence |
|---|---|---|
| Anthropic Messages API | None. Only sampling-based and verbalized signals work. | Web search result, not Anthropic docs. |
| Vertex Gemini 2.5 (Flash) | Yes: `response_logprobs=True`, `logprobs` 1 to 20. | Google developer blog. Forum reports of inconsistent availability. |
| Vertex Gemini 3.x (3.1 Pro, 3.6 Flash) | None. | Google replied on 5 Aug 2026 that logprobs are "no longer returned for 3.X models" ("WAI", working as intended). |
| OpenAI chat | Yes (not re-checked for reasoning models). | Search result. |
| Local MLX model (Qwen3-8B-Base, as in PRI) | Full logits at every position, free, with teacher forcing. | Own knowledge. |

The white-box and PMI ideas are therefore local-model only. The sampling ideas work on any API.

### Verified evidence (checked this session)

- **Semantic entropy (SE).** Farquhar, Kossen, Kuhn, Gal, Nature 2024. Averaged over 30 task-model pairs with 10 samples each: SE AUROC 0.790, naive entropy 0.691, P(True) 0.698, embedding regression 0.687. Source: PMC11186750. The target is whether the answer is wrong.
- **Original SE paper.** Kuhn, Gal, Farquhar, ICLR 2023 spotlight, arXiv 2302.09664. It says SE beats predictive entropy. I did not get its numbers (unverified).
- **P(IK).** Kadavath et al. 2022, arXiv 2207.05221, from the PDF table.
  - A 52B P(IK) classifier trained on TriviaQA reaches AUROC 0.864 on TriviaQA.
  - Trained on TriviaQA only, it transfers poorly: LAMBADA 0.606, Python function synthesis 0.687, GSM8K 0.624. The GSM8K row layout was slightly ambiguous when parsed.
  - Trained on all tasks except GSM8K, it reaches 0.873, 0.853, 0.881 and 0.752 on TriviaQA, LAMBADA, Python and GSM8K.
  - The abstract reports good P(True) calibration. I did not check calibration numbers.
  - This is the most relevant warning for us: a learned "do I know" signal does not carry over to a new domain.
- **Verbalized confidence.** Xiong et al., ICLR 2024, arXiv 2306.13063. Models are overconfident when they verbalize. White-box methods beat black-box ones only modestly, and the AUROC range quoted is 0.522 to 0.605. This comes from the abstract summary; I did not re-check the exact table.
- **Agent self-consistency.** Mehta, arXiv 2602.11619, an ICML 2026 workshop paper (weak venue).
  - There are 2.3 to 4.2 distinct action sequences per 10 runs.
  - Failure-detection AUROC is 0.62 to 0.78.
  - Tasks with at most 2 unique paths are 82 to 87% correct; tasks with at least 4 paths are 41 to 65% correct.
  - Data are HotpotQA (200 questions, 8,000 runs) and SWE-bench (50 tasks).
  - It needs 10 full agent runs, so it is not directly usable by us.
- **UQ for function calling.** Ye et al., arXiv 2604.22985. SE brings no clear advantage over single-sample logprob methods for tool calls. Clustering function-call outputs by AST helps the multi-sample methods, and keeping only semantically meaningful tokens helps the single-sample ones. No AUROC numbers were seen.
- **Reasoning length.**
  - Palod et al., arXiv 2509.07339: intermediate-token length and ground-truth A* trace length "only loosely correlate", mostly near the training distribution.
  - Su et al., arXiv 2505.00127: models overthink easy problems and underthink hard ones.
  - So reasoning length is a weak difficulty signal.
- **Prompt perplexity.** Gonen et al., Findings of EMNLP 2023, arXiv 2212.04037. Lower prompt perplexity goes with better task performance across OPT and Bloom. I saw no coefficients.
- **PMI.** Holtzman et al., arXiv 2104.08315. Domain-conditional PMI fixes surface-form competition and gives consistent gains. I did not check numbers.
- **Perplexity for attacks.** Alon and Kamfonas, arXiv 2308.14132. A Light-GBM on perplexity plus length detects most adversarial prompts. No numbers were visible.
- **Difficulty in hidden states.** Lee et al., arXiv 2510.05969. Linear probes on the final-token state model the difficulty of math problems, and particular last-layer attention heads fire differently on easy and hard problems. No numbers seen.
- **Refusal direction.** Arditi et al., arXiv 2406.11717. One direction mediates refusal across 13 chat models up to 72B parameters.
- **Semantic entropy probes (SEP).** Kossen et al., arXiv 2406.15927. Hidden states approximate SE from a single generation at almost zero cost. The paper claims better out-of-distribution generalization than accuracy probes; I did not verify the numbers.
- **Kernel language entropy.** Nikitin et al., arXiv 2405.20003. A soft-kernel version of SE.
- **Prompt sensitivity.** Razavi et al., arXiv 2502.06065. Existing methods struggle to predict prompt sensitivity.
- **Q-DAPS.** Mozafari et al., ACL 2026 long paper, arXiv 2605.12398. Question difficulty as the entropy of plausibility scores over candidate answers, tested on TriviaQA, NQ, MuSiQue and QASC. The ACL page returned 404, so I have no numbers.
- **Unverified.** A search snippet said response entropy gives ROC-AUC 0.73 for difficulty on biology but 0.55 on math. I could not identify the source paper. Use it only as direction: entropy works for knowledge tasks and fails for reasoning tasks.

The pattern across the literature is that these signals predict whether a specific answer is wrong, usually through hallucination. Nobody has shown that they predict IRT case difficulty of procedure compliance. All our recipes are therefore extrapolations, and the expected effects below are my estimates.

## Catalog

All local-model recipes assume Qwen3-8B-Base (or similar) on MPS or MLX, which was used for PRI. Universality for SOPBench, tau2, banking and a new process is "yes" unless stated.

---

### OD1. Conditional request surprisal gain, `reqgain` (tier 1)

- **Measures.** How much the policy and tools explain the request. A request unlike anything the SOP anticipated suggests an edge case or off-script behavior.
- **Targets.** Off-path handling and skipped checks caused by an unexpected request.
- **Inputs.** Policy, tools, request. Local base model.
- **Recipe.**
  - Compute NLL_cond = mean NLL of the request tokens given the prompt "[policy][tools] Customer:", and NLL_unc = mean NLL given only "Customer:".
  - Use gain = NLL_unc minus NLL_cond (this is PMI between request and documentation) and also the raw NLL_cond.
  - Z-score within procedure.
- **Level and cost.** L1 (reads the case). Two forward passes per case, about 1 s on MPS.
- **Varies by.** Case.
- **Universality.** Yes for all four. SOPBench uses the message text, tau2 and banking use the scenario text. It is somewhat odd on tau2, where the scenario is written to a user simulator and not spoken by a customer.
- **Status.** New. PRI is surprisal of the policy, procedure-level. This is the request given the policy, case-level. The conditioning (PMI instead of raw perplexity) follows Holtzman's PMI_DC logic.
- **Evidence.** Own reasoning. The nearest support is Gonen's prompt perplexity result, and Alon's perplexity-based attack detection for atypical text.
- **Expected effect.** Case level about 0.00 to 0.10, because the quick screen (below) found that compression-style atypicality mostly reproduces length. High risk that it fails on SOPBench, where the hidden perform/refuse state is not in the text.

### OD2. Sampled-plan agreement (self-consistency of plans), `planconsist` (tier 1)

- **Measures.** Disagreement among k independently sampled plans for this case, as a cheap proxy for Mehta's behavioral consistency.
- **Targets.** Genuine ambiguity in what to check or do. It does not catch a model that is confidently wrong.
- **Inputs.** Policy, tools, request. Any API (no logprobs needed): Sonnet, Gemini 3.x or a local model.
- **Recipe.**
  - Sample k=5 to 8 times at temperature about 0.8 with the existing PLAN prompt (ordered list of tool calls with arguments, plus a final perform or refuse).
  - Canonicalize each plan to a sequence of (tool, sorted argument keys).
  - Compute the mean pairwise normalized edit distance. Also compute the number of distinct canonical sequences, and the entropy of the perform/refuse vote.
  - Features: mean distance, distinct count, vote entropy.
- **Level and cost.** L1. 5 to 8 calls per case, roughly 40 to 100 times a single cheap call.
- **Varies by.** Both case and procedure.
- **Universality.** Yes for all four.
- **Status.** Variant of PLAN and of "two planners disagree" (n.s.). The difference is within-planner sampling disagreement (aleatoric to the sampler) with AST-style clustering, not two fixed planners. It is also distinct from "compile disagreement" because it samples plans, not programs.
- **Evidence.** Mehta 2602.11619: AUROC 0.62 to 0.78 for failure detection, with 82 to 87% accuracy for at most 2 paths versus 41 to 65% for at least 4 paths. That was on agent runs, not on plans. Ye et al. say AST clustering helps multi-sample UQ.
- **Expected effect.** Case level +0.02 to 0.06 over c_lad if the planner is good. The risk is that LLM plans are overly deterministic, which collapses the variance. Procedure-level gains are likely larger.

### OD3. Perform/refuse vote split and P(True) of the decision, `decvote` (tier 2)

- **Measures.** Whether the model is split about whether the action should be taken, plus the model's own P(True) that all preconditions hold.
- **Targets.** "Act despite a failed condition" (78% of must-refuse failures).
- **Inputs.** Policy, tools, request. API, or local for the P(True) part.
- **Recipe.**
  - Prompt: "Given the policy and message, answer whether the action should go ahead, assuming nothing is known beyond the message."
  - Take the empirical vote entropy over 8 samples.
  - Locally, compute logP(True) minus logP(False) of the P(True) template (Kadavath: "Is the proposed answer: True/False").
- **Level and cost.** L1.
- **Varies by.** Case.
- **Universality.** SOPBench yes (decision is the label). tau2 and banking partial: there is no clean perform/refuse for many scenarios, so use it as "can the main request be fulfilled".
- **Status.** Variant of the failed "label from message" attempts (AUC 0.49 to 0.55). The difference is that I use the uncertainty (vote entropy, P(True) margin) as the feature and not the label. On SOPBench the message mostly cannot reveal the label, so the entropy is likely high everywhere and uninformative.
- **Evidence.** Kadavath P(IK) 0.864 in-domain but 0.606 to 0.687 transfer to other tasks. Farquhar: P(True) 0.698, SE 0.790.
- **Expected effect.** Case level about 0.00 to 0.05. High risk of failure.

### OD4. Verbalized difficulty and confidence of an agent, `verbal` (tier 1)

- **Measures.** Ask the model, "what is the probability that a competent agent finishes this case without a mistake?" and "which step will fail?".
- **Targets.** General difficulty, by self-report.
- **Inputs.** Policy, tools, request. API.
- **Recipe.** Elicit a number 0 to 100 three times with different wordings. Take the mean, and the spread as a second feature. Xiong's recommendations: ask for the probability, use multiple prompts, and aggregate.
- **Level and cost.** L1, 3 cheap calls.
- **Varies by.** Both.
- **Universality.** Yes.
- **Status.** The ADeLe levels already contain verbalized difficulty on 18 demands, so this is an overlapping variant. The only new part is "probability that a competent agent fails", and it is probably redundant with c_lad.
- **Evidence.** Xiong 2306.13063: models are overconfident, AUROC 0.522 to 0.605 range for confidence methods (abstract-level).
- **Expected effect.** +0.00 to 0.02 over c_lad. Mostly a sanity-check feature.

### OD5. Hidden reasoning length at a fixed effort, `thinklen` (tier 2)

- **Measures.** Number of thinking tokens a reasoning model spends deciding whether and how to handle the case. This differs from ENT, which used the entropy profile of a think-aloud trace and hurt c_lad.
- **Targets.** Effective load, the number of checks and branches the model must work through.
- **Inputs.** Policy, tools, request. Reasoning API (Gemini 3.x reports thinking token counts in usage metadata; Anthropic extended thinking returns the thinking blocks and I did not verify a separate token count).
- **Recipe.**
  - Ask a verification-only question: "List every precondition that must hold before the action is taken and whether the message settles it. Do not take the action."
  - Fix the thinking budget or effort level, run it 3 times, and take the median thinking token count.
  - Also record the answer length.
- **Level and cost.** L1. 3 reasoning calls per case, the most expensive of the cheap signals.
- **Varies by.** Both.
- **Universality.** Yes.
- **Status.** Variant of ENT (different quantity, length not entropy, and a verification task not think-aloud solving).
- **Evidence.** Palod 2509.07339: only loose correlation with true complexity. Su 2505.00127: models overthink easy and underthink hard problems. Both argue this will be a weak signal.
- **Expected effect.** Case level about 0.00 to 0.05. Probably dominated by message length.

---

### OD6. Paraphrase flip rate, `paraflip` (tier 2)

- **Measures.** How often the sampled plan or decision changes when the request is reworded without changing its content.
- **Targets.** Fragile reading of the request, such as a key fact buried in a clause.
- **Inputs.** Request, policy, tools. API.
- **Recipe.**
  - Generate 4 meaning-preserving rewrites of the request (change order of sentences, voice and register, keep every fact).
  - Run the plan prompt once on each and on the original, at temperature 0 or 0.3.
  - Features: fraction of canonical plans differing from the original, mean edit distance.
- **Level and cost.** L1. 1 rewrite call plus 5 plan calls per case.
- **Varies by.** Case. This is the one family that tests the message itself and not the procedure.
- **Universality.** Yes. On tau2 and banking, rewrite the scenario instruction.
- **Status.** New for this project. It differs from OD2 because the randomness comes from the input and not from the sampler.
- **Evidence.** Search results say paraphrase robustness correlates with correctness, but I did not verify a specific paper or number. Razavi 2502.06065 found that existing methods "struggle" to predict prompt sensitivity (verified), so a negative prior.
- **Expected effect.** Case level +0.00 to 0.04. Cost is high relative to the gain.

### OD7. Tool-choice entropy along the written plan, `toolent` (tier 3)

- **Measures.** For each step of an LLM-written plan, teacher-force the plan into a local model, take the next-step distribution restricted and renormalized over the K tool names, and compute its entropy and the probability margin of the planned tool over the runner-up.
- **Targets.** Wrong-tool confusion and confusable tool schemas.
- **Inputs.** Policy, tools, request, plan. Local model.
- **Recipe.**
  - Prompt: "[tools] [policy] Request. Agent calls:" then the plan, one call per line.
  - At each tool-name position, take logits over the first token of each tool name (use a disambiguating prefix when names share a first token).
  - Features: sum and max of per-step entropy over K, minimum margin, number of steps with margin below 1 nat.
- **Level and cost.** L1 (needs the plan). One forward pass per case plus the plan call.
- **Varies by.** Case, through the plan.
- **Universality.** Yes (any tool list).
- **Status.** Variant of "small-model first-action logprob margin" (failed) and "wrong-tool confusion" (listed). The differences are (a) all steps of a written plan, not only the first action, (b) renormalization over the tool list so the entropy is a K-way ambiguity, and (c) teacher forcing, so a step's difficulty is conditional on the plan so far.
- **Evidence.** Ye et al. 2604.22985: single-sample logprob methods are competitive for function calling, especially when only meaningful tokens are used (verified, no numbers).
- **Expected effect.** Case level +0.00 to 0.04. Risk: the first-action version already failed, and tool-choice is rarely what fails (most failures are skipped checks).

### OD8. Plan-step grounding (PMI of each plan step with the policy), `stepgrounding` (tier 3)

- **Measures.** For each step of the LLM-written plan, logP(step | policy, request) minus logP(step | request only). Steps with low gain are "ungrounded", meaning the plan includes actions or checks the policy did not call for, or omits grounded ones.
- **Targets.** Hallucinated checks and missing mandatory checks (skipped checks).
- **Inputs.** Policy, request, plan. Local model.
- **Recipe.**
  - Two passes over the plan with and without the policy in the context.
  - Per-step PMI, then aggregate: mean, minimum, count of steps with PMI below a small threshold.
  - Mirror version for omission: for each sentence of the policy that contains a modal word ("must", "only if"), compute logP(sentence | plan) against logP(sentence | nothing) to find policy conditions the plan seems to ignore.
- **Level and cost.** L1. 2 to 3 forward passes per case.
- **Varies by.** Case, via the plan.
- **Universality.** Yes. It needs a plan, which every procedure can have. A new company process just needs policy text.
- **Status.** New. PRI measured the surprisal of rules alone. This measures how much each plan step is explained by the policy, and the omission version is a data-free test of "the plan skipped a rule".
- **Evidence.** Own reasoning. Min-K% style detection is related, but I did not verify any source for it.
- **Expected effect.** Case level +0.02 to 0.06 if it picks up skipped conditions. Risk: the plan tends to copy policy words, so PMI saturates.

---

### OD9. Within-procedure atypicality (conditional compression of the request given sibling requests), `sibling_nll` (tier 3)

- **Measures.** How unlike its same-procedure siblings this request is. This is a pure within-procedure feature, which is what the SOPBench cap requires.
- **Targets.** Off-distribution cases within a procedure.
- **Inputs.** Request and the other (unlabeled) requests of the same procedure.
- **Recipe.**
  - The LM version: NLL of the request given up to 10 sibling requests as few-shot context, subtracted from NLL with no context. Local model.
  - The pure-code version: zlib bytes of (siblings plus request) minus bytes of siblings, divided by bytes of the request. This is the quick screen below.
- **Level and cost.** L0 or L1 (reads cases, transductive: it uses the texts of other cases but no labels).
- **Varies by.** Case only. By construction it is mean-centered inside a procedure.
- **Universality.** Needs several cases per procedure. SOPBench has about 30 to 100 per procedure. Tau2 and banking have a few. A new company process needs a pool of historical requests, which is not always available.
- **Status.** New.
- **Evidence.** Exploratory screen below: the compression version is about the same as length (tau2, banking) and slightly negative on SOPBench. The LM version is untested.
- **Expected effect.** Case level about 0.00 to 0.05. The screen suggests it is mostly a length proxy.

---

### OD10. Request-conditioned decision PMI against the policy-only baseline, `refusepmi` (tier 3)

- **Measures.** logP("I cannot / this is not allowed" | policy, request) minus logP(same | policy, no request), over a fixed set of 6 refusal and 6 compliance templates.
- **Targets.** Must-refuse cases where the agent acts despite a failed condition.
- **Inputs.** Policy, request. Local model.
- **Recipe.**
  - Prompt "[policy] Customer: [request]. Agent decision:" and score each template's first tokens; take the log-sum-exp over templates for each class.
  - Subtract the same quantity computed with the request replaced by a blank. This removes the model's prior for "refuse" under this policy, which is what failed in the first-action margin.
- **Level and cost.** L1. Two forward passes per case.
- **Varies by.** Case.
- **Universality.** Yes in principle. It is meaningful where the message pushes toward a forbidden action (tau2, banking), and doubtful on SOPBench, where the message does not reveal the record.
- **Status.** Variant of the failed first-action margin. The difference is the policy-only baseline (Holtzman PMI_DC) and class-level template pooling to avoid surface-form competition.
- **Evidence.** Holtzman 2104.08315: PMI_DC gives consistent gains in zero-shot multiple choice (verified, numbers unchecked). Arditi 2406.11717 shows that refusal is a distinct internal direction in chat models, which suggests the logit-level signal is real but concerns harm, not policy compliance.
- **Expected effect.** Case level about 0.00 to 0.05. High risk of failure on SOPBench.

---

### OD11. Silent-gate count (what the message leaves unsettled), `silentgates` (tier 4)

- **Measures.** For each gating condition of the procedure, how much the message moves the model's belief that the condition holds. A gate whose P(yes) does not change when the message is removed is "silent": the message does not settle it, so a correct agent has to read the record with a tool.
- **Targets.** Skipped checks. The number of silent gates is the number of lookups a good agent must do, which is where agents skip steps.
- **Inputs.** Policy gates, request. Local model for the logprob, or an API with a "yes/no" answer and sampling.
- **Recipe.**
  - Once per procedure, an LLM splits the policy into atomic conditions c_1..c_n (the compiled program already does this).
  - For each condition, compute P(yes | policy, message, "Does the message state that c_i holds? Yes or no") and P(yes | same, message deleted).
  - Features: number of conditions with |logit change| below a threshold (silent), number clearly settled, and the sum of binary entropies.
- **Level and cost.** L1. n conditions times 2 forward passes (cheap on a local model, sharing the prefix).
- **Varies by.** Case. Customers who volunteer information settle more gates.
- **Universality.** SOPBench yes (conditions come from the policy text). Tau2 and banking: partly, since conditions in long policies are many, so restrict to conditions that mention fields in the scenario. New process: yes.
- **Status.** Variant of "NLI between message and gating clauses" (about 0) and of "written-condition skip model". Differences: PMI against a message-deleted baseline, so it measures information added by the message and not entailment, and it counts silence as the feature and not entailment or contradiction.
- **Evidence.** Own reasoning, with Holtzman's PMI idea. Fact 3 of the brief (77% skipped checks) motivates it.
- **Expected effect.** Case level +0.02 to 0.08 on tau2, and 0.00 to 0.04 on SOPBench, where the record decides. Risk: the NLI variant already gave about 0.

---

### OD12. Leave-one-clause-out attribution, effective clause count, `clauseattr` (tier 4)

- **Measures.** Delete each policy clause in turn and measure how much a decision or plan log-probability moves. The entropy of the normalized attribution distribution, and the number of clauses with large effect, say how many policy rules this case actually leans on.
- **Targets.** Simultaneous load and interference (listed), but computed per case from model behavior and not from clause counts.
- **Inputs.** Policy, request, a reference plan or decision text. Local model.
- **Recipe.**
  - Split the policy into m clauses.
  - Score s_full = logP(reference plan | policy, request).
  - For each clause j, compute s_j with clause j removed and delta_j = s_full minus s_j.
  - Features: the count of clauses with delta above a threshold, the entropy of |delta| normalized, and the sum of positive deltas.
  - m+1 forward passes with a shared prefix cache, so a few seconds per case on MPS.
- **Level and cost.** L1 (plan needed).
- **Varies by.** Case. It differs from clause retrieval margin (listed) because it is about the model's actual use of clauses, with the plan as the output.
- **Universality.** Yes.
- **Status.** New relative to the listed items.
- **Evidence.** Own reasoning. Leave-one-out attribution is standard in interpretability, but I did not verify a source for use as a difficulty feature.
- **Expected effect.** Case level +0.02 to 0.06. Risk: on SOPBench the same procedure uses the same clauses for all cases, so variance is small.

---

### OD13. Weak-versus-strong plan divergence, `capgap` (tier 4)

- **Measures.** Distance between the plan of a weak local model and the plan of a strong API model for the same case.
- **Targets.** Fact 4: strong and weak agents find different things hard (case rho 0.41 between top 5 and bottom 5). If weak and strong planners agree, the case is probably robustly easy. If they diverge, there is a capability-sensitive step.
- **Inputs.** Policy, tools, request. Local model plus API model.
- **Recipe.**
  - Plan with Qwen3-8B (or a local instruct model) and with Sonnet, using the same PLAN prompt.
  - Compute the canonical-plan edit distance, the set difference of tools called, and the perform/refuse agreement.
  - Report the signed difference in the number of writes and checks (strong minus weak).
- **Level and cost.** L1. One cheap local call plus one API call (the API plan already exists for PLAN).
- **Varies by.** Both.
- **Universality.** Yes.
- **Status.** Variant of "cross-model disagreement on checks" (listed) and "two planners Gemini plus Sonnet (n.s.)". The difference is the large capability gap and the signed difference in effort, not two similar strong models. If the n.s. result for the Gemini-Sonnet pair holds, expect little.
- **Evidence.** Own reasoning, plus the brief's case rho 0.41.
- **Expected effect.** Case level 0.00 to 0.04. The previous two-planner test was n.s.

---

### OD14. Hidden-state difficulty probe trained on other domains, leave-one-domain-out, `hsprobe` (tier 5)

- **Measures.** Linear readout of case difficulty from a local model's hidden state at the last token of "[policy][tools][request]. Will an agent handle this correctly?".
- **Targets.** Anything the model represents about difficulty that its text outputs do not show.
- **Inputs.** Policy, tools, request. Local model, mid and late layers.
- **Recipe.**
  - Extract layer-wise last-token states from Qwen3-8B (a few layers, for example every 4th).
  - Fit ridge on IRT difficulty from the other domains and predict the held-out domain. This reproduces the protocol, so it has to be fit inside the protocol. Compare with bge embeddings, which are the tried baseline.
  - Add the SEP variant: instead of IRT difficulty, train the probe to predict semantic entropy of the model's answers, computed on generic QA, then apply it unchanged to our cases.
- **Level and cost.** L1. One forward pass per case, minutes in total, plus GPU memory.
- **Varies by.** Case.
- **Universality.** Yes, but fitting needs training domains.
- **Status.** Partly tried: embeddings (bge, fine-tuned encoder 0.255, mostly tau2). The new parts are decoder middle layers after a task-specific question and the SEP variant, which needs no fitting on our labels at all.
- **Evidence.** Lee 2510.05969: difficulty is linearly decodable from final-token states for math problems (numbers not seen). Kossen 2406.15927: SEPs from hidden states approximate SE at near-zero cost. Kadavath: a learned P(IK) transfers weakly (0.606 to 0.687 AUROC on new tasks when trained on TriviaQA only), so cross-domain failure is the main risk.
- **Expected effect.** Case level 0.00 to 0.08, with a high chance of just reproducing the encoder result.

---

### OD15. Entropy of the model's own answer to "what is the first thing you need to find out?" (information-need entropy), `needent` (tier 5)

- **Measures.** Distribution over what a model would ask or look up first. If the model's distribution over "the first fact to check" is spread over many different conditions, the case has many competing checks. If it is concentrated, there is one obvious gate.
- **Targets.** Verification chain depth and ordering (listed) in the sense of ambiguity in what to verify first, which is exactly where agents delay or skip a check.
- **Inputs.** Policy, tools, request. Sampling API (no logprobs) or local logprobs over the labelled list.
- **Recipe.**
  - Give the model the numbered list of atomic conditions (from compilation).
  - Ask "which condition do you verify first, and which second?". Locally, read the distribution over the index tokens; via API, sample 8 times and tally.
  - Features: entropy of the first-check distribution, entropy of the second-check distribution, and the mean pairwise disagreement of the full orderings across samples (Kendall tau).
- **Level and cost.** L1. One forward pass locally, or 8 API calls.
- **Varies by.** Both; the case-level part comes from which conditions the message already settles.
- **Universality.** Yes where conditions can be listed. Tau2 and banking policies are large, so restrict to the top 15 retrieved conditions.
- **Status.** New. Related to "guard distance and order" (listed), but it measures the model's ambiguity about order, not distance in the text.
- **Evidence.** Own reasoning. Q-DAPS (ACL 2026, 2605.12398) uses the entropy of plausibility over candidates as a difficulty estimate, so entropy over candidate items is the same family. I did not see its correlations.
- **Expected effect.** Case level 0.00 to 0.05. Speculative.

---

## Count by tier and by distance from the seeds

Seeds are tier 1 to 2 (OD1, OD2, OD3, OD4, OD5, OD6). Beyond the seeds: OD7 (partially), OD8, OD9, OD10, OD11, OD12, OD13, OD14 and OD15, which is more than half of the 15.

## Top 3

1. **OD11 silent-gate count.** It is the only idea here that targets the mechanism with the strongest evidence in the brief (skipped checks, 77%) and that necessarily varies by case: whether the message settles a gate. It is cheap on a local model, and it differs from the NLI attempt in that it uses a message-deleted baseline. The honest risk is on SOPBench, where the record decides. I would bet on tau2 and banking.
2. **OD8 plan-step grounding, with the omission mirror.** It exploits the one thing local logprobs measure well, which is how much of a text is explained by a context, and applies it to the LLM plan, which is already the best case-level signal (about 0.29 on tau2). The omission version checks whether the plan ignored a "must" clause, again a skipped-check detector.
3. **OD2 sampled-plan agreement.** It is the most defensible use of the sampling literature (SE and Mehta), needs no logprobs and works on Claude and Gemini 3.x. Its worth is the cheap extra variation on a plan the pipeline already produces. If it does not add to c_lad, the whole sampling branch of this family can be closed.

## What I would not spend money on

Plain token entropy of reasoning (already tried), first-action margin (tried), verbalized confidence (probably redundant with ADeLe, and Xiong found it overconfident), thinking length (two papers say it correlates loosely with real difficulty), and any learned "do I know" probe, because Kadavath's transfer numbers (0.606 to 0.687 AUROC when moving task) match our observed failure to transfer to a new domain.

## Optional quick screen (exploratory screen, not the protocol)

This is the pure-code version of OD9 (gzip conditional compressibility of the request against up to 40 same-procedure siblings). I computed Spearman against the full-data IRT difficulty (`b`) per domain, with a length-only baseline. Script: `prototype/screen/ideas8/outdist/ncd_screen.py`, run under `cpu_lock`.

| Benchmark | Domain | n | rho (compression) | rho (length) |
|---|---|---|---|---|
| SOPBench | bank | 134 | -0.247 | -0.250 |
| SOPBench | dmv | 97 | -0.140 | -0.056 |
| SOPBench | healthcare | 124 | -0.093 | -0.001 |
| SOPBench | hotel | 195 | -0.044 | -0.097 |
| SOPBench | library | 66 | -0.111 | +0.072 |
| SOPBench | online_market | 172 | -0.009 | +0.133 |
| SOPBench | university | 42 | +0.071 | +0.155 |
| tau2 | airline | 50 | +0.397 | +0.402 |
| tau2 | retail | 114 | +0.398 | +0.443 |
| tau2 | telecom | 114 | -0.060 | -0.057 |
| tau-Knowledge banking | banking_knowledge | 97 | +0.544 | +0.532 |

- SOPBench mean: -0.082 for compression (-0.097 after centering within procedure), versus -0.006 for length.
- tau2 mean: +0.245 for compression, +0.263 for length.
- The within-procedure centered version is undefined on tau2 airline and retail and on banking, because each case there is its own procedure (constant input).
- Reading: on tau2 and banking the compression feature is a length proxy. On SOPBench it is weakly negative, mostly bank and dmv, and inconsistent across domains. Atypicality alone does not beat length, which lowers my expectation for OD1 and OD9. The LM version may still differ, but I would not prioritize it.
