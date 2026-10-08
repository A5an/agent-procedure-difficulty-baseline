# Swarm agent `internals`, round 8 (6 Oct 2026)

Saved by the main session from the agent's final message.

White-box catalog for the round 8 brief (agent `internals`). I wrote no files, ran no inference and ran no screen. All ideas are idea-level and nothing is tested.

## Ground rules shared by all ideas

**Transfer from a small local model to big API agents.** The 8B model does not need to behave like the API agents. It needs to act as a sensor, and a ridge or logistic head trained on training-fold IRT difficulty turns that sensor into a predictor. Two things support this:
- Item difficulty is shared across models. The brief reports case-level rho 0.41 between the top 5 and bottom 5 agents, which is modest.
- Papers report that hidden states encode human-labelled difficulty (arXiv 2510.18147, below). The same paper says model-derived difficulty is weaker and scales poorly with size. So a small model's probe may track "human-visible difficulty" better than "what this agent finds hard".

**Leakage-free training.** Every head is fit under leave-one-procedure-out, or leave-one-domain-out for the four new-process scenarios.
- Fit the PCA, the standardisation and the ridge only on training-fold items.
- Fit the probe target on the full-data IRT difficulty of training-fold items only.
- Centre hidden states by procedure (subtract the procedure's mean state, estimated from the policy plus that procedure's cases without labels). This forces the probe to learn within-procedure variation. That is the only part that can beat the SOPBench case-level cap near 0.2. Without it, the probe will mostly re-learn procedure identity, which is already solved at 0.5 to 0.7.
- Always report a prompt-length baseline. The code-correctness paper does the same (2606.14530).

**MLX cost assumptions (my own, to be calibrated on the machine).**
- Qwen3-8B 8-bit prefill is about 500 tok/s, and decode is about 20 tok/s.
- A policy is about 3k tokens and is cached as a prefix once per procedure. A case plus request is about 400 tokens.
- About 2,000 cases across the four scenarios.
- One cached forward pass is about 1 s per case, so about 35 minutes for all cases. Multiples below are multiples of this.

**What changes with other models.**
- Qwen3-instruct: refusal and verification behaviour is shaped by post-training, so act/refuse directions become meaningful (ideas 5 and 12).
- Llama and Gemma-2: Gemma Scope SAEs exist only for Gemma 2 2B and 9B, plus selected layers of 27B (2408.05147). I could not verify any equivalent SAE suite for Qwen3, so that is "unverified".
- Tuned-lens translators are not shipped for Qwen3 as far as I verified. I did not find any, so train one (idea 4) or use the plain logit lens.

## Verified citations (abstract read on arxiv.org on 6 Oct 2026)

- **2606.14530** (12 Jun 2026): Qwen3-4B-Instruct-2507, 444 LiveCodeBench tasks. A linear probe on the hidden state at the last prompt token, before any generation, predicts the correctness of the first-attempt code. Held-out AUC is 0.881 ± 0.008 over 50 outer splits. After residualising prompt length it is 0.842 ± 0.010, against 0.657 ± 0.014 for a prompt-length-only baseline. Nonlinear models do not help.
- **2510.18147**: probes across layers and token positions on 60 models. Human-labelled difficulty is linearly decodable, with AMC rho about 0.88. LLM-derived difficulty is much weaker. Steering along the difficulty direction changes accuracy and hallucination.
- **2406.15927** (Kossen et al.): semantic entropy probes approximate semantic entropy from the hidden states of a single generation. They generalise better out of distribution than probes that predict accuracy directly. The abstract gives no AUROC. "Unverified" for any specific number.
- **2407.03282**: internal states of an LLM indicate whether it has seen the query in training. A probing estimator reaches an average hallucination-risk estimation accuracy of 84.32% across 15 NLG tasks and 700+ datasets. Pre-generation.
- **2410.14516** (Heo et al.): there is an "instruction-following dimension" in the input embedding space. It generalises across unseen tasks but not across unseen instruction types. The authors say it relates more to prompt phrasing than to task difficulty. This is a warning for us.
- **2406.11717** (Arditi et al.): refusal is mediated by a single direction in 13 open chat models up to 72B. Erasing it prevents refusal, and adding it elicits refusal.
- **2406.08391**: a thousand graded examples are enough to train an uncertainty estimator on model features (LoRA). Prompting alone is insufficient.
- **2207.05221** (Kadavath et al.): P(IK) is the probability that the model knows the answer, predicted without reference to a specific proposed answer. No number quoted.
- **2410.02707**: truthfulness information is concentrated in specific tokens. Error detectors do not generalise across datasets, so encoding is "multifaceted". Another warning.
- **2310.16789**: Min-K% Prob gives a 7.4% improvement on WIKIMIA over previous detection methods.
- **2212.04037**: lower prompt perplexity goes with better prompt performance.
- **2305.14739**: context-aware decoding contrasts output probabilities with and without context. It reports a 14.3% factuality gain for LLaMA on summarisation.
- **2407.07071** (Lookback Lens): attention ratio on context versus generated tokens, per head, feeds a linear detector as good as a hidden-state detector.
- **2411.00348** (Attention Tracker): "important heads" shift attention from the original instruction to an injected one. The abstract claims AUROC gains up to 10.0%.
- **2303.08112**: tuned lens, with the trajectory of latent predictions able to detect malicious inputs.
- **2302.00294** (Valeriani et al.): the intrinsic-dimension profile across layers rises and then falls. The relative minimum marks the layers with the most semantic content. The numbers I did not read.
- **2212.03827** (CCS): an unsupervised direction found from consistency, with no labels. It beats zero-shot accuracy by 4% on average and halves prompt sensitivity.
- **2310.01405** (RepE): population-level representation reading and control.
- **2310.12962** (Mitchell et al.): emulated fine-tuning with a small model. Only the title and the start of the abstract were read, so the logit-difference "implicit reward" claim is "unverified".
- **2408.05147** (Gemma Scope): JumpReLU SAEs on all layers of Gemma 2 2B and 9B, selected layers of 27B, plus some for instruction-tuned 9B.

An earlier id I tried, 2302.11939, was a time-series paper and was dropped.

## Catalog

All ideas use the three inputs through a local open model. "Varies by case" means the feature is read after the case text, so it varies within a procedure. SOPBench is capped by the hidden perform/refuse state, so every case-level claim there is limited by fact 1 of the brief.

### Tier 1

**I1. Pre-generation difficulty probe on the last prompt token.**
- Measures: whether the model's own state at the "about to act" token encodes how hard this case is, which includes skipped checks. This is the direct analogue of 2606.14530, 2510.18147 and 2407.03282.
- Inputs: policy + tools + request, Qwen3-8B-Base local. Instruct would be better in principle, since 2606.14530 used an instruct model.
- Recipe:
  1. Prompt = policy, tool schemas, then the case, then "Assistant, first step:".
  2. Take the residual state at the last token from 6 layers (for example 12, 16, 20, 24, 28, 32). Also take a mean-pool over case tokens.
  3. Subtract the procedure mean, take PCA to 64 dimensions on training folds, then fit ridge on IRT difficulty.
  4. Report within-domain Spearman under leave-one-procedure-out.
- Level L1. Cost 1 s per case (one forward), about 35 min in total, with no sampling.
- Varies by case, yes.
- Universality: all four, yes. It needs only text. For tau2 and banking the case is the user scenario. SOPBench has only the message.
- Status: new here. PRI (rule surprisal) is a procedure-level logprob. This is a trained head on hidden states.
- Evidence: 2606.14530 (AUC 0.881 for correctness, not rho for difficulty), 2510.18147 (rho 0.88 on human labels).
- Expected: tau2 case level +0.03 to +0.10 over c_lad, if any. The risk is that it mostly encodes procedure or length. Headline numbers in those papers come from single-domain, in-distribution training, and our setting is a cross-procedure transfer, which is the setting where 2410.02707 reports failures to generalise. About 40% chance of nothing on SOPBench cases.

**I2. Case-token-only pooled state variance and norm profile.**
- Measures: how much the case moves the state away from the policy-only state, per layer. A trivial case barely moves it.
- Recipe: run the cached policy prefix, then the case. Compute ||h_case_mean − h_policy_last|| and cosine per layer. Use 8 layer scalars as features in a small ridge, so there are no high-dimensional heads.
- Level L0/L1, cost 1×. Varies by case. Universal: all four.
- Status: new. Evidence: own reasoning.
- Expected: weak (+0.01 to +0.03). The advantage is that it has 8 parameters, so it cannot overfit. The risk is that it measures case length.

### Tier 2

**I3. Attention mass and concentration on gating clauses.**
- Measures: whether, at the decision token, the model looks at the clauses that gate the action. Targets the dominant failure of skipped checks (fact 3).
- Inputs: policy sentence spans, request; local model with manual attention.
- Recipe:
  1. Split the policy into sentences. Tag gating clauses cheaply by regex (if, unless, only, must, before, verify, not allowed). That is pure code, so no LLM is needed.
  2. At the last token, in each of 8 chosen layers, read the attention probabilities per head, summed over each clause span.
  3. Features: mass on gating clauses, entropy of the distribution over clauses, max-clause mass, and the Gini coefficient across clauses, averaged over the top 20% heads.
  4. Pick heads as "important heads" by correlation with difficulty on training folds only, the same way Attention Tracker picks heads from a calibration set (2411.00348).
- MLX note: the fused scaled_dot_product_attention does not return the weights. Recompute softmax(QK^T) for the last query row only. That is cheap, because one query row costs O(n) per head.
- Level L1, cost about 1.3×. Varies by case, yes, since attention depends on the case text.
- Universality: all four. Gating tags use language cues, not benchmark fields.
- Status: variant of "clause retrieval margin" and "guard distance and order" from the councils. It differs by reading the model's attention instead of retrieval scores with an embedding.
- Evidence: 2411.00348 (instruction attention shifts under distraction), 2407.07071 (attention ratios as classifier features). Neither studies policy difficulty.
- Expected: case level +0.01 to +0.05. The risk is that attention does not equal causal use.

**I4. Instruction-following dimension probe (Heo-style).**
- Measures: whether the phrasing of the request puts it in a region where this local model follows instructions well. For tau2, customer phrasing carries pressure and conditional instructions.
- Recipe: build a probe on the embedding-layer (or early-layer) state of the request, trained on training-fold IRT difficulty. Heo et al. found that the dimension generalises across tasks but not across instruction types, so train only within the same benchmark family and check leave-one-domain-out.
- Level L0/L1, cost 0.3×, since it needs only the request. Varies by case.
- Universality: tau2 and banking yes. SOPBench is borderline (templated messages). New company process: yes if there are customer messages.
- Status: new. Overlaps with SCN, but SCN used hand-defined features.
- Evidence: 2410.14516. Its own finding that the dimension tracks phrasing, not task difficulty, is exactly the risk here.
- Expected: near 0 to +0.03.

### Tier 3

**I5. Act-versus-refuse direction projection (Arditi-style, instruct model).**
- Measures: how strongly the request pushes the model toward acting versus refusing.
- Recipe:
  1. Take Qwen3-instruct or Llama-instruct. Build a direction by difference-in-means between the hidden states of the compliant and refused set. Use training-fold cases labelled perform and refuse (SOPBench gives these labels for training folds only), at the last prompt token.
  2. Project each test case on it.
  3. Variant without labels: use the original harmful-vs-harmless direction from Arditi et al. as a generic stance direction. This does not mean "violates this procedure's condition", so expect it to be noisy.
- Level L1, cost 1×, plus one pass over training cases. Varies by case, but see the caveat.
- Universality: SOPBench yes (labels in training folds), tau2 and banking only if perform/refuse labels exist, new process no unless labels come from synthetic cases.
- Status: variant of the failed perform/refuse classifiers and the five failed label-from-message attempts (AUC 0.49 to 0.55). Why it might differ: it works in the model's state, not text. Why it likely will not: the label depends on the database record, which is not in the message. Honest expected value is about 0.
- Evidence: 2406.11717 (refusal direction exists, 13 models, up to 72B). No evidence for procedural refusals.
- Expected: about 0 on SOPBench. Possibly small on tau2 pressure and forbidden requests.

**I6. Policy pull: KL(with policy || without policy) on the first decision token.**
- Measures: how much the policy changes the model's expected first action for this case. A large pull means the policy matters for the case, a small pull means the model ignores it, which is where skipped checks likely come from.
- Recipe: two forward passes per case. Pass A is policy+tools+case. Pass B is tools+case with the policy removed. Compare the next-token distributions over the top 50 tokens at the first assistant step. Features: KL, logit-margin change, and the change in the probability of the verification-type tokens (identify these by tokenising the tool names of read-only tools).
- Level L1, cost 2×. Varies by case.
- Universality: all four, since it needs only text.
- Status: variant of "KL with and without gating clause" in the seeds and of the failed first-action logprob margin. Differs by contrasting policy presence, not reading one distribution.
- Evidence: 2305.14739 (CAD uses this contrast for decoding, not for prediction).
- Expected: procedure level small, case level +0.00 to +0.04. Risk that it measures only prompt length.

**I7. Clause-ablation influence profile.**
- Measures: for this case, how many individual clauses actually matter to the model's first-step distribution.
- Recipe: for each of up to K=12 gating clauses, mask it with the attention mask and not by deleting text (this keeps the prefix cache valid up to the clause). Compute the KL of the first-step distribution. Features: sum, max, number of clauses above a threshold, and the entropy of the influence vector.
- Level L1, cost K×. About 12 s per case, and about 6.5 h in total. Cut to 4 clauses per case for about 2 h.
- Varies by case. Universality: all four.
- Status: new in this form. It relates to "partial evaluation of the compiled program" and "simultaneous load" (the clause count, but measured by the model).
- Evidence: own reasoning.
- Expected: +0.02 to +0.06. Expensive for the gain.

**I8. Semantic-entropy probe trained on the local model's own samples (label-free w.r.t. benchmark).**
- Measures: the local model's own decision uncertainty, predicted from hidden states without sampling at test time.
- Recipe:
  1. For a training set of prompts, sample N=8 short decisions ("will you perform, refuse, or ask first, and which tool") at temperature 1. Compute the entropy of the decision clusters.
  2. Train the probe from the last-prompt-token state to that entropy.
  3. The target is the model's own behaviour, so no IRT labels are used in training. Training can run on synthetic cases from documentation (E-type), which makes it usable for a brand-new process.
  4. The probe output becomes a feature. The final combiner is fit on training-fold IRT.
- Level L1 (or E if synthetic cases are used). Cost: sampling 8×30 tokens is about 12 s per prompt, so 1,000 training prompts is about 3.5 h one-off. Test time is 1 forward.
- Varies by case.
- Universality: all four. This is the most defensible label-free recipe in the list.
- Status: variant of ENT (token-entropy profile of a think-aloud trace, which hurt) and of failed first-action margin. Differs because the target is a decision-level semantic entropy and the readout is a probe, not raw entropy.
- Evidence: 2406.15927 (SEP approximates semantic entropy from hidden states, with better out-of-distribution generalisation than accuracy probes). Number "unverified".
- Expected: +0.00 to +0.05. Entropy of a small model's decision might correlate with ambiguity, but ENT hurt, so this carries a real failure risk.

**I9. Min-K% tail surprisal of the case, and of the case given the policy.**
- Measures: how unusual the customer's message is for the model (outlier tokens), and how unusual it is conditional on the policy.
- Recipe: per-token logprob of the case tokens under the local base model, once without the policy and once with it. Features: Min-K% (K=20) mean, the same conditional on policy, and their difference (how much the policy "explains" the case).
- Level L1, cost 1.5× (the policy prefix cached). Varies by case.
- Universality: all four.
- Status: variant of PRI (policy-rule surprisal under Qwen3-8B-Base). Differs by reading the case and by the conditional difference.
- Evidence: 2310.16789 (7.4% over earlier detection methods on WIKIMIA, a different task). 2212.04037 supports prompt perplexity versus performance.
- Expected: case level +0.00 to +0.03. Surprise is not difficulty, and PRI worked only at procedure level.

### Tier 4

**I10. Logit-lens convergence depth at the decision token.**
- Measures: at what layer the decision becomes stable. Late convergence means the model deliberates more, which could track hard-to-decide cases.
- Recipe: apply the final norm and the unembedding to each layer's last-token state. For the candidate set (first tokens of read-only calls, write calls, refuse phrasing, ask-for-info phrasing), find the first layer after which the top-1 candidate class never changes ("commit layer"). Also record the margin at layers 20, 24, 28. A tuned lens (2303.08112) is better, but needs training per layer on Qwen3, about 1 h of MLX compute (unverified estimate).
- Level L1, cost 1×. Varies by case. Universality: all four.
- Status: new. Evidence: 2303.08112 for the lens. I found no verified work linking commit depth to task difficulty, so "own reasoning".
- Expected: +0.00 to +0.04. Logit lens on base models is brittle, as the tuned-lens paper states.

**I11. Representation velocity at condition-bearing tokens.**
- Measures: the case text is processed in one causal pass, so the hidden state at each token is free. The jump ||h_t − h_(t−1)|| at tokens that carry numbers, dates and negations shows where the model has to do work.
- Recipe: tag case tokens by regex (digits, month names, "not", "no", "never", "unless"). Take the mean velocity in layers 16 to 28 at tagged versus untagged tokens, and the ratio.
- Level L1 (reads the case), cost 1× (same pass as I1, so free once I1 runs). Varies by case.
- Universality: all four. Targets the datetime 42% and relation 46% reliability from fact 3.
- Status: new. Evidence: own reasoning.
- Expected: small, +0.00 to +0.03. High variance.

**I12. Intrinsic dimension and effective rank of the token cloud.**
- Measures: the geometric complexity of the representation of policy+case, per layer.
- Recipe: TwoNN intrinsic dimension of the token states in a prompt, per layer, then the layer profile (peak height, location of the first minimum). Effective rank = exp(entropy of the normalised singular values). Case-level variant: the same on the case tokens only. Valeriani et al. say the first ID minimum marks the layers with the most semantic content, so also use that minimum to pick the layer for probes in I1 (a label-free layer-selection rule).
- Level L0 (policy only) or L1 (with case). Cost 1.2×, plus the TwoNN in numpy.
- Mostly procedure level for the policy-only version. The case-token version varies by case but has few tokens (about 400), so the estimate is noisy.
- Universality: all four.
- Status: new. Evidence: 2302.00294 (ID profile structure, not difficulty prediction). Own reasoning for the rest.
- Expected: procedure level maybe 0.1 to 0.3 correlation. Case level near 0. The risk is that it is length in disguise.

### Tier 5

**I13. Quantisation-disagreement margin.**
- Measures: how fragile the model's decision is. Run the same prompt on the 8-bit, 4-bit and (if memory allows) bf16 models, or with layer-skipped early exit. Where the distribution flips under small perturbation of the weights, the case is borderline for the model.
- Recipe: KL and top-1 disagreement of the first-step distribution between precisions, plus the logit margin at each.
- Level L1, cost 2 to 3×, plus loading the extra model variants. Varies by case.
- Universality: all four.
- Status: new. Evidence: own reasoning, with the same idea in spirit as ensembling-based uncertainty (no verified citation).
- Expected: +0.00 to +0.04, a cheap cross-check of I1 and I6.

**I14. Instruct-minus-base logit gap (alignment shift).**
- Measures: how much post-training changes the decision distribution for this case, a proxy for where trained behaviours such as verification and refusal kick in over the pre-trained prior.
- Recipe: run Qwen3-8B-Base (already local) and Qwen3-8B-instruct on the same prompt, and take the KL and the signed logit differences on act, verify and refuse candidate tokens. Emulated fine-tuning (2310.12962) uses the instruct-minus-base logit difference as an implicit reward, but I verified only the title and the start of the abstract.
- Level L1, cost 2×, plus the instruct model download. Varies by case.
- Universality: all four.
- Status: new. Evidence: own reasoning, plus the partially read 2310.12962.
- Expected: +0.00 to +0.04. Risk: the post-training gap of an 8B model may not resemble that of the API agents.

**I15. Gradient attribution mass on policy versus case tokens.**
- Measures: the share of the first-step decision logit's input-gradient norm that goes to the policy gating clauses versus case tokens.
- Recipe: mx.grad of the margin (best act token minus best verify token) with respect to the input embeddings, through the quantised model. I could not confirm this runs on 8-bit weights in MLX ("unverified"), so test it first, or use the bf16 model on a subset. Features: share of gradient norm on gating clauses, on case numbers, and on the rest.
- Level L1, cost about 3× plus memory for backward. About 1.5 to 2 h per 2,000 cases if it works.
- Varies by case. Universality: all four.
- Status: new. Evidence: own reasoning.
- Expected: unknown. Interesting mainly as the causal check on I3 (does attention equal use).

**I16. Gemma Scope SAE feature loads on gating clauses (Gemma-2-9B).**
- Measures: counts of active SAE features tied to negation, conditionals, comparisons and dates at the gating-clause tokens and at the case tokens.
- Recipe: run Gemma-2-9B (8-bit in MLX) and apply a JumpReLU SAE encoder from Gemma Scope at one mid-layer (for example layer 20). I did not check which widths exist per layer, so "unverified". For each token, record the active feature set. Find features by supervised screening on training folds only (correlation of feature activation at gating tokens with difficulty). Run a stability selection over 5 splits and keep features that survive. Use the selected feature count and activation sums as features.
- Level L0 for the policy, L1 for the case. Cost 1.5× the Qwen forward. The SAE encode is about 16k×3.5k matmul per token, which is small.
- Procedure level for the policy side. The case side varies by case.
- Universality: all four.
- Status: new. Evidence: 2408.05147 verifies the SAE suite only. No difficulty result. Feature interpretation is not checked here.
- Expected: low, with a high risk of selection noise given hundreds of features and a few hundred items.

That is 16 ideas, above the 10 to 15 target. If this has to be cut to 15, drop I2 (it is covered by I1 and I11). Ideas beyond the seeds: I2, I6 (partly), I7, I8 (partly), I11, I12 (layer selection), I13, I14, I15, I16. That is 10 of 16, so more than half.

## Top 3

1. **I1 with procedure centring and the I12 layer-selection rule.** This is the best-evidenced direction: pre-generation hidden-state probes predict correctness (AUC 0.881, 0.842 after length control, in 2606.14530) and difficulty (rho about 0.88 on human labels, 2510.18147). Our setting differs in two ways: cross-procedure transfer, and an indirect IRT target. I would run it first because the compute is minutes and the same pass yields I2 and I11 for free. Procedure-mean centring is the key design choice. The warnings in 2410.02707 and 2410.14516 say probes often learn the wrong thing, so the prompt-length baseline and leave-one-procedure-out are mandatory.

2. **I8, the SEP trained on the local model's own sampled decisions.** It is the only idea in which the probe's training labels come from the model itself and not from the benchmark. That removes the IRT-leakage worry and the small-n overfitting, and it works for a brand-new process (train on synthetic cases from documentation). It also directly answers "does a small model's uncertainty transfer": the final combiner is the only part fit on IRT. ENT's failure is the reason to expect modest results.

3. **I6, policy pull (KL with versus without policy), paired with I3 attention on gating clauses.** Both aim at the main failure mechanism: skipped checks. I6 asks whether the policy changes the model's first move, and I3 asks whether the model looks at the gating clauses. Both vary by case and both are universal and cheap (2× and 1.3×). If either adds anything, the combined signal would be a mechanistic version of "the agent did not consult the rule".

## Quick screen

Not run. Every idea needs local model inference, which the brief forbids this round.

Files: only the brief at `research/screen/ideas8/BRIEF.md` was read. Nothing was written to `prototype/screen/ideas8/internals/`.
