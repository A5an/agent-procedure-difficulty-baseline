# Swarm agent `request`, round 8 (6 Oct 2026)

Saved by the main session from the agent's final message.

# Request-side signals: idea catalog from agent `request` (round 8, 6 Oct 2026)

I could not write files, so the full catalog is below. All scripts are in `research/screen/ideas8/request/` (`screen.py`, `chk.py`, `uk.py`, `rebut.py`). I made no LLM calls, no GPU or MPS use and no agent runs.

## What the data showed first

- **SOPBench messages are templated.** Each message gives a username, a credential string and a few parameter values, then ends "Thank you!".
- **Every SOPBench case already supplies all arguments of the target tool.** I checked the raw `user_known` field against the target-tool arguments, and no case is missing any. So argument-completeness ideas cannot help SOPBench through missing arguments. They can help through the credentials and the gate inputs the request does or does not supply.
- **Within a SOPBench procedure, difficulty b depends strongly on the hidden label.** The mean b residual is +1.42 for must-perform and -0.75 for must-refuse. Procedure identity explains 26% of the variance in b.
- **tau2 and banking scenarios** are long scripts with "Reason for call", "Known info" and "Instructions", and the scripts contain conditional rebuttals.

Level L0 means documentation only. L1 means a few cheap LLM calls per case. L2 means reading the hidden case record. E means emulation. "Whitebox" means a local open model.

## Catalog (tier 1 = most obvious, tier 5 = least obvious)

### R1. Argument ledger: given / derivable / must-fetch / missing (tier 1)
- **Measures:** for each argument of the target tool, whether the request gives it verbatim, whether it can be derived from the request, whether a lookup tool must fetch it, or whether it is missing. This targets argument hallucination, premature action and skipped lookups (fact 3).
- **Inputs:** tool schemas plus request. Model: black-box API.
- **Recipe:**
  - One call per case. Prompt: "Given the target tool schema and this request, label each required argument as GIVEN, DERIVABLE (say from what), FETCH (say which tool), or MISSING."
  - Features: counts of each label and the fraction not GIVEN.
  - Run a second pass on the policy's gating checks: for each check, is its input GIVEN or FETCH?
- **Level and cost:** L1, 1 to 2 calls per case.
- **Varies by case:** yes. On tau2 and banking the user reveals different facts per case. On SOPBench it varies only through the gate inputs.
- **Universality:** SOPBench partial (the target tool's arguments are never missing, but gate inputs vary). tau2 yes. Banking yes. New process yes.
- **Status:** a concrete recipe for the "argument provenance" item on the 6 Oct council list, applied to the request side.
- **Evidence:** diagnostic in R6. UserBench (arXiv 2507.22034, checked): agents give fully aligned answers only 20% of the time when user goals are underspecified. LLMs Get Lost in Multi-Turn Conversation (arXiv 2505.06120, Laban et al. 2025, checked): -39% average when instructions are underspecified across turns.
- **Expected effect:** small on SOPBench (cap near 0.2). Moderate on tau2 and banking. Risk: LLM labels are noisy.

### R2. Credential and identity presence flags (tier 1)
- **Measures:** whether the request carries authentication material (password, identification, ID) and how random-looking it is (max character entropy of the longest token). A missing credential forces the agent to ask or refuse. This targets the authentication step.
- **Inputs:** request only. Model: none.
- **Recipe:**
  - Regex for password, identification and ID words.
  - Shannon entropy of tokens of 8 or more characters, taking the maximum.
  - Both features are already in `screen.py`.
- **Level and cost:** L0, free.
- **Varies by case:** yes, including within a procedure (it varies within SOPBench procedures, see the screen).
- **Universality:** SOPBench yes. tau2, banking and new process only if the scenario states credentials, so the regex needs a per-process word list.
- **Status:** new.
- **Evidence:** exploratory screen below.
- **Expected effect:** about -0.2 within SOPBench. Risk: it may only proxy for the credential type of a procedure or domain. Of the 17 SOPBench features tried, this was the best, so multiplicity is a real concern.

### R3. Request clarity and typicality against the procedure (tier 2)
- **Measures:** how unusual this request is compared with the other requests of the same procedure. Two versions: leave-one-out char n-gram TF-IDF cosine, and KL divergence between the request's unigram model and the procedure's request collection. This is an analogue of the query clarity score. It targets distribution shift, which makes agents skip defaults.
- **Inputs:** request, plus the pool of requests of the procedure. Model: none.
- **Recipe:** as above. On a new process, use an LLM to write 20 typical requests as the pool.
- **Level and cost:** L0 on a benchmark, L1 on a new process.
- **Varies by case:** yes.
- **Universality:** SOPBench yes. tau2 and banking: one task per procedure, so the pool must be synthetic. New process yes with synthetic pool.
- **Status:** new. Not the whole-text embedding that was tried.
- **Evidence:** Cronen-Townsend, Zhou and Croft, SIGIR 2002 (checked): clarity score is the relative entropy between query and collection language models, and it correlates with average precision. The number I saw, an optimal threshold of 1.09, comes from a search snippet and is unverified.
- **Expected effect:** SOPBench screen: typicality rho +0.05 per domain and +0.07 within-procedure. Weak.

### R4. Claim inventory: assertions the agent must verify (tier 2)
- **Measures:** the count of unverifiable or checkable assertions in the request (identity, status, prior promises such as "the representative approved it"), split into those a tool can verify and those it cannot. This targets the 78% of must-refuse failures where the agent acts despite a failed condition.
- **Inputs:** request, plus tools to judge verifiability. Model: black-box API.
- **Recipe:** one call per case. Extract the claims, and for each label VERIFIABLE_BY_TOOL, UNVERIFIABLE, or CONTRADICTS_POLICY. Features: the three counts.
- **Level and cost:** L1, 1 call per case.
- **Varies by case:** yes.
- **Universality:** SOPBench: little (messages carry no claims). tau2 yes. Banking yes. New process yes.
- **Status:** a variant of SCN (which had pressure, forbidden request and conditional-instruction counts). It differs by counting claims and whether a tool can check them.
- **Evidence:** own reasoning. tau2 airline scenarios contain "the representative approved it" and "I am a silver member".
- **Expected effect:** tau2 case level, small to moderate. Risk: overlap with the existing pressure flag.

### R5. Anticipated-rebuttal count (tier 2)
- **Measures:** how many scripted reactions to an agent refusal the scenario contains ("If the agent says it is not possible, mention that..."). The scenario author wrote them because the first ask is blocked by policy. So this leaks the must-refuse state through the user script. It targets the label problem on the tau2 side, which is the tau2 analogue of reading perform or refuse from the message.
- **Inputs:** request only. Model: none (regex).
- **Recipe:** regex `If (and only if )?(the )?(service )?agent ... (says|tells|refuses|cannot|not possible|...)`, count matches. The code is in `rebut.py`.
- **Level and cost:** L0, free.
- **Varies by case:** yes.
- **Universality:** tau2 yes. Banking weak. SOPBench no, because messages have no scripts. New process only if scenarios are scripted.
- **Status:** new. SCN's `n_conditional` is a general count of conditionals. This one targets only agent-refusal triggers.
- **Evidence:** exploratory screen below.
- **Expected effect:** small, fragile.

### R6. Gate-input provenance (hidden-fetch count) (tier 2)
- **Measures:** how many gating-check inputs are not given in the request and must be fetched, relative to the checks. This targets skipped checks (fact 3). It is the gate-side form of R1.
- **Inputs:** policy, tools and request. Model: black-box API.
- **Recipe:**
  - Extract the checks from the policy once per procedure.
  - Then ask once per case: "For each check, does the request give its input values? GIVEN or MUST_FETCH."
  - Features: count of MUST_FETCH and its fraction.
- **Level and cost:** L1, 1 call per case. The diagnostic below used the hidden record, so it is L2.
- **Varies by case:** yes.
- **Universality:** SOPBench yes (the rule tree gives the checks). tau2 and banking yes with LLM extraction. New process yes.
- **Status:** a concrete recipe for "argument provenance" and "whether the request mentions the fields that gate the decision". New as a computed feature.
- **Evidence:** the L2 diagnostic below, from the rule tree and `user_known`.
- **Expected effect:** small, around +0.17 within-procedure on SOPBench, with a larger real-world gap likely on tau2.

### R7. Sentence-level distractor mass (tier 3)
- **Measures:** the share of request sentences or clauses unrelated to any policy variable or tool argument. This targets distraction and wasted reasoning.
- **Inputs:** policy, tools and request. Model: embedding model (a local encoder) or none.
- **Recipe:**
  - Split the request into sentences.
  - For each, take the maximum cosine to policy clause embeddings and tool-description embeddings.
  - Feature: fraction of sentences below a threshold set on the training folds, plus the mean of the maximum cosine.
- **Level and cost:** L0 (embeddings of short texts).
- **Varies by case:** yes.
- **Universality:** SOPBench: low variance because messages are templated. tau2 and banking yes. New process yes.
- **Status:** new at the sentence level. Whole-text embeddings were tried.
- **Evidence:** Shi et al., ICML 2023, GSM-IC (arXiv 2302.00093, checked): accuracy drops sharply when irrelevant context is added. The size of the drop is not checked.
- **Expected effect:** tau2 only, small.

### R8. Request surprisal and policy-conditional PMI under a local model (tier 3)
- **Measures:** how surprising the request is, and how much the policy lowers its surprisal: logp(request | policy) minus logp(request). Surprising values (odd names or credentials) may mark an engineered edge case.
- **Inputs:** policy plus request. Model: local open base model (whitebox).
- **Recipe:**
  - Score the request tokens with and without the policy as a prefix.
  - Features: mean surprisal, max surprisal, and the PMI difference.
  - Compute the surprisal on entity-value tokens separately from template tokens.
- **Level and cost:** L1 whitebox (it reads the case), one local forward pass per case. This round forbids running it.
- **Varies by case:** yes.
- **Universality:** yes on all four, since it needs only text.
- **Status:** new. PRI was policy-rule surprisal at the procedure level. This reads the request. Token entropy of a think-aloud trace (ENT) is different.
- **Evidence:** own reasoning. R2's character entropy is its pure-code cousin.
- **Expected effect:** small to moderate. Risk: it measures format artifacts.

### R9. Ambiguity as extraction disagreement across paraphrases (tier 3)
- **Measures:** how much a cheap model's reading of the request (action, arguments) changes under k paraphrases or k samples. This targets the "several readings" and underspecification cases.
- **Inputs:** request, tools. Model: black-box API.
- **Recipe:**
  - k=5 samples at temperature 0.7, each asked for the JSON tool call.
  - Feature: one minus the fraction of samples that agree on the call. Variant: semantic-entropy clusters.
- **Level and cost:** L1, 5 calls per case.
- **Varies by case:** yes.
- **Universality:** yes on all four.
- **Status:** a variant of "compile disagreement as ambiguity" (failed), which looked at policy compilation. This one looks at request interpretation.
- **Evidence:** Kuhn, Gal and Farquhar, ICLR 2023 (arXiv 2302.09664, checked): semantic entropy predicts model accuracy. Sclar et al., ICLR 2024 (arXiv 2310.11324, checked): up to 76 accuracy points of variation from prompt format alone.
- **Expected effect:** probably small. Risk: templated SOPBench messages leave no ambiguity to find.

### R10. Elicitation burden: facts the user reveals only when asked (tier 3)
- **Measures:** the number of scenario facts the simulated user hides until asked ("ONLY MENTION THIS if asked", "only provide details when the agent asks"). This is the number of questions the agent has to ask. It targets missing proactive questions.
- **Inputs:** request (scenario). Model: none (regex) or black-box API.
- **Recipe:** regex count of hiding phrases, or an LLM list of hidden facts and which of them a correct resolution needs.
- **Level and cost:** L0 (regex) or L1.
- **Varies by case:** yes.
- **Universality:** tau2 and banking yes. SOPBench no. New process yes if scenarios are scripted.
- **Status:** a variant of SCN `withholds_info` (a binary flag). This is a count and a need-weighted version.
- **Evidence:** `rebut.py` regex screen (not a headline). tau2 airline +0.23, retail -0.02, telecom -0.44. Length-adjusted tau2 mean is -0.16, and banking is +0.03. Inconsistent across domains. See the screen section.
- **Expected effect:** unclear. Risk: sign flips between domains.

### R11. Goal-policy feasibility: acceptable outcomes vs permitted outcomes (tier 4)
- **Measures:** the user's acceptance set against the set the policy permits. If the two are disjoint the agent must refuse or offer an alternative, which is the situation behind most failures. It targets theory-of-mind demand ("don't cancel if no refund").
- **Inputs:** policy plus request. Model: black-box API.
- **Recipe:**
  - Ask for the user's acceptable outcomes A and the policy-permitted outcomes P for this case.
  - Features: empty intersection (1 or 0), |P|, and |A| (how narrow the user's acceptance is).
- **Level and cost:** L1, 1 to 2 calls per case.
- **Varies by case:** yes.
- **Universality:** tau2 and banking yes. SOPBench no (single goal, no acceptance set). New process yes.
- **Status:** new. SCN `policy_forbidden_request` is related but does not reason over outcome sets.
- **Evidence:** own reasoning.
- **Expected effect:** possibly a stronger case-level label proxy on tau2. Risk: the LLM judging the intersection must itself apply the policy.

### R12. Persuasion-technique profile (tier 4)
- **Measures:** which social-influence techniques the scenario uses (authority, reciprocity, social proof, foot-in-the-door, and so on), and how many. This targets sycophancy and compliance with pressure.
- **Inputs:** request. Model: black-box API.
- **Recipe:** one call classifying against a technique list. Features: technique count and the presence of strong ones.
- **Level and cost:** L1, 1 call per case.
- **Varies by case:** yes.
- **Universality:** tau2 and banking yes. SOPBench no. New process yes.
- **Status:** a variant of SCN `pressure` (0 to 2). This adds technique types.
- **Evidence:** Zeng et al. (arXiv 2401.06373, checked): persuasive prompts reach over 92% attack success on Llama 2-7b-Chat, GPT-3.5 and GPT-4 in 10 trials. That is jailbreak success, not policy-following difficulty. Technique count not verified.
- **Expected effect:** small. Mostly redundant with SCN pressure.

### R13. User-simulator coordination load (tier 4, not universal)
- **Measures:** the number of user-side tool actions the agent must guide the user through, and the information the user can see only through those actions. This targets coordination errors.
- **Inputs:** scenario, plus the user-tool list.
- **Recipe:** count user-tool mentions in the scenario and policy. The LLM version asks which user actions the policy requires.
- **Level and cost:** L0 or L1.
- **Varies by case:** yes, but within telecom only.
- **Universality:** tau2 telecom only. Not universal.
- **Status:** new as a feature.
- **Evidence:** τ²-bench (arXiv 2506.07982, Barres et al., checked): performance drops when agents move from no-user to dual-control.
- **Expected effect:** telecom has only 3 procedures, so do not trust any gain.

### R14. Sequencing markers and intent order (tier 1, weak)
- **Measures:** counts of "first, then, also, finally, else" and "if" markers in the script, as a stand-in for the number and order of intents. This targets state tracking.
- **Inputs:** request. Model: none.
- **Recipe:** regex, as in `screen.py` (`n_then`, `n_if`).
- **Level and cost:** L0, free.
- **Varies by case:** yes.
- **Universality:** tau2 and banking yes. SOPBench no (one intent).
- **Status:** a variant of the SCN counts of goals and conditional instructions.
- **Evidence:** exploratory screen. `n_then`: airline +0.42, retail +0.22, telecom -0.33. `n_if`: airline +0.26, retail +0.30, telecom -0.69. Banking +0.40 for `n_if`.
- **Expected effect:** the sign flips in telecom, so only airline, retail and banking behave. It is probably mostly a length effect, and length is already known to give about +0.27 on tau2.

## Exploratory screen, not the protocol

Pure-code features on SOPBench (830 cases, 7 domains), tau2 (278) and banking (97), correlated with full-data 1PL IRT b. This is a diagnostic run that rediscovered known quantities, not a clean test, and I tried many features per benchmark. It was within the CPU limit and used `cpu_lock("request")`.

**R2, SOPBench:**

| Feature | Domain mean | Within-procedure | Per domain (bank, dmv, healthcare, hotel, library, online_market, university) |
|---|---|---|---|
| `has_pw` (credential words present) | -0.056 | -0.248 | -0.36, +0.05, -0.04, -0.05, +0.13, n/a, n/a |
| `maxent` (credential token entropy) | -0.219 | -0.231 | -0.55, -0.34, -0.16, -0.04, -0.21, -0.02, n/a |

- Pooled over all domains the correlation with b is about -0.4, but that mixes procedure effects with case effects.
- Among must-refuse cases only, `maxent` has pooled rho -0.40 with b. Among must-perform cases only it is -0.35. These are pooled over procedures, so only the within-procedure figure above is a case-level result.
- `maxent` correlates only -0.14 with the label itself. So it is not simply a copy of the perform or refuse state.
- The domain mean is weak because several domains have only a handful of varying cases. Caveat: only the bank domain is clearly negative on `has_pw`.

**R5, tau2 and banking (regex):**

| | Airline | Retail | Telecom | Banking |
|---|---|---|---|---|
| `rebut` rho with b | +0.255 | +0.239 | none (always 0) | +0.088 |
| `rebut` after removing log length | +0.13 | +0.11 | n/a | -0.12 |

- 13% of tau2 scenarios match the pattern.
- The effect shrinks after length adjustment and does not hold on banking. Verdict: fragile.

**R6, L2 diagnostic (SOPBench, rule tree plus `user_known`):** this reads the hidden record, so it is separate from the pure-code screen and is not a legitimate score.
- `n_cfetch` (gate arguments not given in the request) has a domain mean of -0.007 and a within-procedure rho of +0.173 (240 cases). More fetched inputs means harder.
- `n_known` has a within-procedure rho of -0.135. More supplied information means easier.
- `n_args` and `frac_missing` never vary within a procedure (frac_missing is 0 in every case, so it carries no signal).

**Other context from the same run:**
- Length-like features reach about +0.27 on tau2, matching what is already known. On banking, `n_num` reaches +0.54 and `w_all` +0.53, but that is a single domain of 97 cases.
- `numz` (numeric deviation from the procedure mean) gives a SOPBench domain mean of +0.13 and a within-procedure rho of +0.03. Boundary values are not detectable this way.

## Top 3

1. **R6 and R1 (provenance of gate inputs and tool arguments, with an L1 LLM ledger).** This is the clearest case-level mechanism. The L2 diagnostic gives +0.17 within a procedure on SOPBench, which has the weakest case-level signal of all benchmarks. On tau2 and banking the user reveals different facts per case, so the effect should be larger there. It is universal, because every process has gating inputs and either the request supplies them or the agent must fetch them. It costs 1 to 2 calls per case. The risk is that the LLM labels are noisy and that the gain depends on how the checks are extracted.

2. **R5 and R11 (tau2 label proxies: anticipated rebuttals and outcome-set feasibility).** Fact 2 says tau2 has no cap, and gold action counts alone reach 0.40 to 0.47, so any request-side proxy for how many writes the scenario needs is valuable. The scenario script itself encodes the author's knowledge of what the policy blocks. R5 is free and gives +0.25 on airline and retail, but it shrinks after length adjustment and fails on banking. R11 reasons over acceptable versus permitted outcomes and is the more principled version, but it costs LLM calls and risks circular policy application.

3. **R2 (credential and identity presence and randomness).** It is free, and it is the only case-level SOPBench feature that survives within-procedure at about -0.23. It is not a label proxy (rho -0.14 with the label). It is probably capturing whether the agent must ask for authentication, or how realistic the supplied identity is. The honest caveat is the multiplicity of features and the small number of domains where it varies. It is worth carrying into the real protocol as one cheap column rather than as a standalone model.

All checked references are listed with their ids above. Unverified: the 1.09 clarity threshold and the number of persuasion techniques in the taxonomy.
