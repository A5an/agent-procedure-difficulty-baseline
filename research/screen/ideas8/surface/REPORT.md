# Swarm agent `surface`, round 8 (6 Oct 2026)

Saved by the main session from the agent's final message.

# Round 8, agent `surface`: what the plain text of the policy and the request says

I mapped 15 text-surface ideas and ran two pure-code quick screens. The screens turned up one real signal and one null.

- **Real signal:** the amount of digit and identifier text in the customer message correlates with SOPBench case difficulty, even within one procedure and one perform/refuse label.
- **Null:** the "request given policy" compression signal is about 0 on SOPBench, and on tau2 it only repeats text length.

I could not write the folder's report files, so the full catalog is in this message. The screen script and its outputs are in `prototype/screen/ideas8/surface/` (`screen.py`, `features.csv`, `screen_results.csv`).

## Checked citations (arXiv abstract or search result opened)

I did not read full texts, so every number below comes from an abstract or search snippet.

- Singh and Strouse, "Tokenization counts", arXiv 2402.14903. Right-to-left number tokenization raised GPT-3.5 from 75.6% to 97.8% and GPT-4 from 84.4% to 98.9%. The fetched abstract page confirmed the finding; the numbers come from the search snippet.
- Bhatia, Peyrard, Zhao, "Date Fragments", arXiv 2505.16088. Excessive date fragmentation correlates with accuracy drops of up to 10 points on uncommon dates.
- Razeghi et al., "Impact of Pretraining Term Frequencies on Few-Shot Reasoning", arXiv 2202.07206. The page says models are above 70% (absolute) more accurate on frequent terms than on rare ones, in numerical reasoning tasks.
- Levy, Jacoby, Goldberg, "Same Task, More Tokens", arXiv 2402.14848 (ACL 2024). Reasoning degrades at lengths well below the technical maximum, and perplexity does not correlate with performance.
- Shi et al., "Large Language Models Can Be Easily Distracted by Irrelevant Context", arXiv 2302.00093 (ICML 2023). Irrelevant sentences sharply reduce accuracy. The abstract gives no figure.
- Modarressi et al., "NoLiMa", arXiv 2502.05167 (ICML 2025). With minimal lexical overlap between question and needle, 10 of 12 models fall below 50% of their short-context baseline at 32K. GPT-4o goes from 99.3% to 69.7%.
- McCoy, Pavlick, Linzen, HANS, arXiv 1902.01007 (ACL 2019). NLI models adopt a lexical overlap heuristic and do very poorly on HANS.
- Cui et al., "Generalized Quantifiers as a Source of Error in Multilingual NLU Benchmarks", arXiv 2204.10615 (NAACL 2022). Quantifier occurrence at test time is associated with performance drops.
- Kim and Schuster, "Entity Tracking in Language Models", arXiv 2305.02363 (ACL 2023). Only GPT-3.5-class models trained on much code track entity state. I saw no number for operation count, so I quote none.
- CenterBench, "The Dog the Cat Chased Stumped the Model", arXiv 2510.20543. Performance drops with embedding depth, and the plausible-versus-implausible gap reaches a median above 25 points. The figure comes from the search snippet.
- Han et al., "Read Before You Think", arXiv 2504.09402. Backward dependencies are a core bottleneck for decoder-only models and persist under chain-of-thought.
- Gonen et al., "Demystifying Prompts via Perplexity Estimation", arXiv 2212.04037 (Findings of EMNLP 2023). Lower prompt perplexity goes with better performance. No correlation number was visible.
- Sclar et al., arXiv 2310.11324 (ICLR 2024). Prompt format alone changes accuracy by up to 76 points on LLaMA-2-13B.
- Yang et al., "What Prompts Don't Say", arXiv 2505.13360 (Findings of ACL 2026). LLMs infer unspecified requirements 41.1% of the time by default. Underspecified prompts are 2x as likely to regress, sometimes by more than 20 points.
- Moell and Boye, arXiv 2502.11578. Models' accuracy computing the LIX readability metric correlates -0.875 with their MMLU, with N=6 models. This is about models, not cases, so I treat it as weak evidence only.

## Quick screen results (exploratory screen, not the protocol)

- **Computation:** pure code. For the screen only, I used the Qwen3 tokenizer file (a vocabulary lookup) to count tokens. No model inference.
- **Data:**
  - SOPBench: 830 cases in 7 domains. Request = customer message, policy = the action policy block.
  - tau2: 278 cases in 3 domains. Request = scenario, policy = the domain main policy file.
  - Banking skipped (one domain).
- **Target:** Spearman rho with full-data IRT `b` within each domain, averaged over domains. Positive rho means harder.

**Idea S01, digit and identifier burden of the request**

| Setting | mean rho | per-domain |
|---|---|---|
| SOPBench, digit-containing tokens | +0.188 | +0.22 +0.21 -0.05 -0.04 +0.37 +0.24 +0.37 |
| SOPBench, count of numerals | +0.185 | not listed |
| SOPBench, count of ID-like words | +0.159 | not listed |
| tau2, digit tokens | +0.240 | +0.25 +0.03 +0.44 |

- **Within-label check (SOPBench):** rho stays +0.198 inside the perform/refuse label strata.
- **Within-procedure check (SOPBench):** after subtracting procedure means, rho is +0.119. After subtracting procedure-and-label means it is +0.113.
- **Not a label proxy:** the correlation of digit tokens with the label is 0.002.
- **Telecom caveat:** tau2 telecom has only 3 procedures, and its signs flip on several features, so do not read the tau2 average.
- **Mechanism:** this probably reflects more values to copy and check, matching Singh and Strouse and Bhatia et al. That is my reading of the result, not something the screen tested.
- **Size:** about +0.11 to +0.12 for a single count feature. Roughly 0.1 is useful as an extra column but not decisive.

**Idea S02, incremental compression of the request given the policy**

Feature: `(lzma(policy+request) - lzma(policy)) / lzma(request)`.

| Setting | SOPBench mean rho | tau2 mean rho |
|---|---|---|
| lzma | -0.088 | +0.304 (+0.40 +0.38 +0.12) |
| zlib | -0.048 | +0.286 |
| absolute lzma | -0.042 | +0.254 |
| token count, for reference | +0.045 | +0.254 |

- **Verdict:** a null on SOPBench. On tau2 it adds nothing over plain length, since token count already gives 0.254. This variant fails.
- **Lexical overlap:** the content words in the request that are not in the policy (`novl`) show +0.297 on tau2 but -0.004 on SOPBench. That is also just length. Overlap fraction gives +0.108 on SOPBench (+0.102 within procedure and label) and +0.100 on tau2.

## Catalog, tier 1 (obvious) to tier 5

**S01. Numeric and identifier token burden of the request** (tier 1)
- **Measures:** how many digit tokens, numerals and ID-like strings the case contains. Targets skipped or wrong value checks: datetime reliability is 42%, relation reliability 46%.
- **Inputs:** request, plus a tokenizer file (none for a regex version).
- **Recipe:**
  - Tokenize the request with any BPE tokenizer.
  - Compute the digit-token count, the digit-token fraction, the numeral count (regex `\d[\d,.]*`) and the ID-like word count (`\b\w*[_\d]\w*\b`).
  - Optionally add the number of distinct numerals.
- **Level and cost:** L0, a millisecond per case.
- **Varies:** by case.
- **Universality:** SOPBench yes. tau2 yes (scenario ids and amounts). Banking yes, but untested. New process yes.
- **Status:** new. The tokenizer-fertility seed is covered here. Fertility itself was weak, with SOPBench rho -0.057.
- **Evidence:** 2402.14903, 2505.16088, and my screen above.
- **Expected effect:** case level, about +0.1 on SOPBench. The risk is that it duplicates the ADeLe quantitative and numeric levels.

**S02. Request-given-policy incremental compression** (tier 1)
- **Measures:** how much new text the request adds beyond what the policy already contains.
- **Inputs:** policy and request, no model.
- **Recipe:** as above. Variants are the compressed length of the request in the context of the policy (`xz` or `zstd` with a dictionary) and the normalized compression distance (Cilibrasi and Vitanyi, unverified).
- **Level and cost:** L0, trivial.
- **Varies:** by case.
- **Universality:** yes for all four, since it needs only the two texts.
- **Status:** new. It failed in my screen (see above). I list it for the record so nobody reruns it.
- **Evidence:** own reasoning plus the screen.
- **Expected effect:** about 0.

**S03. Lexical coverage and overlap between request and policy** (tier 1)
- **Measures:** the fraction of request content words found in the policy, and the count of out-of-policy words. Targets clause retrieval: the agent has to link the request to the gating clause.
- **Inputs:** policy and request, no model.
- **Recipe:**
  - Tokenize both texts.
  - Drop stop words.
  - Compute the overlap fraction, the out-of-policy word count and IDF-weighted overlap.
  - A stricter version restricts the policy side to gating-clause words.
- **Level and cost:** L0.
- **Varies:** by case.
- **Universality:** yes for all four.
- **Status:** a variant of the already-tried clause retrieval margin. The difference is that it uses no embeddings and no ranking, only a lexical overlap ratio.
- **Evidence:** NoLiMa 2502.05167 (low overlap hurts retrieval), HANS 1902.01007 (overlap as a shortcut), and my screen (SOPBench +0.108, tau2 +0.100).
- **Expected effect:** small, about +0.05 to +0.1 at case level. It overlaps with length on tau2.

**S04. Date, time and duration arithmetic load in the request** (tier 2)
- **Measures:** whether the case forces date arithmetic. Targets datetime checks, which are the weakest step type at 42%.
- **Inputs:** request and policy.
- **Recipe:**
  - Regex for absolute dates in several formats, weekdays, relative expressions ("next Tuesday", "last month", "in 3 days"), durations, and clock times.
  - Count them.
  - Mark the load as high when the policy states a time window and the request contains a relative date.
  - Add the date fragmentation ratio: tokens per date string.
- **Level and cost:** L0.
- **Varies:** by case.
- **Universality:** SOPBench yes. tau2 airline and retail yes (flights, return windows). Banking partly. New process yes.
- **Status:** new as a case-level, regex-based feature. The listed "numeric boundary margin" is about the margin to a threshold, not the date expression type.
- **Evidence:** 2505.16088 (up to 10 points), Test of Time 2406.09170 (title seen in search only, unverified).
- **Expected effect:** small case-level gain on date-heavy procedures. It may be near zero on SOPBench domains with few dates.

**S05. Approximators and hedged numbers in the request** (tier 2)
- **Measures:** words such as "about", "around", "roughly", "just over", and a policy that has an exact threshold.
- **Targets:** boundary-handling failures, where the agent mishandles a fuzzy value.
- **Inputs:** request and policy.
- **Recipe:**
  - Regex the approximator words next to a numeral.
  - Count the numerals that also appear as a threshold in the policy.
  - Add a flag for a value that is round or rounded.
- **Level and cost:** L0.
- **Varies:** by case.
- **Universality:** yes if the request has numbers.
- **Status:** new, though close to the listed numeric boundary margin. The margin uses the actual distance, while this one uses the epistemic marking of the value.
- **Evidence:** own reasoning only. I found no verified LLM study of approximator words.
- **Expected effect:** low frequency, so a small effect. It mainly helps tau2 scenarios with fuzzy claims.

**S06. Deontic and quantifier profile, selected by what the request touches** (tier 2)
- **Measures:** counts of must, may, must not, ALL, ANY, at least and only in the clauses the request is about. Targets polarity and exception errors.
- **Inputs:** policy and request.
- **Recipe:**
  - Count the modal and quantifier lexicon per clause.
  - Weight each clause by its S03 overlap with the request.
  - The procedure-level version is the unweighted sum.
- **Level and cost:** L0.
- **Varies:** the procedure-level sum varies by procedure only. The weighted sum varies by case but is noisy.
- **Universality:** SOPBench yes (the policy text uses "ALL of these conditions must be met" and "must have"). tau2 yes. Banking yes. New process yes.
- **Status:** a variant of the listed polarity and exceptions. The difference is the plain lexicon count and the request-weighted selection.
- **Evidence:** 2204.10615 (quantifiers are associated with drops, with no single figure verified).
- **Expected effect:** procedure level about 0.1 to 0.2. Case level near 0 on SOPBench, where the same policy serves all cases.

**S07. Entity and coreference load of the request** (tier 3)
- **Measures:** how many distinct people, accounts and objects appear, how many pronouns, and whether a third party is involved ("my wife's account"). Targets authentication and identity-mixing errors.
- **Inputs:** request.
- **Recipe:**
  - Count capitalized names and ID strings.
  - Count pronouns.
  - Count possessive references to other people.
  - Count distinct entity types.
- **Level and cost:** L0 with regex, or a small NER model.
- **Varies:** by case.
- **Universality:** SOPBench partly (one customer). tau2 yes (scenarios often include other people and several reservations). Banking yes. New process yes.
- **Status:** new. The listed scenario features count requests and pressure, not entities.
- **Evidence:** 2305.02363 (entity tracking depends on code pretraining), otherwise own reasoning.
- **Expected effect:** small and mainly on tau2.

**S08. Clause nesting and dependency distance of the request and of the gating clauses** (tier 3)
- **Measures:** the distance between a condition's head and the item it governs, subordinator count, and center embedding. Targets simultaneous load, here the parsing load of reading the condition.
- **Inputs:** request and policy.
- **Recipe:**
  - A pure-code proxy: count subordinators ("if", "unless", "provided that"), the maximum number of words between "if" and its matching main clause, and the maximum parenthesis or comma depth.
  - A parser-based version computes the average dependency distance with a dependency parser (spaCy and stanza were not installed in the repo's Python, so this needs an install).
- **Level and cost:** L0.
- **Varies:** by case for the request, by procedure for the policy.
- **Universality:** yes for all four.
- **Status:** new (Gibson locality and center embedding were seeds).
- **Evidence:** 2510.20543 (degrades with embedding depth), 2504.09402 (backward dependencies).
- **Expected effect:** the policy side is procedure level, about 0.1. The request side is short, so the range is limited.

**S09. Backward information order within the request** (tier 3)
- **Measures:** whether facts that change the answer come after the ask ("Please refund my order. By the way it was delivered 40 days ago"). Targets skipped checks caused by late conditions.
- **Inputs:** request.
- **Recipe:**
  - Find the first request verb and the position of each numeral or fact clause.
  - Compute the share of fact clauses that follow the first ask.
  - Compute the character offset of the last numeral divided by the request length.
- **Level and cost:** L0, regex only.
- **Varies:** by case.
- **Universality:** SOPBench yes (the customer message lists parameters after the request). tau2 partly. New process yes.
- **Status:** new. The listed position of gating checks is about the policy, not the request.
- **Evidence:** 2504.09402 for decoder-only models.
- **Expected effect:** small, with a real risk of being zero because the models are strong readers.

**S10. Propositional idea density of the policy and its gating clauses** (tier 4)
- **Measures:** propositions per word, in the Kintsch / CPIDR sense. Targets condition packing.
- **Inputs:** policy.
- **Recipe:**
  - Approximate the density with the verb, adjective, adverb, preposition and conjunction counts per 10 words, using a part-of-speech tagger.
  - Add the same density restricted to sentences that contain a must or if.
  - A parser-free proxy is function-word to content-word ratio.
- **Level and cost:** L0.
- **Varies:** by procedure.
- **Universality:** yes for all four, since each has a policy text.
- **Status:** a variant of the listed condition packing density. This one uses the established psycholinguistic measure.
- **Evidence:** Levy et al. 2402.14848 (length hurts, perplexity does not track it), otherwise own reasoning.
- **Expected effect:** about 0.1 or less at the procedure level, and overlapping with length.

**S11. Semantic-minus-lexical gap between request and policy** (tier 4)
- **Measures:** how much the request refers to the gating facts by different words than the policy uses. Targets failure to notice that a condition applies.
- **Inputs:** policy and request. Optionally a small sentence encoder (bge-small, already cached locally).
- **Recipe:**
  - For each gating clause take the cosine similarity and the lexical overlap with the request.
  - The feature is the maximum semantic similarity minus the lexical overlap of the best clause.
  - A high value means a paraphrase the agent must infer.
- **Level and cost:** L0 with a small encoder, about 1 s per case.
- **Varies:** by case.
- **Universality:** yes for all four.
- **Status:** a variant of the listed clause retrieval margin. The difference is the explicit semantic-minus-lexical gap, taken from NoLiMa.
- **Evidence:** 2502.05167.
- **Expected effect:** uncertain. SOPBench requests restate the policy only loosely, so the gap may be mostly noise.

**S12. Numeral rarity and roundness** (tier 4)
- **Measures:** how unusual the specific values are ($4537.82 against $4500, a long account number against a short one). Targets arithmetic or comparison errors on unfamiliar numbers.
- **Inputs:** request, and optionally a tokenizer.
- **Recipe:**
  - Compute the digit count of each numeral and the number of significant digits.
  - Compute the share of trailing zeros.
  - Compute the digit-string entropy.
  - Use the tokens-per-numeral count from the tokenizer.
- **Level and cost:** L0.
- **Varies:** by case.
- **Universality:** yes.
- **Status:** new, a refinement of S01. If S01 holds, this tests whether the effect comes from count or from rarity.
- **Evidence:** 2202.07206 (above 70% accuracy gap on frequent versus rare terms), but that is arithmetic few-shot, not agent tasks.
- **Expected effect:** same order as S01, and mostly redundant with it.

**S13. Surface-perturbation instability of the case** (tier 5)
- **Measures:** how much a cheap model's output changes when the request is rewritten without changing its meaning. Targets sensitivity to format and wording.
- **Inputs:** request, with a cheap LLM or a local model.
- **Recipe:**
  - Make 4 to 6 meaning-preserving rewrites with code only: reorder the facts, change number formats ($4,500 to 4500 dollars), change names, and swap synonyms from a small dictionary.
  - Ask a cheap model the single question "which clause decides this case?" or "will this request be approved?" for the original and each rewrite.
  - The feature is the disagreement rate or the logprob variance across rewrites.
- **Level and cost:** L1, 5 to 6 short calls per case.
- **Varies:** by case.
- **Universality:** yes for all four.
- **Status:** new. It is different from the failed compile-disagreement, which varied the compiler and not the request.
- **Evidence:** Sclar et al. 2310.11324 (up to 76 points on format changes, in few-shot classification).
- **Expected effect:** unknown. The risk is that the cheap model's disagreement measures its own weakness and not the agent's.

**S14. Request-given-policy pointwise mutual information under a local base model** (tier 5)
- **Measures:** log p(request | policy) minus log p(request). A low value means the request is off-script relative to the policy. Targets cases that need clause lookup because nothing in the request points to the right rule.
- **Inputs:** policy and request, plus a local base model (Qwen3-8B-Base is cached).
- **Recipe:**
  - Compute the summed token logprob of the request with the policy as prefix.
  - Compute it again with a neutral prefix.
  - Subtract and divide by token count.
- **Level and cost:** L0 with white-box access, since it reads the case. About one forward pass per case in two conditions.
- **Varies:** by case.
- **Universality:** yes for all four.
- **Status:** new. It is different from the policy-rule surprisal (PRI), which is policy-only and procedure-level, and from the token-entropy trace.
- **Evidence:** Gonen et al. 2212.04037 for prompt perplexity, but Levy et al. 2402.14848 report that perplexity does not track long-input reasoning. The two disagree, so this is a real risk.
- **Expected effect:** unknown. S02 (compression, a cruder version of the same idea) failed on SOPBench.

**S15. Specification gap of the request against the tool schemas** (tier 5)
- **Measures:** how many required tool parameters the request does not supply, so the agent has to ask or assume. Targets unspecified-value errors.
- **Inputs:** request and tool schemas.
- **Recipe:**
  - List the required parameters of the tools that the policy names.
  - Fill each by regex or typed matching from the request (numbers, ids, dates, names).
  - The feature is the fraction of required parameters left unfilled.
  - A variant counts only user-supplied ones.
- **Level and cost:** L0 by regex.
- **Varies:** by case. Within a procedure it varies with how many parameters the customer mentions.
- **Universality:** SOPBench yes. tau2 yes. Banking yes. New process yes if schemas exist.
- **Status:** a variant of the listed argument provenance. The difference is that it counts only what the request fails to specify.
- **Evidence:** 2505.13360 (default inference 41.1%, regressions of more than 20 points).
- **Expected effect:** a plausible case-level signal on tau2 only, where the scenario hides information. On SOPBench the message often lists the same parameters for every case.

## Top 3

1. **S01, numeric and ID burden.** It is the only idea with a measured result. It holds inside a procedure and inside a label (about +0.11), it is uncorrelated with the perform/refuse label, it costs nothing, and published tokenization work explains why it should hurt. It is worth testing as an extra column on top of c_lad under the full protocol.
2. **S15, specification gap, with S04 and S12 as sub-features.** All three are code-only case-level counts. They share the S01 mechanism and make it more specific, so they could sharpen it rather than duplicate it. This is the cheapest next test.
3. **S13, surface-perturbation instability.** It is the least obvious idea with a real mechanism. Prompt wording alone moves accuracy by up to 76 points, and no earlier idea varies the request surface. It is L1 and needs a few cheap calls, so it should wait for a protocol run if S01 confirms that surface form matters at all.

## Caveats

- Tokenizer fertility on its own is weak (SOPBench -0.057), so the signal comes from counting digits, not from token inefficiency.
- Every case-level number above is a single-feature correlation, not a pooled protocol result. None of it is validated under folds.
- The SOPBench domains I screened are not the same as the held-out domain scenario, so do not read the average as a new-domain result.
- All tau2 numbers are dominated by length and by the three-procedure telecom domain.
