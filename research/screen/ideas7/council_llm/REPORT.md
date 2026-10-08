# Council member `llmnative` (signals native to LLMs), 6 Oct 2026

Saved by the main session. No code, sources from memory (unverified).
Filter: features must vary by case (use the message) or by check inside the case's policy, procedure-level ones are capped.
F1 clause retrieval margin and distractor density (embedding similarity of each gating clause to message + action, rank
and margin to the best distractor clause; 0 LLM calls); F2 position of gating checks in the prompt (lost in the middle,
procedure level); F3 wrong-tool confusion (similarity of the target tool to others, entropy of a tools-only choice);
F4 argument provenance and format gap (argument given verbatim, needs reformatting, or must be fetched; ID copy chains);
F5 cross-model disagreement on extracted required checks (checks only one model lists = implicit checks); F6 NLI between
message sentences and each gating clause (user asserts a condition is met, contradiction, silence; local DeBERTa-MNLI,
no API); F7 premature-action propensity from a small open model (logP of writing first vs looking up first, KL when a
gating sentence is removed; MPS, needs approval); F8 prior surprisal of each gating rule under a base model (procedure level).
Top 3: F6 (+0.01 to +0.03 guess, free), F7 (+0.02 to +0.05, needs a local model), F1 (+0.01 to +0.03, minutes).
Develop on synthetic cases only, evaluate on real, report without telecom.
