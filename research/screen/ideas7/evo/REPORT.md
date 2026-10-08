# Evolutionary feature search, FunSearch/AlphaEvolve style (agent `evo`, round 7, 6 Oct 2026)

Saved by the main session from the agent's final message. Files: build_records.py, records*.jsonl, evaluator.py,
sandbox_worker.py, FREEZE.json (hashes frozen at 21:34 before the first proposal), FINALISTS.md (before any held-out look),
finalists.json, archive.jsonl (504 candidates), evolve.log, null_calibration.json, heldout_results.json, score.txt, out_protocol/.

Setup: per-case records from existing documentation-only files (no outcomes, gold, constraints, labels, databases; the one
outcome-derived input is the condition skip-model score fitted on other domains). Search split: SOPBench bank, dmv,
healthcare, hotel + tau2 airline, retail (714 cases). Held-out: SOPBench library, online_market, university + tau2 telecom
(394 cases). Candidates: Python features(record) -> up to 8 floats, sandboxed. Score: gain in within-domain Spearman when
added to c_lad's out-of-fold prediction (ridge, procedure-grouped CV). Null: 40 noise candidates, max +0.007.
Proposer gemini-3.8-flash, 3 islands, 12 rounds x 14, 504 evaluated, 510 calls.
Search best +0.09 per island (biased, max over about 500). Finalists by rule: F1 (id 485, msgpol, +0.093, fresh +0.096),
F2 (id 459, plantools, +0.086, fresh +0.081).

Held-out (single look; c_lad alone 0.129): F1 A +0.098 mean of 4 domains (SOPBench only +0.014; telecom +0.350 drives it),
B transfer +0.052 (SOPBench +0.069; online_market negative in both). F2 about flat (A +0.010, B -0.021).

Full protocol (biased: about 70% of cases in search domains; mild skip-model leakage in new_domain):
| column | pooled | vs c_lad |
|---|---|---|
| evo_f1 | 0.353 | +0.095 [+0.034, +0.123] |
| evo_f1_fam | 0.372 | +0.114 [+0.037, +0.162] |
| evo_f2 | 0.288 | +0.030 [-0.000, +0.052] |
F1 by scenario: sop-proc .294 (+.057), sop-dom .270 (+.058), tau2-proc .430 (+.132), tau2-dom .419 (+.134).
F1 features: disagreement between the dry-run refuse probability and the plans or program; conditions settled by the
request minus those needing system data; plan ambiguity spread; message vagueness; id/number density in the message;
highest weighted skip probability among conditions; tools in conditions vs plans mismatch; a benchmark-specific friction term.
Verdict: promising but not confirmed; needs a clean test (fresh benchmark or the blind set).
