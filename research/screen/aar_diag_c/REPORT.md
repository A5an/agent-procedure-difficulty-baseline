# E4-C: the diagnostic test on fixed SOPBench labels, Haiku and Sonnet predictors. Result (12 Nov 2026)

Plan `PREREG.md`, code `diag_c.py` (copy of `orig_diag.py` from the A5an branch with label source and model as
arguments), predictions `preds_<model>_<labels>.jsonl`, outputs `score_<labels>_<model>.out`. Claude subscription
(`claude -p`), $0. The published-label Haiku run is Asanali's own predictions, rescored here: identical numbers.
Under fixed labels, V0 and V1 are the same predictions rescored (their prompts hold no labels), V2 and V3 were called
again with fixed history counts. Haiku fixed: 5 of 168 calls failed three times (bad JSON), so 224 of 233 test cases
are scored for that run.

AUC (mean over the four agents), [95% bootstrap over procedures]:

| input of the predictor | Haiku, published | Haiku, fixed | Sonnet, published | Sonnet, fixed |
|---|---|---|---|---|
| V0 case text only | 0.509 [0.44, 0.58] | 0.504 [0.43, 0.57] | 0.513 [0.46, 0.57] | 0.508 [0.46, 0.56] |
| V1 + domain and procedure names | 0.504 | 0.492 | 0.576 [0.52, 0.64] | 0.574 [0.52, 0.63] |
| V2 + stats of similar procedures | 0.590 | 0.557 | 0.612 | 0.590 |
| V3 + stats of this procedure | 0.629 | 0.595 | 0.656 | 0.663 |
| DomainRate, no LLM | 0.590 | 0.564 | 0.590 | 0.567 |
| ProcRate, no LLM | 0.661 | 0.666 | 0.661 | 0.664 |

Brier skill vs BaseRate: Sonnet V3 +0.031 (published) and +0.035 (fixed), ProcRate +0.052 / +0.053, V3 minus ProcRate
-0.022 [-0.064, +0.016] and -0.018 [-0.053, +0.014]. Haiku V3 -0.052 (published) and -0.219 (fixed); Haiku's V2 under
fixed labels collapsed to Brier skill -0.336 (mean forecast 0.46 against a success rate of 0.68), while its V2 on
published labels was -0.027: the same model gives very different calibration from one run of counts to another.

## Reading

- With the grader fixed, a zero-shot reader of the case text is still at chance (V0 about 0.50 for both predictors).
  The fix does not rescue case-level prediction from text, which agrees with our within-procedure result
  (`sopaudit/runtime/REPORT.md`).
- This is a statement about zero-shot LLM reading. Supervised features of the same documentation (c_lad, f2) rank
  procedures at rho 0.68-0.72 on fixed labels (`sopaudit/clad_fixed/REPORT.md`), so "the text does not predict" should
  not be written without "for a zero-shot reader, per case".
- The procedure name helps a stronger reader: Sonnet V1 beats V0 (Brier skill +0.061 [+0.020, +0.110] published,
  +0.077 [+0.028, +0.137] fixed). With Haiku it did not. So "category adds nothing" is predictor-dependent.
- With the procedure's own counts Sonnet reaches the counting baseline (V3 about ProcRate), Haiku stays below it.
  Counting without an LLM remains the best or tied-best option in all four runs.
