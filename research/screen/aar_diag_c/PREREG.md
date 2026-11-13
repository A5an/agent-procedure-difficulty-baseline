# E4-C: the diagnostic test on fixed SOPBench labels and with a second Claude predictor (pre-registration, 12 Nov 2026)

Handoff E4 asked for Gemini predictors; Gemini money is spent (70 USD), so per Tima (12 Nov, "юзай клод модели через
подписку") the second predictor is Claude Sonnet via `claude -p`. Design, prompts, split and metrics are those of
`orig_PREREG.md` / `orig_diag.py` (A5an branch research/2026-11-11-diagnostic-memory, aar_diag), reproduced exactly on
the published labels (AUC V0 0.509, V3 0.629, ProcRate 0.661) before any new call.

Changes (`diag_c.py`): (1) labels: published, or fixed = the same published episodes re-scored by the fixed SOPBench
grader (`prototype/data/sopbench_fixed`), with the 24 broken cases removed from history counts and test (233 test cases
left, same split); (2) predictor haiku or sonnet. V0 and V1 prompts do not contain labels, so under fixed labels their
predictions are the published-run predictions rescored; V2 and V3 contain history counts and are called again.

Question: with the grader fixed, does the text alone (V0) still rank cases at chance, and do statistics still carry all
the signal? Outcomes as in the original: AUC, Brier skill vs BaseRate, success in the top half, bootstrap over
procedures (1000), paired Brier-skill differences.
