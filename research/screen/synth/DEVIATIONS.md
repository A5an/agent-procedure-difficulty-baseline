# Deviations from PREREG.md (logged as they happen)

1. 3 Nov, after the main generation started, before any judge call on main specs: the policy judge (`judge.py`) and the
   label audit (`audit.py`) use gemini-3.8-flash with reasoning_effort "low" instead of gemini-2.5-flash. Reason: on the
   pilot specs gemini-2.5-flash flagged consistent statements as conflicts (for example "if you cannot find the account,
   make no change", which is what the program grades as correct). The judge prompt now explains the program's semantics
   and returns conflicts with an explicit same-result flag. Sanity check before use (`judge_sanity.py`,
   `judge_sanity.out`): 6 of 6 pilot specs pass, 3 of 3 deliberately broken policies are flagged (two changed
   thresholds, one deleted rule).
2. 3 Nov, during the first lite run: the proxy's re-ask of empty agent replies (in PREREG.md) re-sent the same request
   at temperature 0 and got the same empty reply (10 of the first 89 lite episodes crashed). The re-asks now sample at
   temperature 0.5 and then 1.0, for that one turn only. Crashed episodes are re-run with this proxy (FREEZE.md was
   written before; the proxy hash there is the earlier version).
   Still crashing after that (11 episodes): up to 4 re-asks, at temperature 0.5 then 1.0, with the request seed changed.
3. 3 Nov, analysis code after FREEZE.md: `analyze.py` got (a) the default L2 penalty (C = 1) in the factor models,
   because with almost no penalty some levels were nearly separable in bootstrap resamples and the fits diverged;
   (b) a bootstrap helper that returns no interval when fewer than 20 resamples are valid; (c) a descriptive
   cross-generator similarity of policy texts and titles on the same brief. Not pre-registered additions:
   `audit_strong.py` (every failure of flash and g38 audited with the same judge) and the hard round (PREREG_V2.md).
4. Hard round, exploratory addition: `audit_strong.py` run on every failure of flash and g38 in the hard round
   (`v2_audit_strong.jsonl`), same judge. Spend after it: 19.39 USD of the 20.
