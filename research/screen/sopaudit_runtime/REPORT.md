# Which request fails: documentation vs data, and a commit-time guard (11 Nov 2026)

No model calls. Scripts: data_vs_doc.py (SOP venv; writes case_branch.json = per case: procedure, policy constraints
text, gold perform/refuse, conditions false on the initial DB via fixsop.truth), data_vs_doc2.py / data_vs_doc3.py
(python3), runtime_check.py (guard). Targets: our four agents on SOPBench-fixed (806 cases). History: the 28 published
agents re-scored with the fixed grader. Pre-check: LLM-compiled policy programs from ideas2/compile (A = policy only,
AC = with the domain's example DB in the prompt, post hoc and optimistic; both compiled from the ORIGINAL policy text).

## Within one procedure, what tells failing requests apart (AUC over pairs inside the same procedure, chance 0.5)
Score = group fail rate of the 28 published agents minus their procedure rate, 5-fold cross-fitted x 20 (the
self-exclusion artifact cancels).

| | 3.8-flash (19 fails) | 2.5-flash (115) | Sonnet 5.5 (93) | Haiku 5.5 (94) |
|---|---|---|---|---|
| policy text of the case | 0.56 | 0.52 | 0.48 | 0.54 |
| data branch (which conditions are false) | 0.72 | 0.49 | 0.63 | 0.60 |
| same request, 28 public agents' outcome on it | 0.84 | 0.69 | 0.62 | 0.70 |
| fail % perform / refuse requests | 4.5 / 1.3 | 24.0 / 9.5 | 18.0 / 8.3 | 18.7 / 8.2 |
| failures with the wrong perform/refuse decision | 10 | 71 | 18 | 24 |

70 procedures, 236 distinct policy texts, 331 data branches. Reading: the documentation carries no signal inside a
procedure; the request's data carries some (except 2.5-flash), the specific request more. Perform requests fail 2-3.5x
more often (full chain of checks plus the write, vs stopping at the first false condition). Most Sonnet/Haiku failures
are process failures (right decision, a required check not called before the action).

## Commit-time guard: flag when the agent's decision differs from the pre-check program, or the action comes before a
check the program needs; flagged requests go to a human

| | flagged | failures caught | success after hand-over (alone) |
|---|---|---|---|
| 3.8-flash, AC / A | 12.4% / 19.4% | 58% / 53% | 0.990 / 0.989 (0.976) |
| 2.5-flash, AC / A | 21.7% / 24.7% | 92% / 79% | 0.989 / 0.970 (0.857) |
| Sonnet 5.5, AC / A | 16.6% / 23.0% | 59% / 54% | 0.953 / 0.947 (0.885) |
| Haiku 5.5, AC / A | 16.9% / 23.1% | 53% / 49% | 0.945 / 0.940 (0.883) |

Pre-check decision accuracy: A 0.845, AC 0.962. Decision-only guard (AC): flags 4-12%, false alarms about 3% of
successes, catches 16-63% of agent-fault errors. Second agent as guard (flag on different perform/refuse decision):
2.5-flash checked by 3.8-flash flags 8.3%, catches 57%, 0.857 to 0.938; for the stronger agents it catches 11-37%.

Limits: one run per agent, no intervals; 3.8-flash has 19 failures. Failures are the SOPBench grader's, which also
fails right decisions taken without the demanded check. AC programs saw the example DB after the failures were known.
