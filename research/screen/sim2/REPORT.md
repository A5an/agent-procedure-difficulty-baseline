# E2: second user simulator. Partial result (12 Nov 2026)

Plan `PREREG.md`. Stopped at about a third by the 70 USD total for E1 and E2 (ledger in `../seedk`): gemini-2.5-pro as
the simulated customer ran 123 of 300 seeded probes with agent gemini-2.5-flash and 79 of 300 with gemini-3.8-flash,
0 errors. The pre-registered outcomes need all 50 families, so only a description is given (`../seedk/analyze_seedk.out`).

| agent | probes run with both simulators | outcome differs | passed with flash simulator | passed with pro simulator |
|---|---|---|---|---|
| gemini-3.8-flash | 79 | 14 (18%) | 61 | 61 |
| gemini-2.5-flash | 123 | 21 (17%) | 90 | 101 |

On the families complete under both simulators, Spearman with the real failure rate: gemini-3.8-flash (12 families)
0.62 with the flash simulator, 0.60 with pro; gemini-2.5-flash (20 families) 0.83 with flash, 0.40 with pro.

Reading, with care because the sample is partial and not random over families: about one probe in six changes its
outcome when only the model behind the customer changes, which is the size of effect the pre-registration called
support (above 10%), but without an interval over all 50 families this is not a test.
