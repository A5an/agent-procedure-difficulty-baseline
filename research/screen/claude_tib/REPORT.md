# E1-C: the seeds curve with Claude agents. Result (12 Nov 2026)

Plan `PREREG.md` (with the harness fix before the pilot), hashes `FREEZE_PRE.txt`, code `bridge.py`, `run_claude_tib.py`,
`full_run.sh`, analysis `analyze_claude.py` (hashed before the full run finished), output `analyze_claude.out`,
`results_claude.json`. Claude subscription, $0. 3,323 episodes: for each of Claude Haiku and Claude Sonnet the 300 real
tasks and the probes k0 (299), k1 (164, 28 families), k2 (300), k5 (299), foreign (300); 0 errors in the final files.
One runner (Sonnet k0) crashed at 227 of 299 when a simulator call stalled for 5 minutes; the bridge now retries such
calls and the remaining 72 episodes were run with it. Each episode is a fresh `claude -p` session with no other context
in its own directory; the customer is Claude Haiku with tau2's simulator prompt for every condition including the real
tasks, so these numbers are compared within Claude, not with the Gemini runs.

Real success: Haiku 0.697, Sonnet 0.693.

## Spearman between probe failure rate and real failure rate, 50 families (bootstrap over families, 2000)

| condition | Claude Haiku | Claude Sonnet | (Gemini: g38 / flash, `../seedk/REPORT.md`) |
|---|---|---|---|
| k=0, documentation only | 0.48 [0.19, 0.71] | 0.45 [0.19, 0.66] | 0.31 / 0.53 |
| k=2, two past conversations | 0.65 [0.40, 0.83] | 0.73 [0.53, 0.88] | 0.70 / 0.73 |
| k=5, five past conversations | 0.79 [0.62, 0.91] | 0.73 [0.56, 0.86] | 0.75 / 0.80 |
| foreign, another family's two | 0.30 [0.01, 0.56] | 0.53 [0.30, 0.72] | 0.40 / 0.47 |

| paired difference | Claude Haiku | Claude Sonnet |
|---|---|---|
| k5 - k0 | +0.31 [+0.04, +0.60] | +0.28 [+0.05, +0.54] |
| k5 - k2 | +0.14 [-0.04, +0.34] | -0.00 [-0.16, +0.16] |
| k2 - foreign | +0.35 [+0.05, +0.66] | +0.20 [-0.06, +0.49] |
| foreign - k0 | -0.17 [-0.41, +0.05] | +0.09 [-0.14, +0.29] |

On the 28 families with k1: Haiku k0 0.45, k1 0.58, k2 0.57, k5 0.70; Sonnet 0.51, 0.42, 0.79, 0.77 (k1 is noisy and
not monotone).

"All 6 probes passed" and real success: k5 Haiku 25 families, 137/150 = 0.91; Sonnet 22, 121/132 = 0.92. k2 Haiku 22,
116/132 = 0.88; Sonnet 25, 135/150 = 0.90. foreign 0.78 / 0.87. k0 0.86 / 0.86. Brier of the raw probe rate: Haiku
k0 0.202, k2 0.164, k5 0.149, foreign 0.227; Sonnet 0.225, 0.152, 0.156, 0.205.

## Reading, across the four agents (two Gemini, two Claude)

- H1 holds for all four: five same-procedure past conversations rank families better than documentation probes
  (k5 - k0 above 0 for every agent).
- Two conversations already give most of it; 2 -> 5 is not significant for any agent.
- Another procedure's conversations never beat documentation probes (foreign - k0 includes 0 for all four), and k2
  beats foreign with the interval above 0 for gemini-3.8-flash and Claude Haiku, touching 0 for gemini-2.5-flash and
  Claude Sonnet. The gain comes from seeing the procedure's own scenario; in templated families that is close to
  seeing the test, so the k2 and k5 numbers are an upper bound for real past tickets.
- With k5, "6 of 6" families reach 0.90-0.95 real success for all four agents.
