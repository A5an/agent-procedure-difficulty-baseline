# E8 and E8-0: step routing pilot on SOPBench (13 Oct 2026)

Pre-registration: `PREREG.md`. Runner `run_e8.py`, analysis `analyze_e8.py` (output `results.json`, `analyze_e8.out`),
hand-off breakdown `handoff_breakdown.json`. Episodes in `episodes/`, all API costs in `ledger.jsonl` (26.6 USD in total,
including tests, duplicated and re-run episodes). 150 cases (75 perform, 75 refuse, 46 procedures), one attempt each,
temperature 0, max_tokens 4096, SOPBench evaluator unchanged.

## Results

| arm | success | reweighted to the benchmark mix | perform | refuse | acted where it had to refuse | refused where it had to act | cost of 150 cases, USD |
|---|---|---|---|---|---|---|---|
| gemini-2.5-flash | 0.700 | 0.718 | 0.640 | 0.760 | 0.187 | 0.067 | 1.17 |
| gemini-2.5-pro | 0.693 | 0.697 | 0.680 | 0.707 | 0.227 | 0.080 | 6.22 |
| pro until the target action, flash decides | 0.780 | 0.802 | 0.707 | 0.853 | 0.107 | 0.093 | 5.71 |
| flash until the target action, pro decides (negative control) | 0.807 | 0.821 | 0.760 | 0.853 | 0.093 | 0.067 | 3.59 |
| gemini-2.5-flash-lite (exploratory) | 0.113 | 0.140 | 0.027 | 0.200 | 0.707 | 0.040 | 0.20 |
| gemini-3.8-flash (exploratory) | 0.833 | 0.848 | 0.787 | 0.880 | 0.080 | 0.053 | 3.13 |
| flash-lite until the target action, 3.8 flash decides (exploratory) | 0.767 | 0.785 | 0.707 | 0.827 | 0.093 | 0.107 | 2.11 |

Primary contrast (fixed in the plan): pro>flash minus the better single model (flash): +8.0 points [+2.9, +12.8], paired
bootstrap over procedures; 17 cases only the mix solved, 5 only flash; exact McNemar p = 0.017.
Secondary: pro>flash minus pro +8.7 [+3.2, +14.1]; flash>pro minus flash +10.7 [+4.4, +16.0], minus pro +11.3 [+5.9, +16.5];
pro minus flash -0.7 [-6.2, +4.3]. Acting where the case had to be refused: pro>flash minus pro -6.0 [-11.5, 0.0], flash>pro
minus pro -6.7 [-11.6, -1.9]. Refusing where the case had to be performed: no change (differences -0.7 to +1.3 points).
By stratum, pro>flash minus flash: perform +6.7 [-2.9, +16.1], refuse +9.3 [+2.7, +15.7].
Exploratory: lite>g38 minus g38 -6.7 [-11.5, -2.1] at 67% of its cost; minus lite +65.3.

Hand-off: model A proposed the target action in 59% (pro>flash) and 57% (flash>pro) of episodes. Model B repeated the call in 76%
and 75% of those, and did something else in the rest (more checks, a refusal in text). On refuse cases where B did something else,
the mix succeeded in 82% and 57% of cases against 27% and 14% for A alone on the same cases (11 and 7 cases; these conditional
numbers are approximate, because which episodes hand off is itself a random outcome of the run). With flash-lite as A the hand-off
happens in 83% of episodes and 3.8 flash almost never repeats the call (2%): flash-lite proposes the action before the checks and
3.8 flash goes back and does them.

Run-to-run noise: the same flash arm on the same cases a week earlier (harness study, 5 Oct) differs by -3.3 [-8.2, +1.3],
3.8 flash by +2.0 [-1.1, +5.4].

## E8-0: the output limit

gemini-2.5-pro alone on the 75 perform cases: max_tokens 512 gives success 0.12, 4096 gives 0.68 (+56 points [+40, +72]).
The action rate is the same (0.88 and 0.92). At 512, 22% of calls stop at the limit; truncated tool calls produce tool errors, and
the strict grader fails every episode with a tool error (54 of 75 episodes differ on this flag). Empty replies appear in 23% of
episodes at 512 and none at 4096. The public pro logs (OpenRouter, 512) show the same problem in another form: 68% of episodes
with empty replies, perform success 0.43.

## Verdict against the pre-registration

- Support criterion 1 (primary at least +7 points, interval above zero): met, +8.0 [+2.9, +12.8].
- Criterion 2 (gain where predicted, at the final action on perform cases): not as predicted. The gain is mostly on refuse cases,
  the perform gain is not significant. No loss on refuse cases.
- The negative control (reverse assignment) gains as much or more (+10.7). The plan predicted +1 point for pro>flash and -1 for
  flash>pro from per-step skill profiles. Both predictions are wrong: the gain does not come from one model being better at checks
  and the other at the action. It comes from a second model looking at the history before the irreversible step and acting only
  when it agrees.
- E8-0: the 512 cap is a budget artifact, as suspected on 4 Oct. Per-step profiles taken from the public logs (gpt-5 "checks
  well, rarely acts") should not be used.

Practical reading: within one model generation, a two-model gate before the final action adds 8 to 11 points and removes about
half of the unsafe actions, at less than the cost of the more expensive model alone (flash>pro 3.59 USD against pro 6.22). A newer
single model (3.8 flash, 0.833 for 3.13 USD) is still better than both mixes of the older generation, and a cheap model that
skips checks cannot be rescued fully by a strong model at the gate (lite>g38 0.767).

## Deviations from the pre-registration

- gemini-2.5-* calls went to regional Vertex endpoints instead of `global` after the global endpoint answered 429 most of the time
  (europe-west, europe-north, us-south1, us-west4, northamerica-northeast1, asia-northeast1 and others, round robin; same models).
  Request timeout 150 s and IPv4 after hung connections.
- Retries after an exception were planned at temperature 0.7 (harness default). Infrastructure errors (404 in a region without
  flash-lite, 429, a failed token refresh) made that branch fire, so retries now stay at temperature 0, and the 83 lite>g38
  episodes that had used it or failed were moved to `episodes_test/` and re-run. No other arm had a warm retry.
- Two runners overlapped for a few minutes on the flash arm: 11 duplicate records, the first record per case is used.
- E8-0 on all 75 perform cases (plan said 30), as written in PREREG.md.
