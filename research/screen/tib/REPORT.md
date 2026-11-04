# P0-2: the MF model on TauIndianBankBench (2–3 Nov 2026)

Plan: `PREREG.md` (written before any probe or episode), hashes: `FREEZE.md`. Spend: 19.69 USD of the 30 USD cap
(probe generation 7.20, agents 10.90, user simulator 1.58), every call in `ledger.jsonl`. 480 test episodes plus the
10-episode cost pilot, no episode lost.

## What ran

TauIndianBankBench (commit 1ccb64f, data NPCI/tau-indian-banking): 20 of the 50 scenario families drawn at random, 6 real
tasks each. 120 probes (6 per family) written by gemini-3.8-flash from the family's one-line purpose, the policy, the
assistant instructions, the tool schemas and 3 candidate customers' records, without seeing any benchmark task; every
probe's gold actions executed cleanly on the environment. Agents gemini-3.8-flash and gemini-2.5-flash on all 240 tasks
(real + probes), user simulator gemini-2.5-flash, one attempt. MF parameters frozen from SOPBench.

Real success: 3.8 flash 0.56, 2.5 flash 0.61. Probe success (scoring v2, see deviations): 0.69 and 0.54.

## Results against the pre-registration

| | gemini-2.5-flash | gemini-3.8-flash |
|---|---|---|
| reliability of the real target (split-half, Spearman-Brown) | 0.83 | 0.80 |
| rho, frozen MF from own probes (primary) | **0.63 [0.34, 0.81]** | 0.16 [-0.36, 0.62] |
| rho, raw probe failure rate | 0.64 [0.32, 0.82] | 0.22 [-0.30, 0.65] |
| rho, MF with both agents' probes | 0.66 [0.32, 0.86] | 0.27 [-0.27, 0.67] |
| rho, MF refitted here, leave one family out | 0.42 [0.01, 0.70] | 0.21 [-0.28, 0.63] |
| log-loss on real tasks: frozen MF / raw rate / MF prior only / refit | **0.546** / 0.564 / 0.673 / 0.667 | 0.887 / 0.804 / 0.826 / 0.790 |

1. Rho >= 0.40 with the lower bound above 0 for at least one agent: **met** (2.5 flash, 0.63 [0.34, 0.81]). Not for 3.8 flash.
2. Router better than the best constant policy (upper bound below 0): **not met** in any cost setting.

| costs (USD per case) | best constant | oracle | router on frozen MF | router on raw probes | MF router - best constant |
|---|---|---|---|---|---|
| F 20, human $5 @ 0.99 | 5.20 (human) | 2.83 | 6.10 | 5.59 | +0.90 [-1.30, +3.48] |
| F 20, human $2 @ 0.95 | 3.00 (human) | 1.96 | 4.80 | 4.29 | +1.80 [-0.09, +4.18] |
| F 50, human $5 @ 0.99 | 5.50 (human) | 3.58 | 10.67 | 9.42 | +5.17 [+0.16, +11.43] |
| F 10, human $5 @ 0.99 | 3.92 (2.5 flash) | 2.28 | 3.88 | 3.46 | -0.04 [-1.37, +1.22] |

3. Calibration: for 2.5 flash the frozen MF beats both the raw rate and the no-evidence prior. For 3.8 flash it is worse
   than both: the model carries over 3.8 flash's SOPBench level (about 14% failure) and predicts 0.03 to 0.22 for
   families where it fails 3 to 6 times out of 6 here (real failure 44%).

## Reading

The probes transfer for one agent and not for the other, and the real target is reliable for both (0.80 to 0.83), so
the miss is in the probes, not in the noise of the target. The 3.8 flash misses are concentrated in families whose
difficulty sits in a detail the one-line purpose does not spell out:

- edgbstmtwin ("transactions (corrected window)"): the real tasks mean "from the later of April 1 or the account's open
  date"; the generator read "corrected" as a customer correcting themselves. Real 6/6 failures, probes 0/6.
- tlreqstat and sq2insur: the real tasks end with a raise_request (a follow-up on a still-open request, a claim request),
  and every failed 3.8 flash episode gets the reads right and misses that final request (tlreqstat 5/6 real failures,
  0/6 on probes; sq2insur 5/6 and 1/6).
- hpservice: on the real tasks 3.8 flash makes exploratory calls with empty arguments and stops without the action; the
  probes' customers state more up front and it passes.

So probes are only as good as the document they are written from. On SOPBench the policy text encodes every constraint
the checker uses, which is why probes reached about 0.75 there; here part of each family's difficulty lives only in the
benchmark's task scripts. The router fails for the same reason plus the level problem: with both agents failing around
40% on this benchmark, "always human" is the best constant, and a router that trusts the SOPBench level of 3.8 flash
sends work to it where it fails.

What this says about the model: the frozen MF does not carry an agent's level across benchmarks, and the ranking part
works only when the probes contain the failure mode. Refitting the level on this benchmark (leave one family out) fixes
part of the calibration for 3.8 flash (log-loss 0.790, best of the four) but not the ranking.

## Deviations from the pre-registration

1. Probe generation, first pass: 54 of 164 generator calls were cut by the 8,192-token limit while thinking and returned
   no usable JSON, leaving 6 families short (sq2gold had 0). A second pass with a 32,768-token limit filled them (16 of 16
   calls valid) before any test episode ran. Generation cost 7.20 USD instead of the planned 1 to 2.
2. Probe scoring. The generator gave compare_args only for writes, so every read in a probe was compared on all
   arguments, including the free-text query of search_knowledge_base, which the benchmark never does. After 10 g38 and
   62 flash probe episodes had finished (aggregate success seen, nothing per family examined), `rescore.py` applied to every
   probe action the benchmark's most common compare_args for that tool and re-scored the recorded trajectories
   (trajectories do not depend on compare_args; the v1 re-score reproduced all 240 recorded rewards). Primary numbers use
   v2. With the original scoring (v1): rho 0.55 [0.22, 0.75] for 2.5 flash, 0.13 [-0.36, 0.59] for 3.8 flash; same
   verdicts. `results_v1.json`, `results_v2.json`.
3. The cost pilot's first five 3.8 flash episodes failed on a Gemini 3 thought-signature error (litellm drops it); the
   proxy now re-attaches the signature. The pilot was re-run; pilot tasks are from families outside the test.

## Files

`select_tasks.py`, `gen_probes.py`, `run.py`, `rescore.py`, `analyze.py`, `proxy.py`; `selection.json`, `probes.json` (as
generated), `probes_v2.json` (scoring v2), `probes_log.jsonl`, `runs/*.jsonl`, full messages in `runs/msgs/` (19 MB, keep
out of git), `results_v2.json`, `results_v1.json`, `analyze_v2.out`.
