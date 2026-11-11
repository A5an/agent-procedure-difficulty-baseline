# E8 log analysis (task P1-4 of the context repo, 2 Nov 2026)

No new runs, no API cost. Inputs: the recorded E8 episodes (`../e8/episodes/`, 150 SOPBench cases, one attempt per
case and arm) and the 5 Oct repeat runs of the same flash and 3.8 flash configs (harness study), used as a run-to-run
noise reference. Code: `analyze.py` (output `analyze.out`, `results.json`), `router_sim.py` (`router_sim.json`).

## Q1. How many cases did the second model rescue, how many did it spoil

Two counts, because they answer different questions.

**Cross-run** (pair arm vs model A alone, two separate runs of the same case):

| comparison | rescued | spoiled | net |
|---|---|---|---|
| pro>flash vs flash alone | 17 | 5 | +12 |
| flash>pro vs flash alone | 18 | 2 | +16 |
| pro>flash vs pro alone | 22 | 9 | +13 |
| flash>pro vs pro alone | 22 | 5 | +17 |
| noise: flash 13 Oct vs flash 5 Oct (same config) | 6 | 11 | -5 |
| noise: 3.8 flash 13 Oct vs 5 Oct | 4 | 1 | +3 |

Two identical flash runs disagree on 17 of 150 cases. So a single "rescued" case across runs means little; only the
net counts are interpretable, and they are well above the noise net.

**At the gate** (inside the pair episode we know the call A proposed and what B did, so no second run is needed):

| | pro>flash (B = flash) | flash>pro (B = pro) |
|---|---|---|
| refuse cases where A proposed the forbidden action | 18 | 12 |
| ... B did not act, case solved (true rescue) | 9 | 4 |
| ... B repeated the action (both wrong) | 7 | 5 |
| perform cases where A proposed the action | 70 | 73 |
| ... B repeated it at once | 60 (49 solved) | 59 (48 solved) |
| ... B checked more, acted later | 8 (4 solved) | 14 (9 solved) |
| ... B never acted (wrongful refusal, a spoil) | 2 | 0 |

The gate stops about half (pro>flash) or a third (flash>pro) of the forbidden actions A was about to take, and spoils
at most 2 to 6 perform cases (the "acted later and failed" cases may have failed for A too). The cross-run net
(+12 to +17) is larger than the gate count (+3 to +7); the difference is run-to-run behaviour before the gate.

## Q2. Is the gain concentrated on cases a predictor calls hard

Gain of the pair over model A alone by tercile of predicted difficulty (bootstrap over procedures):

| predictor | arm | easy third | middle | hard third |
|---|---|---|---|---|
| Sonnet pre-mortem (text, zero-shot) | flash>pro | +0.08 [+0.02, +0.16] | +0.06 [0.00, +0.13] | +0.18 [+0.03, +0.27] |
| Sonnet pre-mortem | pro>flash | +0.06 [-0.02, +0.15] | +0.10 [-0.03, +0.23] | +0.10 [+0.02, +0.26] |
| public IRT b (28 agents' runs, warm start) | flash>pro | 0.00 [-0.06, +0.06] | +0.08 [+0.02, +0.16] | +0.24 [+0.13, +0.31] |
| public IRT b | pro>flash | +0.06 [-0.02, +0.15] | +0.10 [-0.03, +0.22] | +0.10 [-0.05, +0.26] |

The text predictor separates A-alone failures only weakly on these cases (AUC 0.63) and the gain is not reliably
concentrated by it. Difficulty from other agents' real runs separates well (AUC 0.84 to 0.92) and concentrates the
flash>pro gain in the hard third, but not the pro>flash gain.

Selective gate (gate only the top x% predicted-hard cases, flash>pro): with IRT b, gating half of the cases gives
0.807, the same as gating all, at 81% of the API cost (2.91 vs 3.59 USD per 150 cases). With the text predictor half
gives 0.773 against 0.753 for a random half. The gate already fires only when A proposes the action (57 to 59% of
episodes), so most of the saving is already there.

## Q3. Are errors more frequent where the two models disagree

Yes, by a wide margin.

- At the gate. Perform cases: success 0.82 when B repeats A's call, 0.40 (pro>flash) and 0.64 (flash>pro) when B deviates.
  Refuse cases: when B agrees with A's forbidden action the case is lost every time (12 of 12); disagreement is where the
  rescues are.
- Independent runs. flash and pro alone agree on act / not act in 90% of cases. When they agree flash is right 74% of
  the time, when they disagree 33% (pro 75% vs 20%). Two runs of the same flash config disagree on 7% of cases, and on
  those flash is right 10% of the time.

Escalating gate disagreements to a human (assumed correct) sends 14% of cases to the human and lifts flash>pro from
0.807 to 0.86. The errors disagreement cannot catch are the ones both models make: forbidden actions both agree on
(7 + 5 cases here), which matches the literature on correlated judge errors.

## Expected loss of simple policies (input for the router one-pager)

`router_sim.py`. Loss per case = API cost + human cost + F x failure + U x unsafe action. At F = 20 USD:

| policy | success | unsafe | human share | loss, U=0, human $5 @ 0.99 | loss, U=200, human $5 @ 0.95 |
|---|---|---|---|---|---|
| flash alone | 0.700 | 0.093 | 0 | 6.01 | 24.67 |
| 3.8 flash alone | 0.833 | 0.040 | 0 | 3.35 | 11.35 |
| flash>pro gate | 0.807 | 0.047 | 0 | 3.89 | 13.22 |
| flash>pro gate, disagreement to human | 0.853-0.859 | 0.034 | 0.14 | 3.55 | 10.80 |
| human always | 0.95-0.99 | 0.005-0.025 | 1 | 5.20 | 11.00 |

API cost is cents per case, so the choice is driven by the error terms and by the human's price and accuracy. With a
large price on unsafe actions the human or "gate + human on disagreement" wins, without it the newest single model
wins. The ordering flips with the human assumptions, which is why the one-pager has to state them.

## Caveats

One attempt per case and arm, 150 cases, 46 procedures, one benchmark with a scripted user. Cross-run comparisons mix
the gate effect with run-to-run noise. "Human" is an assumption (cost, accuracy), not a measurement.
