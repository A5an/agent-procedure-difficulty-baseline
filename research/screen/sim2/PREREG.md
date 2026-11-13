# E2: does the probe outcome depend on who plays the customer (pre-registration, 12 Nov 2026)

Written before any call of this experiment. Plan from the handoff of 11 Nov (Asanali), E2. Approved by Tima on 12 Nov:
pilot first; E1 and E2 pilots share 20 USD on the ledger of `../seedk` (one proxy, port 8095). The full run needs a
separate approval after the pilot.

## Question

All probes so far ran with one user simulator (gemini-2.5-flash). LLM customers are known to inflate success
(arXiv 2603.11245). If the same probe passes with one simulated customer and fails with another, the probe-based
decision depends on the simulator.

## Design

- The existing 300 seeded probes (k=2; `../tib/seeded/probes.json`, `../combo_rep/probes_seeded.json`) run again with the
  user simulator gemini-2.5-pro. The simulator prompt is unchanged; only the model changes. One deviation that cannot be
  avoided: the proxy turns the simulator's thinking off (budget 0), which gemini-2.5-pro does not allow, so it gets the
  minimum budget of 128 tokens.
- Both agents, one attempt per probe, harness `../seedk/run_seedk.py` (same settings as the earlier runs). Outputs
  `runs/seeded_pro_<agent>.jsonl`; the flash-simulator outcomes are the recorded ones.

## Pilot

The 5 pilot families of E1 (hpemi, edgbcarddisamb, sqmand, hpgold, edgbfdcap): 30 probes x 2 agents = 60 episodes. Purpose:
errors, episode length and price with the pro simulator. Pilot families stay in the full run.

## Outcomes (full run, 50 families)

- Share of probes whose outcome differs between the two simulators, per agent, with a bootstrap interval over families.
- Spearman with the real failure rate over families for the flash simulator, the pro simulator and their mean; paired
  differences, bootstrap over families (2000, seed 20261112).
- The set of families with 6 of 6 probes passed under flash, under pro and under both; real success of each set.
- Support for "the simulator matters": the disagreement share has a lower bound above 0.10 for at least one agent, or
  the "6 of 6" set changes by at least 3 families.

## Stated up front

Both simulators are Gemini models with the same prompt, so this tests the model behind the customer, not human
customers. A difference in thinking budget (0 vs 128) is part of the comparison.
