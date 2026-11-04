# Can a cheaper agent's probes give the expensive agent's percentage? (4 Nov 2026, exploratory, no API calls)

`transfer.py`, output `transfer.out`, 50 TauIndianBank families, seeded probes. Brier skill against the base rate:

| target | own probes | other agent's probes | other, level-shifted | both pooled |
|---|---|---|---|---|
| g38 | +0.26 | +0.15 | +0.14 | +0.26 |
| flash | +0.36 | +0.23 | +0.21 | +0.36 |

Another agent's probes carry a bit more than half of the information: the shared part of procedure difficulty. The rest
is agent-specific, so each agent has to be probed itself; pooling adds nothing to own probes. With flash's probes
alone, delegating g38 at predicted >= 0.9 would give 84% success instead of 91%.

Cost of the probes in the replication ledger (combo_rep): generation about 0.03 USD per probe, an episode about
0.041 USD (g38), 0.006 USD (flash) and 0.003 USD (user simulator). Six seeded probes for both agents cost about 0.5
USD per procedure; the probes + monitor router saved 2.27 USD per case against "always human", so the evaluation pays
back within the first case of a procedure; with sequential stopping (`../seqtest`) it needs 4 to 6 probes.
