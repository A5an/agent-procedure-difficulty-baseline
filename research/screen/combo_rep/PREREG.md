# Replication of Exp 5 on 30 new TauIndianBank families (pre-registration, 4 Nov 2026)

Approved by Tima on 4 Nov ("максимум 45 даю"): 45 USD of Gemini on this folder's ledger (`ledger.jsonl`, cap in
`cap_usd.txt`). Written before any probe or episode of the new families. Exp 5 (`../combo/REPORT.md`) found on the 20
families of P0-2 that probes plus a probe-trained monitor beat "always human"; the result was fragile.

## Data

`select_rep.py` (seed "20261104-rep"): the 30 families not used in P0-2, 6 real tasks per family, 2 other tasks per
family as the seeds of the seeded probes, P0-2 pilot tasks never drawn, probe customers exclude every customer of these
families' tasks. Agents gemini-3.8-flash and gemini-2.5-flash, user simulator gemini-2.5-flash, harness and agent
instructions of P0-2 (`run_rep.py` is `../tib/run.py` with this folder's selection and outputs; proxy as in
`../synth`, which re-asks empty agent replies).

Probes (`gen_rep.py`): 6 seeded probes per family (Exp 3 prompt with the two seeds) and 6 documentation probes per
family (P0-2 prompt), generator gemini-3.8-flash, same candidate filter and gold-action check; max_tokens 32768 from the
first call; the per-tool compare_args convention applied at generation for both kinds.

## Order and stop rule

1. seeded probe generation; 2. real episodes, both agents; 3. seeded probe episodes, both agents; 4. documentation
probe generation; 5. documentation probe episodes. Spending stops at 45 USD. Steps 1 to 3 are needed for the primary
hypothesis; if the cap stops step 4 or 5, the analyses that need documentation probes are reported as not run.

## Analysis (`analyze_rep.py`, frozen; the decision rule, monitor, curves and costs of `../combo/combo.py`, unchanged)

Router per family over {human, agent a, agent a + monitor at flag rate 10% to 90%} by expected loss; realized loss on
the real cases; paired bootstrap over families, 4000 draws. Costs: F 20, human 5 USD at 99% (primary), plus the three
other settings of combo.py.

- R1 (primary): on the 30 new families, the router with calibrated seeded-probe estimates and the monitor trained on
  seeded probe episodes ("combined [seeded+cal | M2 seeded]") beats the best constant policy (upper bound of the
  difference below 0). On the earlier 20 families this variant gave -1.64 [-2.92, -0.16].
- R2: on the 30 new families it beats the probes-only router with the same estimates (upper bound below 0).
- R3 (if documentation probes ran): the same with the monitor trained on documentation and seeded probe episodes
  (the strongest variant on the earlier 20: -2.32).
- R4: R1 on all 50 families pooled.
- Descriptive: monitor AUC on real episodes (pooled and within family x agent), M2 vs M1.

A replication failure (R1 not supported) is reported as such; no variant is added afterwards to rescue it.
