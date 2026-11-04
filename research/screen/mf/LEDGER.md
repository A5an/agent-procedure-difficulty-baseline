# Gemini API spend, session from 2 Nov 2026

Cap set by Tima on 2 Nov: 30 USD total. All paid calls go through `../tib/proxy.py`, which writes every call with its
USD cost to `../tib/ledger.jsonl` and refuses calls once the ledger total reaches 29.5 USD (the cap survives restarts).
Prices per million tokens as in the proxy (gemini-2.5-flash 0.30 / 2.50, gemini-3.8-flash 0.75 / 3.75).

| date | what | USD |
|---|---|---|
| 2 Nov | multi-fidelity model, E8 logs, routing, VOI (no API) | 0.00 |
| 2 Nov | P0-2 cost pilot: 5 tasks x 2 agents (+5 failed g38 calls before the thought-signature fix) | 0.28 |
| 2 Nov | P0-2 probe generator tests on one family (two rounds) | 0.42 |
| 2–3 Nov | P0-2 probe generation, 20 families, two passes | 6.78 |
| 2–3 Nov | P0-2 test episodes: 480 (agents 10.90 incl. pilot, user simulator 1.58) | about 12.2 |
| | subtotal, 30 USD session cap | 19.69 |
| 3 Nov | Exp 3, seeded probes: generation 3.19, 240 episodes 6.32 (separate 15 USD allowance) | 9.51 |
| | **total in `../tib/ledger.jsonl`** | **29.20** |

Running total: read `python3 -c "import json;print(sum(json.loads(l).get('usd',0) for l in open('../tib/ledger.jsonl')))"`.
