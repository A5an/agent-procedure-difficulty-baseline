# E5: honest text similarity for TauIndianBank families. Result (12 Nov 2026)

Handoff E5 (Asanali, 11 Nov). No LLM calls, $0. `embed_tib.py` -> `emb_tib.npz`; `ready_proc_e5.py` = `orig_ready_proc.py`
(A5an branch research/2026-11-11-diagnostic-memory) with only the family similarity switched by `SIM_MODE`; outputs
`out_<mode>.txt`, `results_<mode>.json`. Only the TauIndianBank part was rerun.

Text per family: its purpose line. TauIndianBank's `policy.md` is one short global text with no per-procedure section,
so the purpose line is the only procedure-specific text a company would have; no family name or prefix. Embeddings:
BAAI/bge-base-en-v1.5, normalize_embeddings=True, cosine, the recipe of `src/embed_features.py` (installed versions:
sentence-transformers 6.1.0, torch 2.14.0, numpy 2.4.6, from the baseline repo's venv instead of the pinned list in the
handoff; the constant baseline's Brier comes out 0.2325 against 0.2315 in the original run, so split draws differ
slightly between numpy versions). The Gemini-embedding robustness check of the handoff was not run (no Gemini budget).

Three similarities: `tfidf_prefix` (the original: TF-IDF of the purpose plus a token of the family-name prefix such as
edga/hp/sq2), `tfidf` (the same without the prefix token), `bge`.

## Results (1000 dev/qualification splits; CP = share of splits qualified with the one-sided Clopper-Pearson bound,
BS = with the bootstrap bound)

| agent, estimator | Brier: prefix / tfidf / bge | target 0.90 CP, BS: prefix / tfidf / bge | target 0.85 CP, BS: prefix / tfidf / bge |
|---|---|---|---|
| g38, const (0 probes) | 0.2325 | 0, 0 | 0, 0 |
| g38, knn text (0 probes) | 0.231 / 0.244 / 0.248 | 0.00,0.04 / 0.00,0.01 / 0.00,0.01 | 0.01,0.06 / 0.00,0.04 / 0.00,0.02 |
| g38, seeded bb (no Memory) | 0.173 | 0.00, 0.00 | 0.00, 0.02 |
| g38, seeded bb_knn | 0.170 / 0.172 / 0.172 | 0.01,0.20 / 0.01,0.17 / 0.00,0.07 | 0.10,0.25 / 0.09,0.25 / 0.05,0.17 |
| flash, const (0 probes) | 0.2247 | 0, 0 | 0, 0 |
| flash, knn text (0 probes) | 0.251 / 0.268 / 0.264 | 0.02,0.28 / 0.01,0.15 / 0.00,0.05 | 0.11,0.37 / 0.06,0.24 / 0.00,0.06 |
| flash, seeded bb (no Memory) | 0.147 | 0.00, 0.00 | 0.26, 0.24 |
| flash, seeded bb_knn | 0.150 / 0.151 / 0.149 | 0.14,0.42 / 0.07,0.35 / 0.02,0.16 | 0.45,0.43 / 0.41,0.41 / 0.33,0.38 |

## Reading

- Text similarity alone (0 probes) is worse than the constant once the family-name prefix is removed: Brier 0.244-0.268
  against 0.225-0.233. The purpose lines of TauIndianBank do not carry success information about other families.
- "Probes + Memory" was the only rule that sometimes qualified at 0.90 in `ready_proc`. With bge similarity it does so
  in 0-2% of splits by Clopper-Pearson and 7-16% by bootstrap, against 1-14% and 20-42% with the prefix token. About
  half or more of that advantage came from the prefix, which a company would not have.
- At 0.85 for gemini-2.5-flash Memory still adds a little over seeded probes alone (CP 0.33 vs 0.26), at 0.90 it does
  not reach a usable rate.
