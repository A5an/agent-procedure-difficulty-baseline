# ready_proc for each seeds condition (13 Nov 2026, item 3 of the 13 Nov plan)

No LLM calls. `ready_proc_k.py` imports the TauIndianBank part of `../e5_embed/ready_proc_e5.py` unchanged (1000 random
25/25 splits of the 50 families into dev and qualification, qualification by one-sided Clopper-Pearson (CP) and by
bootstrap (BS), targets 0.90 and 0.85) and swaps the probe sources for the seedk conditions k0, k2, k5 and foreign, for
the Gemini agents (g38 = gemini-3.8-flash, flash = gemini-2.5-flash; runs of `../seedk`, k0/k2 from the earlier tib and
combo_rep runs) and the Claude agents (Haiku, Sonnet; `../claude_tib`). Family similarity is bge-base without the
family-name prefix, from E5. Not pre-registered: the plan specified what to run, the reading below is written after
seeing the output. Output `out.txt` (all estimators), `results.json`, compact table `table.md`.

The hierarchical-model row (mf) is left out: it needs torch and the model file of the old tib run and was not part of
this request.

Columns: share of the 1000 splits in which the estimator qualified a set of families at the target by CP and by BS, and
the mean real success of the accepted families on the qualification half.

| agent | estimator | Brier | 0.90: share qualified CP / BS | 0.90: real success of accepted | 0.85: share qualified CP / BS |
|---|---|---|---|---|---|
| gemini/g38 | const (0 probes) | 0.232 | 0.00 / 0.00 | n/a | 0.00 / 0.00 |
| gemini/g38 | knn text (0 probes) | 0.248 | 0.00 / 0.01 | 0.63 | 0.00 / 0.02 |
| gemini/g38 | k0 bb_knn | 0.221 | 0.00 / 0.02 | 0.78 | 0.00 / 0.07 |
| gemini/g38 | k2 bb_knn | 0.171 | 0.00 / 0.10 | 0.88 | 0.06 / 0.19 |
| gemini/g38 | k5 bb_knn | 0.159 | 0.00 / 0.12 | 0.89 | 0.06 / 0.15 |
| gemini/g38 | foreign bb_knn | 0.211 | 0.00 / 0.00 | 0.76 | 0.00 / 0.04 |
| gemini/flash | const (0 probes) | 0.226 | 0.00 / 0.00 | n/a | 0.00 / 0.00 |
| gemini/flash | knn text (0 probes) | 0.267 | 0.00 / 0.04 | 0.58 | 0.00 / 0.04 |
| gemini/flash | k0 bb_knn | 0.200 | 0.01 / 0.20 | 0.87 | 0.09 / 0.29 |
| gemini/flash | k2 bb_knn | 0.151 | 0.02 / 0.17 | 0.91 | 0.33 / 0.39 |
| gemini/flash | k5 bb_knn | 0.142 | 0.05 / 0.33 | 0.93 | 0.46 / 0.49 |
| gemini/flash | foreign bb_knn | 0.204 | 0.02 / 0.29 | 0.87 | 0.10 / 0.38 |
| claude/haiku | const (0 probes) | 0.219 | 0.00 / 0.00 | n/a | 0.00 / 0.00 |
| claude/haiku | knn text (0 probes) | 0.251 | 0.00 / 0.10 | 0.76 | 0.00 / 0.11 |
| claude/haiku | k0 bb_knn | 0.202 | 0.01 / 0.33 | 0.88 | 0.07 / 0.35 |
| claude/haiku | k2 bb_knn | 0.164 | 0.02 / 0.21 | 0.87 | 0.07 / 0.20 |
| claude/haiku | k5 bb_knn | 0.150 | 0.03 / 0.21 | 0.89 | 0.09 / 0.17 |
| claude/haiku | foreign bb_knn | 0.216 | 0.00 / 0.31 | 0.86 | 0.02 / 0.32 |
| claude/sonnet | const (0 probes) | 0.219 | 0.00 / 0.00 | n/a | 0.00 / 0.00 |
| claude/sonnet | knn text (0 probes) | 0.234 | 0.00 / 0.11 | 0.78 | 0.01 / 0.11 |
| claude/sonnet | k0 bb_knn | 0.210 | 0.01 / 0.41 | 0.90 | 0.07 / 0.37 |
| claude/sonnet | k2 bb_knn | 0.150 | 0.10 / 0.50 | 0.94 | 0.22 / 0.33 |
| claude/sonnet | k5 bb_knn | 0.151 | 0.14 / 0.43 | 0.94 | 0.28 / 0.35 |
| claude/sonnet | foreign bb_knn | 0.198 | 0.01 / 0.27 | 0.88 | 0.04 / 0.25 |

## Reading

- At 0.90 no condition qualifies reliably by Clopper-Pearson: at most 14% of splits (Claude Sonnet, k5), 0-5% for the
  others. The bootstrap bound qualifies more often (10-50%), but it is the optimistic one because cases of one family are
  dependent.
- Probes from past conversations (k2, k5) give the best Brier for every agent (0.14-0.17 against 0.20-0.22 for k0 and
  0.22-0.23 for the constant), and the accepted families reach 0.87-0.94 real success.
- Another family's conversations (foreign) and documentation probes (k0) are not better than k2/k5 for any agent. For
  the Claude agents k0 bb_knn qualifies by bootstrap about as often as k2/k5 (0.33-0.41), at lower real success (0.88-0.90).
- Text similarity alone (0 probes) never qualifies by CP.

## Limits

- Six real tasks per family make the qualification half small; E3-B (`../e3_more_tasks`) adds the unused tasks, and this
  table will be recomputed with `--extra-real` (Claude agents only, the Gemini agents do not have the new tasks).
- k2 and k5 are an upper bound for real past tickets (templated families, see `../seedk/REPORT.md`).
- The Claude runs use Claude Haiku as the customer for probes and real tasks; the Gemini runs use the Gemini customer.
