# E7: the zero-shot forecaster outside business procedures (13 Oct 2026)

Run: `e7.py` (1,333 Sonnet 5.5 calls through `claude -p`, none missing, about 20 USD equivalent on the subscription),
`score.py` (Spearman, `e7_score.json`, `scores_<bench>.csv`), `t5.py` (Agent psychometrics Table 5 pipeline, `t5/*/table.csv`).
Texts from the Hugging Face datasets-server (`fetch.py`), statement level only, as in their information ablation.

## 1. Spearman with IRT difficulty, per benchmark (95% bootstrap over tasks)

| benchmark | n | Sonnet, no training (D) | judge 15, trained in-benchmark (5-fold CV) | judge 15, trained on other benchmarks (Table 5) | embedding, in-benchmark CV | length |
|---|---|---|---|---|---|---|
| SWE-bench Verified | 500 | 0.632 [0.572, 0.689] | 0.516 [0.440, 0.582] | | 0.371 | 0.126 |
| SWE-bench Pro | 641 | 0.362 [0.291, 0.430] | 0.353 [0.282, 0.421] | 0.231 [0.155, 0.306] | 0.366 | -0.204 |
| GSO | 102 | 0.215 [0.028, 0.397] | 0.073 [-0.085, 0.230] | 0.020 [-0.166, 0.209] | -0.062 | -0.048 |
| Terminal-Bench 2.0 | 89 | 0.537 [0.364, 0.680] | 0.457 [0.275, 0.610] | | 0.543 | -0.149 |

D minus the in-benchmark judge: +0.116 [+0.058, +0.171], +0.009 [-0.077, +0.096], +0.142 [-0.084, +0.376], +0.079 [-0.071, +0.239].
D minus the held-out-benchmark judge: SWE-bench Pro +0.130 [+0.037, +0.222], GSO +0.195 [+0.002, +0.403].
Ridge on judge 15 plus D (in-benchmark CV): 0.627, 0.444, 0.119, 0.480. On SWE-bench Pro the combination beats D by 0.082 [0.031, 0.133].
Terminal-Bench against the binary IRT fit: D 0.540 (primary target is the binomial fit).

## 2. Table 5 format (held-out benchmark AUC, their pipeline and bootstrap)

| features of the "judge" slot | SWE-bench Pro | GSO |
|---|---|---|
| their baseline (no task features) | 0.571 | 0.637 |
| their statement-only judge (15 features) | 0.652 | 0.683 |
| their statement-only combined (judge + embedding) | 0.653 | 0.699 |
| their default judge, which also sees the solution patch (paper) | 0.696 | 0.743 |
| D alone (one number) | 0.679, vs baseline +0.108 [+0.084, +0.132] | 0.747, +0.110 [+0.066, +0.151] |
| judge 15 + D | 0.688, +0.117 [+0.094, +0.138] | 0.719, +0.081 [+0.032, +0.129] |

89 SWE-bench Pro tasks are no longer in the public dataset; they got the median D, which can only lower D's AUC there.

## Verdict

The pre-registered hypothesis holds: on all four benchmarks D ranks tasks at least as well as the statement-only judge trained
inside the benchmark (significantly better on SWE-bench Verified), and much better than the same judge trained on other benchmarks,
which is the setting that matches ours. In the paper's own held-out-benchmark format, one zero-shot number at statement level
reaches the AUC of their best judge that also reads the gold patch. Weakest on GSO, as predicted (the text is a repository and an
API name).

Caveat: SWE-bench Verified and its leaderboard are widely discussed online, and Sonnet 5.5 may have seen material about which
issues are hard. SWE-bench Pro, GSO and Terminal-Bench 2.0 are newer and smaller in public discussion, and the result holds there too,
but contamination cannot be ruled out for any public benchmark. The blind tests (HANDBOOK, ThinkingBox) remain the real check.
