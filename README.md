# Predicting agent success on unseen procedures: baseline

Before an LLM agent is deployed on a business procedure, can we tell from the procedure's
documentation alone how reliably it will execute it? This repository contains the baseline for that
question, the evaluation protocol, and everything needed to reproduce the numbers.

**Baseline.** Agent psychometrics (Ge et al., COLM 2026; their code, unchanged) with the 18 ADeLe
demand levels (Zhou et al., Nature 2026; their rubrics and prompt) as the task features: IRT gives
each agent an ability and each training task a difficulty, a ridge regression learns difficulty
from the demand levels of the task text, and a new task gets P(success) = sigmoid(ability -
predicted difficulty).

**Evaluation.** Three benchmarks where the agent must follow written procedures (SOPBench,
tau2-bench, tau-Knowledge banking). The primary metric is how well a method ranks task difficulty
inside a domain on procedures and domains never seen in training, following Krsteski and Meyer
(2026). Pooled AUC is reported only for comparison with the original paper, because it mostly
measures differences between agents.

- `METHOD.md`: every step, what it does and which paper it comes from
- `EVALUATION_PROTOCOL.md`: metrics, reliability checks, the choice of the baseline, and the rule a
  new method has to meet to count as an improvement

## Results

Mean rank correlation between predicted and fitted task difficulty inside a domain, over the four
scenarios where the procedure is new (SOPBench and tau2-bench, new procedures and new domain), with
a 95% procedure bootstrap interval. Full tables: `EVALUATION_PROTOCOL.md`, section 6.

| method | mean rho [95% CI] | worst scenario |
|---|---|---|
| **Agent psychometrics + ADeLe demands (baseline)** | **0.131 [0.027, 0.255]** | 0.120 |
| kNN router | 0.124 [-0.001, 0.181] | 0.010 |
| ADeLe demands fitted jointly as LLTM | 0.113 [0.020, 0.237] | 0.097 |
| estimated human time (reference) | 0.107 [-0.002, 0.283] | 0.079 |
| their grouped ridge, embedding + ADeLe | 0.060 [-0.052, 0.181] | -0.009 |
| routers + embedding (best by pooled AUC) | 0.059 [-0.021, 0.167] | -0.104 |
| Agent psychometrics, embedding only | -0.008 [-0.110, 0.171] | -0.172 |

The baseline has the highest mean and the highest worst case, and no method is significantly better
than it in any of the four scenarios. It is still weak: the reliability of the target puts the
ceiling near 0.9, and its probabilities are barely better than agent ability alone. Section 7 of
`EVALUATION_PROTOCOL.md` explains why this baseline and not the other families.

## Reproduce

Requirements: Python 3.11 (or [uv](https://docs.astral.sh/uv/), which fetches it), git, make, about
2 GB of disk without the raw data, 8 GB with it. Everything runs on a laptop CPU; no GPU is needed.

```bash
git clone https://github.com/a5an/agent-procedure-difficulty-baseline.git
cd agent-procedure-difficulty-baseline
make setup        # .venv with pinned packages, and the two upstream repositories at pinned commits
```

Clone without `--recursive`: agent-psychometrics contains a nested submodule entry without a URL,
which makes a recursive clone fail; `make setup` fetches the two submodules this repository needs.

### 1. The reported numbers, no key needed (15 to 20 minutes on a laptop)

```bash
make reproduce    # out-of-fold runs of every method, final evaluation, cross-benchmark test,
                  # then a comparison of results/ with reference_results/
```

The benchmark inputs, the IRT fits and the task features are shipped in `data/`, including every
answer of the LLM judge (`data/*/adele_responses.jsonl`, `data/*/human_time.jsonl`). The last step
prints `OK` for each results file that matches the reference.

### 2. With your own key

The only step that calls an LLM is the annotation of task features. Put a key in `.env`:

```bash
cp .env.example .env     # then set GEMINI_API_KEY=...  (or GOOGLE_CLOUD_PROJECT=... for Vertex AI)
make smoke               # one call, to check the key
make annotate FRESH=1    # re-annotate everything with your key (old answers kept as *.previous.jsonl)
make run evaluate compare
```

`make annotate` without `FRESH=1` only fills what is missing from the caches. The reference run
made 25,683 calls to `gemini-3.8-flash` (about 40 M input tokens and 20 M output
and thinking tokens), with 80 parallel requests. `LLM_MODEL` switches the judge; a different judge
gives different levels, so expect the numbers to move within the annotation noise reported in
`EVALUATION_PROTOCOL.md` (a second annotation by the same judge keeps 89 to 92% of levels
identical).

### 3. Everything from the raw releases (several hours)

```bash
make full
```

This downloads the original releases (6 GB; SOPBench from GitHub at a pinned commit, tau2-bench task
files at a pinned commit, the leaderboard runs from the public tau2-bench bucket), checks every file
against `data_manifest.tsv`, rebuilds `data/` (the rebuilt files are byte-identical to the shipped
ones), refits the IRT, recomputes the embeddings (downloads DeepSeek-R1-Distill-Qwen-1.5B and
bge-base-en-v1.5 from Hugging Face), recomputes the features from the cached answers, reruns
everything, runs the reliability checks, and reproduces Table 5 of Agent psychometrics with their
own data (all ten published numbers match to three decimals; `reference_results/table5.csv`, and
the same with statement-only inputs in `table5_statement_only.csv`).

`make help` lists the individual steps.

## Layout

| path | what |
|---|---|
| `src/download_data.py` | fetch and verify the raw releases (`data_manifest.tsv`) |
| `src/build_sopbench.py`, `src/build_tau.py` | outcome matrices, task statements and task tables |
| `src/fit_irt.py` | full-data 1PL IRT with the Agent psychometrics trainer |
| `src/embed_features.py` | length, their embedding recipe, bge-base-en-v1.5 |
| `src/adele_annotate.py` | ADeLe demand levels; `validate` compares the judge with GPT-4o on their battery |
| `src/human_time.py` | estimated human time (reference predictor) |
| `src/methods.py` | kNN router, amortised IRT / LLTM, IRT-Router-style MIRT |
| `src/run_baseline.py` | out-of-fold predictions of every method on every split |
| `src/evaluate.py` | final evaluation (`results/final_evaluation.json`, `.txt`) |
| `src/lobo.py` | leave-one-benchmark-out (`results/lobo.json`) |
| `src/check_target.py`, `src/adele_retest.py` | reliability of the target and of the annotation |
| `src/reproduce_table5.py` | Table 5 of Agent psychometrics, with their data |
| `src/compare_reference.py` | compare `results/` with `reference_results/` |
| `data/` | benchmark inputs, IRT fits, features and LLM caches (see `DATA_LICENSE.md`) |
| `reference_results/` | the results the documents report |
| `third_party/` | agent-psychometrics and ADeLe-AIEvaluation as submodules at pinned commits |

## Notes on determinism

- The fold IRT of agent-psychometrics writes its training file by iterating over a Python set, so
  the order of tasks depends on the process's string hash seed, and its trainer does not seed the
  random generators. `run_baseline.py` fixes `PYTHONHASHSEED=0` and seeds every fold, which makes
  a rerun on the same machine identical to the last digit. Without that, repeated runs of the
  upstream code move the baseline by at most 0.005 in any scenario, but the embedding-only model by
  up to 0.27 (tau2, new domain).
- Floating point on another CPU architecture can move the last digits of the IRT fits;
  `compare_reference.py` uses a tolerance of 0.005.
- The LLM answers are the only non-deterministic input. They are cached, so the default path does
  not depend on them.

## References

- Ge, Kryvosheieva, Fried, Girit, Hariharan. Agent psychometrics: task-level performance
  prediction in agentic coding benchmarks. COLM 2026. arXiv 2604.00594.
  Code: github.com/dariakryvosheieva/agent-psychometrics
- Zhou et al. General scales unlock AI evaluation with explanatory and predictive power. Nature,
  2026. doi 10.1038/s41586-026-10303-2. Code: github.com/Kinds-of-Intelligence-CFI/ADeLe-AIEvaluation
- Krsteski, Meyer. Predicting task difficulty without rollouts. 2026. arXiv 2608.05797.
- Li et al. SOPBench: evaluating language agents at following standard operating procedures and
  constraints. 2025. arXiv 2503.08669. Data: github.com/Leezekun/SOPBench
- Barres et al. tau2-bench: evaluating conversational agents in a dual-control environment.
  ICML 2026. arXiv 2506.07982. Data: github.com/sierra-research/tau2-bench
- Rasch. Probabilistic models for some intelligence and attainment tests. 1960.
- Fischer. The linear logistic test model as an instrument in educational research. Acta
  Psychologica, 1973. De Boeck, Wilson. Explanatory item response models. Springer, 2004.
- Truong et al. Reliable and efficient amortized model-based evaluation. ICML 2025.
- Song et al. IRT-Router. ACL 2025. Li. kNN routing, arXiv 2505.12601, 2025.
- Kwa et al. Measuring AI ability to complete long software tasks. NeurIPS 2025. arXiv 2503.14499.
