# Research beyond the baseline (rounds 1 to 10, 4 to 8 October 2026)

This folder holds the methods we built on top of the baseline in this repository, the features they use, and the
reports of every experiment round. The baseline itself (Agent psychometrics + 18 ADeLe levels, pooled ρ 0.131) and the
evaluation protocol are unchanged; see `../EVALUATION_PROTOCOL.md`.

## Results

Pooled within-domain Spearman ρ over the four new-process scenarios (SOPBench and τ², new procedures and new domain).
Brackets: paired 95% bootstrap interval of the difference to the baseline, resampling procedures.

| method | built from | ρ | vs baseline |
|---|---|---|---|
| baseline | psychometrics + 18 ADeLe levels | 0.131 | |
| plan with tools | ADeLe + an LLM's paper plan of tool calls | 0.214 | +0.083 [+0.021, +0.165] |
| c_lad | ADeLe + plan + procedure steps + compiled policy code | 0.258 | +0.127 [+0.071, +0.172] |
| one question to Sonnet, no training | Sonnet's estimate of the success chance, used as is | 0.294 | +0.163 [+0.045, +0.238] |
| c_lad + data gap | c_lad + where each needed value comes from | 0.311 | +0.180 [+0.099, +0.212] |
| everything + Sonnet estimate (f2) | c_lad + data gap + customer behaviour + naive plan + pre-mortem | 0.347 | +0.216 [+0.126, +0.246] |
| same, four estimates (f2x) | pre-mortem replaced by the mean of four estimates | **0.354** | +0.223 [+0.125, +0.254] |

By scenario (SOPBench new procedures, SOPBench new domain, τ² new procedures, τ² new domain): baseline .120 .125 .128
.150; f2 .242 .193 .421 .532; one question to Sonnet .022 .084 .497 .574. Almost all of the gain is on τ². On SOPBench
the difficulty of a case inside a procedure is decided by the customer's database record, which the text does not show:
even an oracle that knows the procedure and the perform-or-refuse label reaches only 0.614 per case.

f2 passes points 1, 2 and 4 of the improvement rule (§8 of the protocol); point 5, the blind test on HANDBOOK, is open.

Checks on data never used for selection:

| check | method | ρ | for comparison |
|---|---|---|---|
| τ², 30 new agents of 2025 (AgentSuite), new-domain split | f2 | 0.678 [0.375, 0.790] | baseline 0.265 |
| same | one question to Sonnet | 0.682 | |
| τ-Knowledge banking | mean of four pre-mortems, no training | 0.650 [0.537, 0.740] | text length 0.532 |
| TheAgentCompany (175 tasks, 17 agents) | pre-mortem Sonnet / Opus | 0.458 / 0.511 | length 0.179 |
| MCPMark (127 tasks, 14 models) | pre-mortem Sonnet / Opus | 0.250 / 0.371 | length −0.116 |
| DrafterBench (240 cases, 30 models) | Sonnet / mean of three | 0.231 / 0.299 | length 0.026 |

## The methods in one paragraph each

- **Ladder model** (`screen/rubric/ladder.py`). All trained methods use logit P(agent a solves task t) = θ_a − w·x_t,
  fitted by penalised maximum likelihood on training tasks; the penalty is chosen by inner 5-fold CV; the predicted
  difficulty of a new task is w·x_t. Methods differ only in the feature vector x_t.
- **Plan with tools** (`screen/ideas3/plan_tau`, `plan_tools`). Gemini writes, on paper, the ordered tool calls the
  agent must make, the rules that apply, what to ask the customer and the outcome; we count calls, writes, reads,
  distinct tools, rules, questions, refusal, handover and ambiguity.
- **Procedure steps** (`screen/rubric`). Gemini splits the procedure into steps of seven types (check, look-up, ask,
  write, irreversible action, branch, refuse); features are the counts per type plus depth and refusal conditions.
- **Compiled policy code** (`screen/ideas2/compile`, `screen/ideas3/compile_consistency`). An LLM rewrites the policy
  text as a Python program five times; features are the mean number of lines and branches (SOPBench only).
- **c_lad** (`screen/ideas3/combo`). The four feature groups above in one ladder model.
- **Data gap** (`screen/ideas9/gap`). One Gemini call per case labels where every needed argument and every gating
  condition gets its value: given by the customer, derivable, fetched with a tool, asked, missing. 13 counts.
- **Pre-mortem and direct question** (`screen/ideas10/fc`, `ablate`, `zeroshot`, `panel`, `fcx`). Sonnet 5.5 (through
  `claude -p`, no tools) reads policy, tools and request and gives the chance that a typical 2024–2025 agent fully
  succeeds under strict grading; in the pre-mortem variant it first lists concrete failure reasons with probabilities.
  The feature is logit(1 − p). Used as is it ranks τ² better than any trained model; listing failure reasons does not
  matter (ablation), the model does (Haiku 4.5 and Gemini 3.5 Flash barely see difficulty).

What did not work, with numbers, is in `screen/ideas10/NOTEBOOK.md` and in each folder's `REPORT.md`. Only reports are
kept for ideas that are not part of the methods above.

## Layout

| path | content |
|---|---|
| `screen/harness.py` | runs any method on the baseline's folds and scores it with the repository evaluator |
| `screen/zr_paths.py` | all paths; override with `ZR_REPO`, `ZR_DATA`, `ZR_DOCS` |
| `screen/rubric/` | ladder model, step-type rubric, prompts, rubric features |
| `screen/ideas*/<idea>/` | one experiment each: `PREREG.md` (written before the run), `REPORT.md` (result), code, feature `*.csv`, `*score*.txt` |
| `screen/ideas10/NOTEBOOK.md` | chronology of the last round with all numbers |
| `screen/ideas10/fresh/` | the fresh-benchmark test (TheAgentCompany, MCPMark, DrafterBench): items, targets, evaluation |
| `screen/ideas10/newpop/` | the independent 30-agent population on τ² |
| `screen/catalog/` | machine-readable catalog of all 117 ideas tried (`cat_*.json`) and the page builders |

## Running

1. Set up the repository as in the main README and run `make reproduce` once: the scorers read the baseline's
   out-of-fold predictions from `results/`.
2. Run a method from its folder with the repository interpreter, for example the best combination:

   ```
   cd research/screen/ideas10/fc
   PYTHONHASHSEED=0 OMP_NUM_THREADS=4 ../../../../.venv/bin/python run.py     # about 5 minutes, writes out/
   PYTHONHASHSEED=0 ../../../../.venv/bin/python score.py                     # paired intervals, fc_score_main.txt
   ```

   This reproduces 0.347 for f2 from the committed feature tables, with no LLM calls.
3. Scripts that create features call LLMs. Gemini goes through Vertex: `export LLM_BACKEND=vertex
   GOOGLE_CLOUD_PROJECT=<your project>`; the shared cached client is `screen/ideas10/common/vx.py`. Sonnet, Opus and
   Haiku go through the Claude Code CLI (`claude -p --model sonnet --tools ""`). Never commit keys or project ids.
4. Scripts that read raw benchmark sources (the SOPBench repository, τ² task and policy files, τ-Knowledge banking)
   expect them under `ZR_DATA` (default `external_data/raw/`) as `SOPBench-d2622008/`, `tau2-domains/tasks/` and
   `tau2-banking/`.

Raw LLM answers, caches and harness outputs are not committed (see `.gitignore`); the feature tables they produced are.

## Rules we kept

- HANDBOOK.md is the blind test. Do not open it until a method is frozen.
- Every run has a `PREREG.md` written before it: the variants and the main hypothesis. One run; all variants reported.
- Numbers in reports are copied from result files, never recomputed by hand.

## Data provenance

Features are derived from SOPBench (CC BY 4.0), τ²-bench and τ-Knowledge (MIT). The fresh-benchmark targets are
per-task success rates aggregated from public results of TheAgentCompany (MIT), MCPMark logs (MIT) and AgentSuite
DrafterBench trajectories (task data MIT); the τ² new-population target is aggregated from AgentSuite τ²-bench
trajectories (no license stated on the dataset card). Only aggregated per-task rates are committed.
