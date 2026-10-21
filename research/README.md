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

f2 passes points 1, 2 and 4 of the improvement rule (§8 of the protocol); point 5, the blind test, was run on ThinkingBox-Bench (see "Experiments of the approved plan" below).

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

## Experiments of the approved plan, E1 to E8 and the blind test (13 to 21 October 2026)

Each experiment has a `PREREG.md` written before the run and a `REPORT.md` with the numbers. Same protocol as above (pooled
within-domain ρ over four scenarios, 95% bootstrap intervals over procedures, paired for differences).

| experiment | folder | question | result |
|---|---|---|---|
| E1 | `targets/` | plain shares of solved tasks instead of IRT | ρ 0.128 vs 0.131, difference −0.003 [−0.009, +0.005]; probabilities better calibrated (Brier skill +0.033 vs 0.000) |
| E3 | `e3/` | does an agent's strength depend on the step type | ρ 0.104 vs 0.131; log-loss interval below zero in 1 of 4 scenarios, within-agent AUC drops there: no |
| E4 | `e4/` | does the forecast help to choose which tasks to hand to a human | top 20% hardest by f2: +1 to +3 points over random, baseline about 0, oracle +6 to +7; handing them to the strongest model does not help |
| E5 | `ideas/pilot/` | how many pilot runs per new procedure | ρ 0.129 without pilots, 0.453 with one adaptive pilot, 0.696 with five |
| E6 | `e6/` | pass^k from the one-attempt probability | independent attempts give 0.385 for pass^4 (observed 0.564), beta-binomial 0.579; Brier −0.019 and −0.043 vs baseline |
| E7 | `e7/` | does the zero-shot question D transfer to code tasks | ρ 0.63 SWE-bench Verified, 0.36 Pro, 0.54 Terminal-Bench 2.0, 0.22 GSO; not worse than the judge trained in-benchmark |
| E8 | `e8/` | two models on different steps of one task (SOPBench, 150 cases) | first model leads, second decides on the final action: 78.0% and 80.7% vs 70.0% for the best single model; the order does not matter |
| blind | `../blind/thinkingbox/` | point 5 of the improvement rule on ThinkingBox-Bench (405 tasks, 3,240 agent runs) | D 0.124 [0.022, 0.224], baseline −0.055, text length 0.051; D minus baseline +0.179 [+0.067, +0.297] |

`bias/` holds two post hoc checks without new LLM calls: whether a forecaster favours agents of its own model family
(no, difference +0.014 [−0.037, +0.063] on τ²) and how much of D is text length (τ² 0.57 to 0.52 after removing it).

The code of the earlier rounds that was left out of the first export is added in this update, with its paths moved to
`zr_paths.py`. Third-party sources that the scripts vendored (the τ² and SOPBench source trees, scraped scouting pages), raw
agent episodes and LLM caches are not committed.

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

- HANDBOOK.md was reserved as a blind test and never opened; the blind test that was run is ThinkingBox-Bench, with predictions committed before any agent run.
- Every run has a `PREREG.md` written before it: the variants and the main hypothesis. One run; all variants reported.
- Numbers in reports are copied from result files, never recomputed by hand.

## Data provenance

Features are derived from SOPBench (CC BY 4.0), τ²-bench and τ-Knowledge (MIT). The fresh-benchmark targets are
per-task success rates aggregated from public results of TheAgentCompany (MIT), MCPMark logs (MIT) and AgentSuite
DrafterBench trajectories (task data MIT); the τ² new-population target is aggregated from AgentSuite τ²-bench
trajectories (no license stated on the dataset card). Only aggregated per-task rates are committed.
