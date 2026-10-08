# Brief for method-screening agents (4 Oct 2026)

## Goal
The plan for the supervisor lists methods from the papers that we have not run yet. Each agent runs its group of
methods (and their combinations with the baseline) under exactly the baseline protocol, so every number is
comparable to the baseline (pooled within-domain rho 0.131 [0.027, 0.255] over the four new-process scenarios).

## Project in two lines
Predict, from the documentation of a new business procedure, how reliably LLM agents execute it. Baseline:
two-stage Agent psychometrics (fold IRT, then ridge from task features to difficulty beta, P = sigmoid(theta - beta_hat))
with the 18 ADeLe demand levels as features.

## Where things are
- Public repo (READ ONLY, never modify, never commit): <repo>
  - data/<bench>/: responses.jsonl (agent x task outcomes), tasks.csv (case_id, domain, procedure, procedure_id, ...),
    statements.jsonl (the task text a method may see), features/*.csv|npz (length, adele, adele16, human_time, embeddings),
    irt/1d_1pl/{abilities,items}.csv (full-data IRT, used only as the evaluation target)
  - src/run_baseline.py (folds, predictors), src/evaluate.py (metrics), src/methods.py (KNNRouter, AmortizedIRT, AmortizedMIRT),
    src/llm.py (Gemini client), EVALUATION_PROTOCOL.md (metrics and the improvement rule in section 8)
  - Benchmarks: sopbench (830 tasks x 28 agents x 1 attempt), tau2 (278 x 15 x up to 4), tauk_banking (97 x 12 x 4, one domain)
- Shared harness: research/screen/harness.py (read its docstring). It runs your
  predictors on the same folds as the baseline (fold IRT read from the repository cache), always includes constant,
  oracle and the baseline "adele", and scores with the repository evaluator. Self-test reproduced the baseline 0.131 exactly.
  Do not edit harness.py. If you need different behaviour, copy it into your folder.
- Run Python as: PYTHONHASHSEED=0 <repo>/.venv/bin/python script.py
- A custom predictor needs: fit(data, train_task_ids), predict_probability(data, agent, task) -> float, and optionally a
  dict attribute _predicted_difficulties {task: beta_hat}. See src/methods.py and the agent-psychometrics classes
  (third_party/agent-psychometrics/experiment_new_tasks/) for the `data` object (train_abilities, responses, ...).
- Earlier prototype code for several methods (may need porting): <project>/prototype/baseline/
  (entropy.py, hidden_probe.py, llm_forecast.py, bpm_metrics.py, extra_methods.py, ...)
- SOPBench raw run files with per-case grader checks: <ZR_DATA>/SOPBench-d2622008/output/<domain>/ast_<agent>-dep_full-fmt_structured-tool_full-shuffle_False.json
  (each record: task, interactions, evaluations[0] with dirgraph_satisfied, action_called_correctly, constraint_not_violated, database_match, success). Task definitions: .../SOPBench-d2622008/data/*_tasks.json

## Rules
1. Work only in your own folder research/screen/<your_name>/. Do not write anywhere else.
2. No leakage. Features may use only what the agent sees before running: statements.jsonl text, the procedure/SOP text,
   tool lists. Never use as a feature: gold actions, evaluation criteria, the hidden initial database, the should_succeed /
   action_should_succeed flag, any outcome of a test task. SOPBench rule trees restate the SOP text the agent sees, so
   counts derived from them are allowed but must be labelled "from the rule tree".
3. Targets may use outcomes, but only of training tasks inside a fold (the harness handles this for predictors).
4. LLM calls (only agents told to use them): use src/llm.py from the repo with Vertex:
   `LLM_BACKEND=vertex GOOGLE_CLOUD_PROJECT=$(gcloud config get-value project 2>/dev/null)` and model gemini-3.8-flash,
   temperature 0. Never print, log or write any key or the project id. Cache every response in your folder (jsonl) so
   reruns are free. Keep the total under 10,000 calls.
5. HANDBOOK.md is a blind test set. Do not open its tasks or anything about it.
6. Be honest: if something cannot be run, say why in the report. Do not tune anything on the test folds.
7. Long runs: run them in the foreground with a generous timeout, or in the background and wait for them; do not stop early.

## Report (required): <your folder>/REPORT.md, in English, plain
1. What was run, one paragraph per method: source paper, exact setup, deviations from the paper.
2. A results table with, for each method and each combination: pooled rho_mean, its 95% CI, worst scenario, rho by the four
   scenarios, the paired 95% CI of the difference to the baseline, mean Brier skill and log-loss skill over the four
   scenarios, and the verdict under the improvement rule of EVALUATION_PROTOCOL.md section 8 (better / not better / worse).
3. What could not be run and why. 4. Paths of scripts and eval.json.
Return a summary of at most 12 lines with the table's key numbers.
