# Brief for the ideas round (night of 4 to 5 Oct 2026)

Read this file, then research/screen/SCREEN_BRIEF.md (data, harness, rules).
Everything in SCREEN_BRIEF.md applies. This file adds what is new.

## Goal of the round
The supervisor wants new ideas and methodologies that close the gap better than what we screened so far, each with
evidence. Your job: implement ONE new idea (described in your task), run it under exactly the baseline protocol with the
shared harness, and report honestly whether it helps. A clean negative result is a useful result. Do not tune on test
folds, do not cherry-pick variants after seeing test numbers (fix your 2 to 4 variants before the first full run and
report all of them).

## Numbers you compare against (pooled within-domain rho over the four new-process scenarios)
- Baseline `adele` (Agent psychometrics two-stage, ridge on 18 ADeLe levels): 0.131 [0.027, 0.255];
  by scenario sop-proc 0.120, sop-dom 0.125, tau2-proc 0.128, tau2-dom 0.150.
- Best earlier rubric results: `lad2_adele` 0.182 (paired CI vs baseline [-0.061, +0.101]), `grp_proc_adele` 0.172.
  Their out-of-fold predictions: research/screen/rubric/out/run/<bench>/obs_<scheme>.csv.gz
- Fine-tuned bge-small encoder 0.255, almost all from tau2: research/screen/rerun/out_final
- A gain is significant only if the paired 95% interval of the difference lies above 0. In practice that needs about +0.08.

## Facts found tonight that should shape your design
1. SOPBench: within a domain, case difficulty is dominated by the hidden case state. Each case either must be performed
   (`should_succeed` = 1, 35% of cases, mean beta +1.64) or must be refused (mean beta -0.62). An oracle that knows only
   this label reaches within-domain rho 0.52 to 0.55. An oracle that knows each procedure's exact mean difficulty (the best any
   procedure-level documentation feature can do) reaches only 0.175 (new procedures) and 0.231 (new domain).
   So on SOPBench, case-level rho above about 0.2 requires reading the individual case (the customer message and its
   parameters against the policy), not only the procedure.
2. The label itself is forbidden as a feature. It may be used (a) for evaluation and analysis, (b) as a training target on
   training-fold tasks only (privileged information), never for test tasks.
3. The rubric's text-based p_refusal predicts the label only weakly (AUC 0.61, within domain 0.57).
   SOPBench statements contain the action's policy and the customer message with its parameters (amounts, names, ids).
   Some conditions can be checked against the message alone, others need the hidden database.
4. tau2 statements contain only the customer scenario. The domain policies are in
   <ZR_DATA>/tau2-domains/tasks/ (airline_policy.md, retail_policy.md,
   telecom_main_policy.md, telecom_tech_support_manual.md, telecom_tech_support_workflow.md). Any LLM feature for tau2 should
   receive the policy of its domain (the agent receives it too). This is allowed.
5. At the procedure level (mean predicted vs mean fitted difficulty per procedure, SOPBench new domain) the baseline reaches
   0.231 and lad2_adele 0.411 (paired CI [-0.008, +0.370]). The extra metrics module reports this.

## Shared tools in research/screen/ideas/
- `cpulock.py`: wrap every harness run (`H.run_all`, `H.evaluate_all`), every model fitting loop, embedding or big file parse in
  `with cpu_lock("<your name>"):`. Only one agent holds it at a time; the laptop overheats otherwise. Never hold it around LLM calls.
- `metrics_extra.py`: after `H.evaluate_all`, call `metrics_extra.report(out_dir, [your columns], extra_dirs={"lad2_adele":
  ".../rubric/out/run", "grp_proc_adele": ".../rubric/out/run"}, combos=[...])`. Report its numbers too (procedure-level rho with
  paired CI, label-conditional rho, hard-case recall at 20%, selective lift at 50% coverage).
- Existing LLM features you may reuse as inputs: the procedure rubric research/screen/rubric/features/rub_proc_<bench>.csv
  (12 columns), ADeLe in the repo data/<bench>/features/adele.csv, human_time.csv, length.csv. The formula ladder
  (logistic L2 with global features) is research/screen/rubric/ladder.py.

## Running
- Python: `cd <your folder> && PYTHONHASHSEED=0 OMP_NUM_THREADS=4 nice -n 10 <repo>/.venv/bin/python script.py`
- `import sys; sys.path.insert(0, "<project>/prototype/screen"); sys.path.insert(0, "research/screen/ideas")`
  then `import harness as H, metrics_extra as X; from cpulock import cpu_lock`.
- You may restrict the harness to `benches=["sopbench", "tau2"]` and `schemes=["new_procedures", "new_domain"]` when banking or
  random cases would cost LLM calls. Say so in the report.
- No GPU/MPS, no local LLM inference, no fine-tuning. Small CPU models (ridge, logistic, BT fits) only.
- LLM (only if your task says so): repo `src/llm.py`, `LLM_BACKEND=vertex GOOGLE_CLOUD_PROJECT=$(gcloud config get-value project 2>/dev/null)`,
  model gemini-3.8-flash. `llm.generate(prompt)` returns (text, usage). Temperature 0 by default; for sampled variants read llm.py
  to see how to pass a temperature (copy the function into your folder if it does not support one; do not edit the repo).
  Use a thread pool of at most 8 parallel calls. Cache every response (jsonl keyed by a hash of the prompt) so a rerun is free.
  Stay inside your call budget. Never print or write any key or the project id.
- Test your code on a small subset first (a few tasks, one fold) before the full run.

## Report: <your folder>/REPORT.md (English, plain, no semicolons, no em dashes)
1. The idea in two plain sentences, its source in the literature (author, year, venue or arXiv id) or "new".
2. Exact setup, every variant, deviations, LLM calls used.
3. Main table as in SCREEN_BRIEF.md (pooled rho, CI, worst, four scenarios, paired CI vs baseline, Brier and log-loss skill,
   verdict by section 8) plus the extra metrics, for your variants and for their combination with the baseline and with lad2_adele.
4. Any diagnostic that explains the result (for example, how well your feature predicts the allow/refuse label, reliability of a
   repeated LLM measurement).
5. Honest verdict in 3 lines: does it help, where, how sure.
Return a summary of at most 15 lines.

## Battery (important)
The laptop runs on battery tonight (68% at the start, no charger). Save CPU:
- Always call the harness with `benches=["sopbench", "tau2"]` and `schemes=["new_procedures", "new_domain"]` unless your task says
  otherwise. Random cases and banking are not needed for the pooled metric.
- Put at most 8 new columns in one harness run (the bootstrap cost grows with columns). One full run plus at most one rerun.
- Debug on one fold of one benchmark (copy the harness loop or pass a tiny scheme) before the full run.
- Never leave a Python process spinning. If something hangs for more than 20 minutes, kill it and report.
- Write REPORT.md as you go (a first version right after the first results), so the work survives if the laptop dies.
