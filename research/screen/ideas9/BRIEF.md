# Brief for round 9 (6 Oct 2026): run the three follow-ups of the round-8 swarm

The user approved three runs, including one local MLX job on the laptop GPU. Read first, in this order:
1. prototype/screen/ideas3/BRIEF.md (rules, information levels, harness, metrics_extra, cpu_lock, Vertex, no leakage)
2. prototype/screen/ideas8/BRIEF.md (round-8 swarm, the "varies by case" requirement, already-tried list)
3. prototype/screen/ideas8/textcount/ (PREREG.md, run.py, score.py, REPORT.md): the template for a protocol run.
   run.py builds c_lad exactly (t0_clad reproduced 0.258); score.py gives paired bootstrap CIs vs adele and vs c_lad.
4. The round-8 report of the agent whose idea you run (named in your task).

## Numbers to compare against (pooled within-domain case-level rho over sop-proc, sop-dom, tau2-proc, tau2-dom)
baseline adele 0.131 (.120 .125 .128 .150); c_lad 0.258 (.237 .212 .298 .285); procedure level SOPBench new domain:
adele 0.231, c_lad 0.486. Round-8 textcount: c_lad + regex counts 0.263 (+0.005 n.s.).
Item difficulty `b` in repo data/<bench>/irt/1d_1pl/items.csv: HIGHER b = HARDER (one round-8 agent got this wrong).

## Rules for this round
- Write PREREG.md in your folder BEFORE the first harness run: 2 to 4 variants, one of them "c_lad + your family"
  built exactly like ideas8/textcount/run.py builds t0_clad, and the primary hypothesis. One harness run, no reruns with
  changed features after seeing results (a crash rerun is fine). Report every variant.
- Harness: benches sopbench and tau2, schemes new_procedures and new_domain. Score with a copy of
  ideas8/textcount/score.py (compare vs adele and vs t0_clad). Also report SOPBench procedure level.
- If your features can be computed for tau-Knowledge banking (prototype/data/external/tau2-banking, repo data/tauk_banking),
  add a fresh-benchmark check: within-domain Spearman of your feature or predicted difficulty on banking, and if cheap, the
  harness on tauk_banking with the random scheme the repo uses for it. Banking was never used to choose anything.
- Wrap harness runs and any heavy CPU work in `with cpu_lock("<your name>")` (prototype/screen/ideas/cpulock.py).
  Never hold the lock around LLM calls or GPU inference. Never kill processes you did not start (no pkill).
- Python for the harness: repo venv <repo>/.venv/bin/python
  with PYTHONHASHSEED=0 OMP_NUM_THREADS=4 nice -n 10.
- Vertex (only the agent whose task says so): before any Python that calls it run
  `unset GEMINI_API_KEY; export LLM_BACKEND=vertex GOOGLE_CLOUD_PROJECT=$(gcloud config get-value project 2>/dev/null)`,
  model gemini-3.8-flash, temperature 0, max 6 parallel calls, cache every response in your folder (jsonl keyed by prompt
  hash). Never print or write keys or the project id.
- HANDBOOK is a blind test set: never open prototype/data/external/handbook. Do not modify the public repo.
- Work only in prototype/screen/ideas9/<your_name>/. Writing REPORT.md may be refused, so return the FULL report as text
  in your final message: idea and level, variants as pre-registered, cost, main table (pooled rho, four scenarios, paired
  CI vs adele and vs c_lad, procedure level), banking check, diagnostics, 3-line verdict, limitations, paths.
