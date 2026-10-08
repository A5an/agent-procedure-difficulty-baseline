# Brief for round 6 (6 Oct 2026): ideas that push the protocol rho toward 0.3 to 0.5

Read first, in this order: prototype/screen/ideas3/BRIEF.md (rules, information levels, harness, metrics_extra, cpu_lock,
Vertex), prototype/screen/ideas3/combo/REPORT.md and run.py (current best model c_lad), prototype/screen/ideas4/ablation/
REPORT.md and run.py (how to add a family to the c_lad ladder), prototype/screen/ideas5/smart_eval/REPORT.md (new data).

## Goal
The user wants new methods as useful as c_lad that raise the main metric, the pooled within-domain case-level rho over the
four new-process scenarios (sop-proc, sop-dom, tau2-proc, tau2-dom), from c_lad's 0.258 toward 0.3 to 0.5, without running
agents on the case being predicted. Every method states its information level (L0 documentation only, L1 a few cheap LLM
calls per case, L2 reading the case record by code, E emulation: agents run only on synthetic cases generated from
documentation). Report L2 and E separately from L0/L1.

## Where the headroom is (read before designing)
- c_lad by scenario: .237 .212 .298 .285. Baseline adele .120 .125 .128 .150.
- SOPBench case level: most within-domain difficulty is whether the case must be performed or refused, decided by the
  customer's record. Oracle with only that label 0.52, procedure means only 0.18 to 0.23. A zero-shot LLM guesses the label
  from the message with AUC 0.56 to 0.64. Any real gain on SOPBench must recover more of that label without the record, or
  use L2.
- tau2: the gold action count and writes reach 0.40 to 0.47 (oracle, uses gold). LLM plans reach 0.29 (new procedures) and
  0.19 to 0.29 (new domain). Customer-scenario features (ideas4/user_scenario) gave tau2 new domain 0.37. Telecom has only 3
  procedures and its difficulty is a hidden fault: do not trust telecom-only gains.
- New data you may use: smart-agent outcomes (Gemini 3.8 Flash and Sonnet 5.5) on 1,365 synthetic SOPBench cases generated
  from documentation, with code-derived perform/refuse labels (ideas5/smart_gemini/synth_results.csv,
  ideas5/smart_sonnet/synth_results.csv, cases in ideas2/emulate_run/synth_all/, generator in ideas2/emulate/ and
  emulate_run/gen_all.py). Outcomes of smart agents on REAL cases (ideas5/*/real_results.csv) are evaluation data only,
  never features or training targets for the protocol.
- Exclude nothing from the protocol run, but report separately without bank/cancel_credit_card and
  bank/pay_bill_with_credit_card (their target tool is missing for agents).

## Rules of this round
- Fix 2 to 3 variants before the first full run, one of them "c_lad + your family" built like ablation/run.py.
- Shared harness on sopbench and tau2, schemes new_procedures and new_domain, metrics_extra, paired CI vs adele and vs c_lad
  (ablation/score.py style bootstrap over procedures if the harness has no paired CI for your column).
- LLM: gemini-3.8-flash on Vertex. Before any Python that calls it: `unset GEMINI_API_KEY; export LLM_BACKEND=vertex
  GOOGLE_CLOUD_PROJECT=$(gcloud config get-value project 2>/dev/null)`. Never print or write keys or the project id. Max 6
  parallel calls, cache everything. Claude Sonnet is available through headless `claude -p --model sonnet` on the user's
  subscription (see ideas2/cc_agent/run.py for the flags; for plain text tasks use --allowedTools "" and
  --disallowedTools for all tools), max 4 in parallel.
- Use the repo venv python; if it lacks a package, ideas2/emulate_run/venv/bin/python.
- Finish within about 90 minutes. Folder prototype/screen/ideas6/<your_name>/. Return the FULL report as text in your final
  message (REPORT.md writes may be refused for subagents).
