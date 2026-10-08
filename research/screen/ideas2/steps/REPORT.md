# Step-level reliability from SOPBench traces (agent `steps`, 5 Oct 2026)

Saved by the main session from the agent's final message (the agent could not write files itself).
No LLM calls. Code: build_steps.py, extract_traces.py, analyze_reliability.py, steppred.py, run_harness.py, paired.py.
Results: out/eval.txt, paired.json, failure_decomposition.csv. Mapping: step_type_mapping.csv, types_map.py.

## Data
- 4,400 required steps for the 830 SOPBench cases derived from the rule tree. Step types: auth, existence, threshold,
  status, datetime, relation, availability, OR-group, action.
- Trajectories of the 28 agents: 23,235 runs, 108,449 step outcomes. 5,082 records had no stored evaluation and were
  re-scored with the benchmark's own evaluator. After that, success matches responses.jsonl in every run.
- Rule-tree truth reproduces action_should_succeed in 805 of 826 cases. 5 runs with no interaction and no response dropped.

## Reliability by step type (25 unflagged agents)
- Authentication is the most reliable step (88%). Datetime and availability are the weakest (42%), then relation (46%).
- Within a type agents differ widely, for example availability 4% to 93%.
- No step type is weak for every family. Availability is weak for claude-3.5, gemma, gemini-1.5/2.0 and o4-mini, strong for
  claude-3.7 and qwen3.5.
- Family intervals are overstated because they ignore clustering by procedure.

## Held-out rho on SOPBench (new procedures / new domain)
Only SOPBench has step structure, so there is no pooled four-scenario number.

| method | new procedures | new domain | mean, 95% CI |
|---|---|---|---|
| baseline adele | 0.120 | 0.125 | 0.123 [0.025, 0.216] |
| count only (geometric law, r_a^n) | -0.013 | 0.023 | 0.005 [-0.113, 0.127] |
| step types, shared logit | 0.160 | 0.065 | 0.112 [-0.016, 0.194] |
| step types + adele | 0.190 | 0.135 | 0.162 [0.041, 0.261] |

- Types vs count: +0.108 [-0.073, +0.265]. New procedures alone +0.173 [-0.030, +0.316], agent-specific version +0.176 [+0.011, +0.317].
- Types vs baseline: -0.010 [-0.186, +0.118]. Types added to the baseline: +0.040 [-0.037, +0.121].
- Product of reliabilities: no better than the count (+0.017 [-0.03, +0.06]), Brier skill about -0.5. A refusal needs no
  check, so success is not a plain AND of steps on must-refuse cases.
- Step count correlates 0.07 with fitted difficulty, so the geometric law fails here.
- Exploratory ablation: authentication steps alone give 0.142 / 0.138, dropping them leaves 0.038 / 0.017, shuffled type
  labels give -0.03. The model reads procedure family, not step reliability.
- Agent x type weights add nothing on held-out procedures (-0.003 [-0.06, +0.03]).
- Procedure-level rho: step types + adele 0.339 [0.100, 0.509] on new domain, baseline 0.231.

## Failure decomposition (20,745 runs, unflagged agents, descriptive)
- Must-perform failures are 70% of 7,199 runs. 76.9% of them are a skipped or late required check (61.0% acted anyway,
  15.9% gave up), 3.7% a misread after all checks were done. Together about 81%.
- Must-refuse failures are 49% of 13,546 runs. 77.7% of them act despite a failed check: 60.5% never made the check,
  17.2% made it and acted anyway.
- Weaker agents mostly skip checks. Claude-3.7, gemini-2.5-flash, gpt-4.x and qwen3.5 look and still act in 29 to 45% of
  their must-refuse failures.

## Verdict
1. "Public data records whole-task success, not step success" was too pessimistic for SOPBench. Step coverage can be
   derived (not for tau2). It measures call coverage and decision consistency, not understanding.
2. Types do not significantly beat the pure step count and do not beat the baseline.
3. Per-type reliabilities are real in sample but do not transfer to new procedures.

## Limitations
Only SOPBench, about 70 procedures. 21 rule trees do not reproduce the label, 4 cases have no tree. Step correctness is a
proxy. Ablations were run after the primary result and are exploratory.
