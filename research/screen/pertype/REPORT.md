# pertype: one ability per agent is not enough? (SOPBench, tau2 for routing only)

No LLM calls. All code in this folder. Date 4 Oct 2026.

## 0. Headline

1. The raw run files separate two different failure modes that one ability hides. On cases where the right outcome is to carry out the request, gpt-5 makes no calls out of order in 96% of runs (`dirgraph_satisfied`) but ends with the action done in only 32% (`action_called_correctly`); 1D IRT expects 55% success there. Weak agents are the opposite: the action is executed in 70 to 85% of carry-out cases while the checks are in order in only 13 to 55%. Across the 28 agents the two rates are negatively correlated (Spearman -0.60, p<0.001).
2. Procedure-level types (action family, rule-tree type, domain), known before running, do not give a held-out gain in difficulty ranking: every per-type model that keeps the baseline difficulty has the same within-domain rho as the baseline (0.12 to 0.13 on both SOPBench scenarios, paired differences within [-0.011, 0.017]). Under the improvement rule none is better.
3. Where the per-agent differences do show up is inside an agent and in routing, but only partly through "types": agent x domain on held-out procedures (seen domains) and agent x rule-tree type on held-out domains. Agent x action family adds nothing.
4. Offline routing: choosing the agent by a per-type model beats choosing by one ability per agent (SOPBench new procedures 0.766 vs 0.735, new domain 0.728 vs 0.659 for the best variant), but it stays below the per-task oracle (0.952), and the gain over the "best agent per domain chosen on training data" is not significant except for the domain-interaction models.
5. Sub-outcomes: the baseline predicts "process in order" and "action right" about as weakly as it predicts overall success (rho 0.10 to 0.19). Agent rankings on the two targets are close but not the same (Spearman 0.90, Kendall 0.74); task difficulties on the two targets are almost unrelated (Spearman 0.28).

Limits: tau2 has no rule tree or raw check flags in the release, so per-type models and sub-outcomes were run on SOPBench only. The pooled four-scenario rho of the protocol therefore cannot be formed; I report the two SOPBench scenarios. Verdicts below are provisional for that reason.

## 1. Step 1: per-check outcome tables (`s01_tables.py`, `flags.csv.gz`)

Every record of the 28 agent configurations x 7 domains was re-scored with SOPBench's own evaluator (same call as the repository's `build_sopbench.py`), which yields the flags for all runs, including the ones without a saved evaluation.

- 28 agents (ids identical to `responses.jsonl` subject ids), 830 cases (case ids `domain/procedure/index`, all found in `tasks.csv`), 23,240 agent x case records.
- 5 records have no run (empty interactions): left missing, as in `responses.jsonl`.
- 21,296 records have a saved evaluation; 1,939 runs have none and are covered by re-scoring (1,944 by the repository's count, 5 of them are the no-run cells).
- Where both exist, re-scored flags equal the saved ones: dirgraph 21,296/21,296, action_called_correctly 21,296/21,296, success 21,296/21,296, constraint_not_violated 21,295/21,296. `success` equals `responses.jsonl` on all 23,235 runs.

What the flags mean (read from `env/evaluator.py`): `dirgraph_satisfied` is False when any call was made before its required prior checks; an agent that calls nothing satisfies it vacuously. So it measures "no call without the required checks", not "all checks done". `action_called_correctly` is `action_should_succeed == target action succeeded`.
Per-agent table with carry-out and refuse rates, 1D IRT expectation and ability ranks on three targets: `per_agent_subcheck.csv`. Examples (carry-out cases, mean over 28 agents: checks in order 0.54, action executed 0.74):

| agent | checks in order | action executed | success | 1D IRT expects | refuse-case success |
|---|---|---|---|---|---|
| gpt-5 | 0.96 | 0.32 | 0.29 | 0.55 | 0.96 |
| gpt-5-mini | 0.78 | 0.60 | 0.39 | 0.55 | 0.91 |
| gpt-4.1 | 0.83 | 0.76 | 0.61 | 0.48 | 0.69 |
| claude-3.5-sonnet act-only | 0.17 | 0.84 | 0.06 | 0.06 | 0.20 |
| gemini-1.5-pro | 0.33 | 0.85 | 0.20 | 0.16 | 0.36 |

The allow/refuse flag (`should_succeed`) was used only in these descriptive tables, never as a feature.

## 2. Step 2: per-type ability, held-out test (`s02_types.py`, `pertype_models.py`, `s03_run.py`, `s04_eval.py`, `s05_within.py`)

Protocol: `harness.py`, same folds as the baseline (new_procedures: 5 folds x 5 seeds; new_domain: 7 folds), fold IRT from the repository cache, scoring with the repository evaluator, plus within-agent AUC and log-loss with a paired bootstrap over procedures (seed 0, 300 draws). SOPBench only.

Types (fixed before any model was run, `types.csv`):
- (a) `fam`: action family of the requested procedure, hand-assigned from the procedure name (create 317 cases, modify 364, terminate 64, pay 52, read 33).
- (b) `tree`: shape of the rule tree "from the rule tree" (`constraints_original`, which restates the SOP the agent sees and is identical for the allow and refuse variant of a case): number of checks (1-2, 3-4, 5+) x (conjunction only | contains or/chain) = 6 types. Note it is per case, not per procedure (a procedure's cases use 1 to 5 tree types).
- (c) `dom`: domain (7). Unseen in the new_domain scenario, so there every domain term is zero.
- Fixed loadings for the confirmatory MIRT (3 dims): `q_deep` (5+ checks), `q_branch` (has or/chain), `q_modify` (family modify or terminate). I first defined two other columns that were 95% ones, saw this from the type counts (no outcomes) and replaced them before running any model.

Models (all use the fold's IRT theta; shrinkage strength chosen by 3-fold CV over training procedures, log-loss):
- (i) baseline `adele`.
- (ii) `share_*`: P(agent a, type k) = (S_ak + kappa s_a)/(N_ak + kappa), s_a the agent's overall training share, kappa in {1..10000}. Stand-alone (does not use beta_hat). `share_none` is the same with no types (kappa irrelevant): a calibrated per-agent constant, the right null for this family.
- (iii) `logit_fam/tree/dom/all`: logit P = theta_a - beta_hat_t + sum_l W[a,l] x_tl, ridge on W (lambda in {3..30000} by inner CV), fitted on training tasks with the fold-IRT beta as offset, so W is what the 1D model leaves unexplained for agent a on type l. `_l10`, `_l30`: lambda fixed (sensitivity, not tuned).
- (iii') `main_all` and `inter_all`: stand-alone logistic, logit = alpha_a + shared type effects [+ agent x type], no ADeLe. `main_all` is the matched no-interaction control, so `inter_all` minus `main_all` isolates the agent-specific part.
- (iv) `mirt_q3`: confirmatory MIRT, logit = theta_a + sum_d q_td phi_ad - beta_hat_t with the fixed Q above, ridge on phi_ad (`_l10` fixed lambda).
- Controls: `typebeta` (ridge from one-hot types to difficulty), `adele_onehot` (ADeLe + one-hot types).

Deviation from the paper-style MIRT: loadings are fixed and abilities are deviations from the 1D theta (ridge), not free abilities.

### Results, SOPBench (rho = within-domain Spearman with beta, mean over seeds; CI from the repository bootstrap; diff = paired 95% CI against the baseline; Brier and log-loss skill are against the harness `constant`, AUC_w = within-agent AUC)

| method | new proc rho | diff to baseline | new dom rho | diff to baseline | Brier skill (proc / dom) | log-loss skill (proc / dom) | AUC_w (proc / dom) |
|---|---|---|---|---|---|---|---|
| baseline adele | 0.120 [0.009, 0.229] | | 0.125 [0.024, 0.238] | | 0.014 / 0.005 | 0.011 / 0.003 | 0.580 / 0.549 |
| (ii) share_none (null) | 0.070 | [-0.203, 0.069] | 0.074 | [-0.209, 0.062] | 0.035 / 0.050 | 0.042 / 0.060 | 0.500 / 0.500 |
| (ii) share_fam | 0.026 | [-0.292, 0.081] | -0.076 | [-0.328, -0.045] | 0.034 / 0.049 | 0.041 / 0.060 | 0.491 / 0.488 |
| (ii) share_tree | 0.011 | [-0.254, 0.012] | -0.002 | [-0.253, -0.022] | 0.043 / 0.054 | 0.049 / 0.064 | 0.561 / 0.536 |
| (ii) share_dom | 0.070 | [-0.203, 0.069] | 0.074 (=null) | [-0.209, 0.062] | 0.105 / 0.050 | 0.098 / 0.060 | 0.637 / 0.500 |
| (ii) share_famtree | 0.047 | [-0.206, 0.085] | 0.049 | [-0.182, 0.044] | 0.033 / 0.050 | 0.041 / 0.060 | 0.543 / 0.529 |
| (iii) logit_fam | 0.120 | [-0.002, 0.003] | 0.125 | [-0.002, 0.004] | 0.014 / 0.005 | 0.011 / 0.003 | 0.580 / 0.549 |
| (iii) logit_tree | 0.121 | [-0.000, 0.017] | 0.125 | [-0.005, 0.005] | 0.014 / 0.004 | 0.011 / 0.003 | 0.580 / 0.549 |
| (iii) logit_dom | 0.121 | [0.000, 0.000] | 0.125 | [0.000, 0.000] | 0.040 / 0.005 | 0.033 / 0.003 | 0.612 / 0.549 |
| (iii) logit_all | 0.119 | [-0.004, 0.007] | 0.125 | [-0.011, 0.008] | 0.028 / -0.002 | 0.023 / -0.004 | 0.602 / 0.539 |
| (iii) logit_all, lambda 30 | 0.120 | [-0.000, 0.013] | 0.128 | [-0.003, 0.009] | 0.031 / 0.001 | 0.027 / -0.000 | 0.605 / 0.542 |
| (iv) mirt_q3 | 0.121 | [-0.004, 0.006] | 0.124 | [-0.004, 0.005] | 0.014 / 0.005 | 0.011 / 0.003 | 0.580 / 0.549 |
| (iv) mirt_q3, lambda 10 | 0.120 | [-0.004, 0.007] | 0.123 | [-0.010, 0.006] | 0.010 / -0.004 | 0.005 / -0.006 | 0.578 / 0.543 |
| (iii') main_all (control) | 0.031 | [-0.254, 0.034] | 0.041 | [-0.203, 0.009] | 0.082 / 0.058 | 0.079 / 0.066 | 0.620 / 0.530 |
| (iii') inter_all | 0.040 | [-0.251, 0.034] | 0.048 | [-0.201, 0.011] | 0.103 / 0.052 | 0.097 / 0.062 | 0.643 / 0.522 |
| control typebeta | 0.011 | [-0.271, 0.013] | 0.010 | [-0.244, 0.002] | 0.049 / 0.009 | 0.037 / 0.006 | 0.616 / 0.522 |
| control adele_onehot | 0.100 | [-0.112, 0.012] | 0.092 | [-0.116, 0.049] | 0.050 / 0.020 | 0.039 / 0.013 | 0.621 / 0.543 |
| oracle | | | | | 0.396 / 0.412 | 0.355 / 0.371 | 0.852 / 0.846 |

Pooled four-scenario rho: not available (tau2 not run for these models). Mean of the two SOPBench scenarios: baseline 0.123; logit_all 0.122; mirt_q3 0.123; logit_tree 0.123; logit_all with lambda 30 0.124 (differences under 0.002, intervals include zero); stand-alone models 0.03 to 0.08.

Reading:
- Verdict under section 8 of the protocol (SOPBench scenarios only): (iii) and (iv) = not better (rho unchanged; log-loss skill in new_domain is below the baseline for logit_all and mirt_q3_l10, so rule 4 fails). (ii) and (iii') = not better, and worse on new domain for share_fam and share_tree (interval below zero). Their rho is not comparable in kind: they give every task of a type the same value for all agents, so they cannot rank tasks inside a type.
- The log-loss skills of the stand-alone models are large because the harness `constant` (sigmoid(theta - mean beta)) is poorly calibrated, not because of types: `share_none` alone already has +0.042 / +0.060. Compare with `share_none`, not with `constant`.
- Within-agent, new procedures (paired, seed 0):
  - `inter_all` minus `main_all`: AUC_w +0.023 [0.008, 0.036], log-loss skill against the agent's own share +0.021 [0.012, 0.030]. This is the agent-specific part, net of the shared type effects.
  - `logit_all` minus `adele`: AUC_w +0.022 (CI [-0.002, 0.027]), log-loss skill +0.015 (CI [0.003, 0.011]). `logit_dom` +0.032 AUC_w [0.010, 0.046]. Gains are small in absolute terms (AUC_w 0.58 to 0.64 against 0.85 for the oracle).
  - New domain: no log-loss gain from any interaction model (all within +-0.02).
- Which type carries it (`s03b_ablate.py`, `s05b_within_abl.py`, `within_abl.json`, single type system, stand-alone, interaction minus matched main-effect control):
  - domain: new procedures log-loss skill +0.021 [0.011, 0.033] (AUC_w difference includes 0, [-0.015, 0.019]); new domain 0 by construction.
  - rule-tree type: AUC_w +0.025 [0.011, 0.039] on new procedures and +0.025 [0.010, 0.041] on new domain, log-loss unchanged ([-0.004, 0.003] and [-0.005, 0.003]). Agents differ a little in how they order tasks by tree type, with no calibration gain.
  - action family: 0 in both scenarios (new procedures [0.000, 0.000], new domain [-0.002, 0.014]).
  - family x tree: no gain.
- The sizes of agent x type effects that the penalty keeps are small. For (iii) with family or tree types alone and for the 3-dim MIRT the cross-validated penalty leaves output equal to the baseline to three decimals. This is not evidence that agents are alike. The gpt-5 gap in section 1 is between allow and refuse cases of the same procedure, and the three type systems do not see that flag (allowed share ranges only 0.21 to 0.52 across tree types, 0.33 to 0.42 across families), so procedure-level types cannot capture it.

## 3. Step 3: sub-outcomes as targets (`s07_sub.py`, `sub/`, `sub_eval.json`, `sub_abilities.csv`)

Baseline pipeline unchanged (fold 1PL IRT with the same split structure, ridge on the 18 ADeLe levels), with response matrices built from the flags: `checks` = `dirgraph_satisfied` (no call without its required checks; 0 or 1 per case, 23,235 runs), `action` = `action_called_correctly`. Own full-data IRT per target (seed 0) as the rank target. Seeds: 5 for new_procedures, 1 for new_domain, as in the protocol. Fold IRT and result files are in `sub/`.

| target | scenario | rho within domain [95% CI] | pAcc | Brier skill | log-loss skill | AUC_w |
|---|---|---|---|---|---|---|
| success (reference, repo) | new proc | 0.120 [0.009, 0.229] | | 0.014 | 0.011 | 0.580 |
| success (reference, repo) | new dom | 0.125 [0.024, 0.238] | | 0.005 | 0.003 | 0.549 |
| checks in order | new proc | 0.101 [-0.105, 0.276] | 0.534 | 0.032 | 0.022 | 0.584 |
| checks in order | new dom | 0.144 [-0.045, 0.297] | 0.549 | 0.006 | -0.005 | 0.553 |
| action right | new proc | 0.141 [-0.008, 0.252] | 0.549 | 0.014 | -0.010 | 0.573 |
| action right | new dom | 0.191 [0.098, 0.276] | 0.567 | 0.048 | 0.037 | 0.561 |
| (oracle for the target) | | | | | | AUC_w 0.82 to 0.86 |

Both are predicted about as weakly as success; "action right" in a new domain is the one interval clearly above zero (0.191), from a different target than the one the baseline was chosen on, so I would not read it as an improvement.

Agent rankings (full-data IRT abilities on 28 agents; Spearman / Kendall):
- checks vs action: 0.902 / 0.741
- checks vs success: 0.979 / 0.905
- action vs success: 0.946 / 0.815
- Largest rank moves between the two targets (rank by checks minus rank by action): gpt-5 1st on checks, 11th on action; gpt-4.1 2nd vs 9th; gemini-2.0-flash-thinking 10th vs 3rd; gemma-4-e4b 12th vs 7th.
- Task difficulty on the two targets: Spearman 0.28 (checks vs action), 0.68 (checks vs success), 0.49 (action vs success). So what makes a case hard for the process differs from what makes the final action fail, even though the agent rankings are close.

## 4. Step 4: offline routing (`s06_routing.py`, `routing.json`)

For each held-out task the agent with the highest predicted P is chosen; realised success is that agent's outcome on the task (tau2: share of its attempts). Averages over seeds, bootstrap over procedures (1,000 draws, seed fixed). Under one ability per agent (i) every task gets the same ranking of agents, so (i) = always the agent with the highest fold theta. Hindsight single best = best agent on all tasks, including the test ones (0.753 on SOPBench, as stated); best agent per domain is chosen on training tasks of the fold (not defined for a held-out domain).

SOPBench, realised success share [95% CI]:

| strategy | new procedures | new domain |
|---|---|---|
| (i) one ability per agent (baseline) | 0.735 [0.671, 0.797] | 0.659 [0.612, 0.711] |
| (iii) logit_all (the declared per-type model) | 0.766 [0.710, 0.816], diff to (i) +0.031 [0.001, 0.058] | 0.695 [0.629, 0.767], +0.036 [-0.011, 0.082] |
| (iii) logit_dom | 0.799 [0.749, 0.847], +0.064 [0.031, 0.099] | = (i) |
| (iii') inter_all | 0.776 [0.727, 0.829], +0.042 [-0.009, 0.086] | 0.728 [0.662, 0.799], +0.069 [0.026, 0.119] |
| (ii) share_dom | 0.790 [0.740, 0.839], +0.055 [0.019, 0.092] | = (i) |
| (ii) share_tree | 0.746 [0.685, 0.808], +0.011 [-0.016, 0.041] | 0.700 [0.641, 0.768], +0.041 [0.007, 0.084] |
| (iv) mirt_q3 | = (i) | = (i) |
| single best agent on training tasks | 0.740 [0.674, 0.801] | 0.659 (same agent as (i)) |
| best agent per domain, chosen on training | 0.759 [0.711, 0.807] | not defined |
| single best agent overall (hindsight, test tasks included) | 0.753 [0.680, 0.819] | 0.753 [0.677, 0.830] |
| per-task oracle (some agent succeeds) | 0.952 [0.923, 0.973] | 0.952 [0.922, 0.975] |
| random agent | 0.465 | 0.465 |

Paired differences that matter:
- New procedures, against the best agent per domain chosen on training: `logit_dom` +0.040 [0.013, 0.075]; `share_dom` +0.031 [0.007, 0.065]; `logit_all` +0.007 [-0.025, 0.037]; `inter_all` +0.018 [-0.001, 0.038]. Against the hindsight single best: `logit_dom` +0.046 [0.018, 0.078]; `logit_all` +0.013 [-0.013, 0.039].
- New domain, against hindsight single best: (i) -0.094 [-0.143, -0.045]; `inter_all` -0.025 [-0.065, 0.007]; `logit_all` -0.058 [-0.123, -0.008]. The per-type models close most of the gap to the hindsight best, not the gap to the oracle (0.952).
- The models that gain (domain and rule-tree interactions) are the ones that showed within-agent gains in section 2; family and the 3-dim MIRT choose the same agent as (i).
- "Best per-type model": I declared `logit_all` (the specified (iii)) and `inter_all` as the headline before looking at routing; the other rows are variants, and picking the best of ~15 on test tasks would be optimistic. Candidates like `logit_dom` should be read as hypotheses to confirm on a new benchmark.
- Agents here include several configurations of the same model, so some of the gain is choosing between siblings.

tau2 (rows for (i) only; per-type models not run because tau2 has no rule tree in the release and its tasks have no procedure names to define families):

| strategy | new procedures | new domain |
|---|---|---|
| (i) one ability per agent | 0.894 [0.823, 0.930] | 0.894 [0.829, 0.931] |
| single best on training tasks | 0.894 (same agent) | 0.853 [0.762, 0.903], diff to (i) -0.041 [-0.081, -0.016] |
| best agent per domain, chosen on training | 0.881 [0.823, 0.915], diff -0.013 [-0.031, 0.009] | not defined |
| single best overall (hindsight) | 0.894 | 0.894 |
| per-task oracle | 0.995 | 0.995 |
| random agent | 0.759 | 0.759 |

tau2 oracle is the best per-task success rate over up to 4 attempts, so it is optimistic. With 15 agents and a 0.894 single best agent, there is little room (0.10) and no sign that per-domain choice helps.

## 5. What could not be run
- Per-type models on tau2 and tau-Knowledge banking: no rule tree and no per-check flags in the releases used here. Hence no pooled four-scenario rho and no verdict on rule items 2 to 5 in full.
- Held-out benchmark confirmation (rule 5) not done; nothing here is frozen for a confirmatory run.
- Type (a) was assigned by hand from procedure names; a different grouping could change the family result. Tree types are per case, not per procedure.
- Within-agent AUC CI uses 300 bootstrap draws on seed 0 (the protocol uses 1,000 for rho only).

## 6. Files
- Scripts: `s01_tables.py`, `s02_types.py`, `pertype_models.py`, `s03_run.py`, `s03b_ablate.py`, `s04_eval.py`, `s05_within.py`, `s05b_within_abl.py`, `s06_routing.py`, `s07_sub.py`, `s08_tables.py` in `research/screen/pertype/`.
- Data and results: `flags.csv.gz`, `types.csv`, `per_agent_subcheck.csv`, `run/eval.json` (harness evaluation), `within.json`, `within_abl.json`, `routing.json`, `sub_eval.json`, `sub_abilities.csv`, `sub/` (targets, fold IRT, obs files).
- Run: `PYTHONHASHSEED=0 PYTHONPATH=<project>/prototype/screen <repo venv python> sNN_*.py`.
