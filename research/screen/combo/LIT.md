# Literature check: probe-trained episode monitors plus difficulty-aware routing

Date: 2026-11-03. Sources: arXiv API (about 14 queries), WebSearch (2 queries), arXiv abstract pages (18 fetched, a bit over the 12 budget). All entries below were read on the paper's own arXiv abstract page ("from abstract only"). Venue is "preprint" because the arXiv page did not show a venue; I did not check for later publication. Long AND-queries on the arXiv API often returned nothing, so coverage of Q2 and Q3 is thin.

## Q1. Failure detectors and monitors for agents

1. Dubey 2026, "Real-Time Detection and Repair of LLM Agent Failures", preprint, https://arxiv.org/abs/2608.02464 (from abstract only).
   Step-telemetry monitors (one-class echo-state network with CUSUM) trained only on healthy runs detect 0.71 of failures at 5% false alarms (AUROC 0.872) on 2,823 episodes. It reports that monitors do not transfer across deployments: AUROC 0.527 cold versus 0.885 recalibrated per deployment. Supports our finding that a monitor must be fitted to the deployment's own data.
   differs from us: trained on healthy real runs, not on pre-deployment simulated test episodes, and no procedure-level difficulty estimate or routing.

2. EsfandyariDoulabi, Arkoh, Tadesse 2026, "Disentangling Task Difficulty from Run-Level Failure in Agent Failure Prediction", preprint, https://arxiv.org/abs/2610.05572 (from abstract only).
   Shows that pooled AUROC (0.85 to 0.94 in prior work) is mostly cross-task: a difficulty oracle reaches up to 0.945 pooled and is at chance within task. Early run-level discrimination is weak (within-task AUROC about 0.50 to 0.55 for early monitors), and task-level allocation beats abort-only strategies under fixed budgets. Closest framing to our "overall vs within procedure" AUCs (0.79 to 0.87 vs 0.64 to 0.79).
   differs from us: analysis paper on repeated attempts of the same task, no monitor trained on pre-deployment probes, and no three-way router with a commit-point escalation.

3. Vasudev, Russak, Bikel 2026, "Accurate Failure Prediction in Agents Does Not Imply Effective Failure Prevention", preprint, https://arxiv.org/abs/2602.03338 (from abstract only).
   An LLM critic with offline AUROC 0.94 can cut agent performance by up to 26 points; they propose a pre-deployment test using a pilot of 50 tasks to estimate whether intervening will help or hurt. Relevant to the idea that a small pre-deployment run decides a deployment-time intervention policy.
   differs from us: the pilot decides one global "intervene or not" switch, it does not train an episode-level monitor from the pilot trajectories.

4. Baidya, Liang, Guo 2026, "When Evidence is Sparse: Weakly Supervised Early Failure Alerting in Dialogs and LLM-Agent Trajectories", preprint, https://arxiv.org/abs/2606.05414 (from abstract only).
   Lightweight external monitor trained from trajectory-level outcomes (attention MIL), with a preference-conditioned stopping policy that trades accuracy against earliness, on five benchmarks. Cost of decisions is measured, but as serving cost.
   differs from us: trained on logged outcomes of the target benchmarks, not on LLM-written probes, no human/agent routing costs.

5. Wang, Lyu, He 2026, "Fail-Fast, Restart-Smart: Early Failure Prediction and Restart for SWE Agentic Tasks", preprint, https://arxiv.org/abs/2608.03222 (from abstract only).
   A 0.6B monitor trained on one policy's trajectories transfers to three other policies on SWE-bench Verified and saves 14.6% to 20.4% of tokens at 5% false-positive rate. Shows cross-policy transfer of a trajectory monitor.
   differs from us: transfer is across agents on the same benchmark, not from simulated probes to real episodes of the same procedure.

6. Zhang, Zhu, Shi 2026, "AgentForesight: Online Auditing for Early Failure Prediction in Multi-Agent Systems", preprint, https://arxiv.org/abs/2605.08715 (from abstract only).
   7B online auditor trained with RL on AFTraj-2K (coding, math, agentic), with failures labeled by LLM judge consensus; tested on an external Who&When benchmark. Prefix-level early alarm.
   differs from us: curated corpus of logged trajectories, no procedure-specific pre-deployment probes.

7. Panthi, Abdelfattah 2026, "Outcome Monitors: Recovery Affordances for Silent Tool Failures", preprint, https://arxiv.org/abs/2608.19303 (from abstract only).
   Monitors check outcome contracts mined from task-disjoint traces or public schemas, and improve completion on tau-bench retail by 14.0 and 12.0 points on two tiers. A tool-result monitor built from other traces and applied at runtime.
   differs from us: rules about tool outputs, not a learned episode-level failure classifier, and the traces are task-disjoint logs rather than probes of the same procedure.

8. Zhang et al. 2026, "MirrorGuard: Toward Secure Computer-Use Agents via Simulation-to-Real Reasoning Correction", preprint, https://arxiv.org/abs/2601.12822 (from abstract only).
   Trains a guard on trajectories generated in a text-based simulated environment and applies it to real computer-use agents, cutting UI-TARS unsafe rate from 66.5% to 13.0%. The clearest "simulated trajectories train a deployment-time guard" case I found.
   differs from us: security (prompt injection) domain, guard rewrites reasoning, the simulation is generic and not a per-procedure test suite, and there is no difficulty estimate.

9. Ruan et al. 2023, "Identifying the Risks of LM Agents with an LM-Emulated Sandbox" (ToolEmu), preprint on arXiv page, https://arxiv.org/abs/2309.15817 (from abstract only).
   LM-emulated tools plus an LM safety evaluator find agent failures in 144 test cases; 68.8% of found failures judged valid real-world failures. Establishes simulated test runs as a failure-rate estimate for an agent.
   differs from us: the emulator results are used for evaluation only, not to train a monitor.

10. Xiang et al. 2024, "GuardAgent", preprint on arXiv page, https://arxiv.org/abs/2406.09187 (from abstract only). Also Chen et al. 2025, "ShieldAgent", https://arxiv.org/abs/2503.22738 (title seen in search listing only, abstract not read, so not summarized).
    GuardAgent checks agent actions against safety requests by generating guardrail code, 98% and 83% guardrail accuracy on two new benchmarks. Representative of pre-action tool-call guards.
    differs from us: rule or policy-based guard written from specifications, not a classifier learned from the procedure's probe episodes.

## Q2. Pre-deployment difficulty plus per-episode deferral with costs

11. DiSorbo, Ju 2026, "Act or Escalate? Evaluating Escalation Behavior in Automation with Language Models", preprint, https://arxiv.org/abs/2604.08588 (from abstract only).
    Frames act-or-escalate as expected-cost comparison with cost ratios, across five non-agentic domains. LLM self-estimates are miscalibrated; SFT on cost-aware reasoning generalizes to held-out domains.
    differs from us: single prediction decisions, not tool-using multi-step agents, and no pre-deployment difficulty estimate from test runs.

12. Shaikh 2026, "Knowing When to Ask for Help: Bayesian Self-Escalation in Hierarchical LLM Agents", preprint, https://arxiv.org/abs/2608.24087 (from abstract only).
    Mid-generation escalation as optimal stopping over a competence posterior learned from labeled trajectories, with regret bounds; real validation is a small code model cascade (MBPP, 257 tasks).
    differs from us: escalation to a stronger model, not a human, no customer-service procedures, and labels come from logged trajectories.

13. Turan 2026, "Oversight Has a Capacity: Calibrating Agent Guards to a Subjective, Fatiguing Human", preprint, https://arxiv.org/abs/2606.08919 (from abstract only).
    Selective classification under asymmetric cost for agent action gating with a finite-capacity human reviewer; names learning-to-defer prior art (FALCON, DeCCaF) as cited background (I did not verify those two).
    differs from us: 125 hand-labeled actions, no pre-deployment difficulty prior, no three-way router.

14. Feng et al. 2026, "CORA: Conformal Risk-Controlled Agents for Safeguarded Mobile GUI Automation", preprint, https://arxiv.org/abs/2604.09155 (from abstract only).
    Guardian model estimates per-action risk, conformal risk control sets an execute/abstain boundary at a user risk budget, rejected actions are routed on.
    differs from us: GUI harm prevention, calibration is on labeled step-level data, no procedure-level difficulty prior and no explicit human-cost trade-off.

## Q3. Same simulated runs as difficulty estimate and monitor training data

15. Williams et al. 2026, "Predicting LLM Safety Before Release by Simulating Deployment", preprint, https://arxiv.org/abs/2607.07184 (from abstract only).
    Resamples candidate model replies on de-identified production prefixes to estimate misbehavior prevalence before release, with registered predictions; flags tool-resampling realism as the main challenge. A pre-deployment rate estimate from simulated runs, close to our procedure-level failure rate.
    differs from us: prevalence estimation only, no monitor trained on the simulated runs and no routing.

16. Seshadri et al. 2026, "Lost in Simulation: LLM-Simulated Users are Unreliable Proxies for Human Users in Agentic Evaluations", preprint, https://arxiv.org/abs/2601.17087 (from abstract only).
    On tau-bench retail, agent success varies up to 9 points across user simulators, and simulated users underestimate success on hard tasks and overestimate it on moderately difficult ones. A caution for any probe-based estimate.
    differs from us: an evaluation-validity study, no monitor or routing.

## What looks new

In our search (sources: arXiv API and WebSearch, about 16 queries over agent failure prediction, runtime guards, tool-call guards, learning to defer, conformal abstention, pre-deployment simulation; date 2026-11-03) we did not find:

- A paper that uses LLM-written probe episodes of one business procedure both as a procedure-level difficulty estimate and as training data for an episode-level failure monitor applied to real episodes of that procedure (Q3). Closest are Williams et al. 2026 (simulation for prevalence estimate, no monitor), MirrorGuard (simulated trajectories train a guard, no difficulty estimate) and Vasudev et al. 2026 (a 50-task pilot decides intervention, no trained monitor).
- A cost-based router that combines a procedure-level pre-deployment failure rate with an episode-level monitor score and offers human, agent, and agent-with-escalation-at-commit-point as options (Q2). Closest are DiSorbo and Ju 2026 (act-or-escalate cost framework, non-agentic), Shaikh 2026 (optimal stopping escalation, code cascade), and EsfandyariDoulabi et al. 2026 (task-level allocation versus abort-only, no cost-based human option).
- A comparison of a monitor trained on the procedure's own probes versus one trained on real logs of other procedures. Closest evidence of the cross-deployment problem is Dubey 2026 (monitor AUROC 0.527 cold vs 0.885 recalibrated).

Related finding to keep in the paper: EsfandyariDoulabi et al. 2026 argue that pooled AUROC mostly reflects task difficulty, so reporting our within-procedure AUCs (0.64 to 0.79) next to the pooled ones (0.79 to 0.87) is the right protocol. Limits of this check: only abstracts were read, venues were not checked, FALCON, DeCCaF, ShieldAgent, tau-bench error-analysis papers and process reward models for agents (ToolPRMBench 2601.12294, AgentProcessBench 2603.14465, seen in search listings only) were not read.
