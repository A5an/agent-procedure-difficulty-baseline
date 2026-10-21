# E3 with step types (13 Oct 2026)

Scripts `run_e3.py` (models), `within.py` (within-agent metrics). Results `run/eval.txt`, `run/eval.json`, `run/within.json`.

Pooled rho over the four scenarios (protocol evaluator): baseline 0.131; agent x step type (`steps_mirt`) 0.104, difference
[-0.074, -0.005]; lambda fixed 10: 0.107 [-0.078, +0.001]; control ADeLe + step counts as difficulty features 0.151 [-0.043, +0.070].

Held-out log-loss, `steps_mirt` minus baseline (seed 0, procedure bootstrap): SOPBench new procedures -0.001 [-0.002, +0.001],
new domain +0.001 [-0.001, +0.003]; tau2 new procedures -0.011 [-0.020, +0.002], new domain -0.007 [-0.015, -0.004].
Within-agent AUC: SOPBench -0.001, -0.009; tau2 +0.065 [-0.023, +0.115], -0.049 [-0.109, -0.020].

Verdict by the pre-registered rule (log-loss interval below zero in at least 3 of 4 scenarios): no. Only tau2 new domain passes,
and there within-agent AUC drops. Together with the procedure-type screening of 4 Oct: we find no measurable agent strength
that depends on the type of step, beyond one ability per agent. Step counts help a little as difficulty features shared by all
agents (control 0.151), not as agent-specific strengths.
