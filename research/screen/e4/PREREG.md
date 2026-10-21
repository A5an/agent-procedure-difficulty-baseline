# E4, escalation curve (pre-registration, 13 Oct 2026)

Plan item E4 left open the curve "when to hand a task to a stronger model or a human". Offline, on the out-of-fold predictions of
the protocol (SOPBench and tau2, new procedures and new domain; `../ideas10/fc/out`). Each agent except the strongest one takes
the new tasks; the share x (10, 20, 30%) with the lowest predicted P goes to a human (counted as solved) or to the strongest agent
(its observed result). Metric: success gain over escalating the same share at random, in points, averaged over agents and seeds.
Methods: baseline (`adele`), c_lad + gap (`c1_all`), best model (`f2_all_fc`), oracle IRT (`oracle`, ceiling). Interval for
f2 minus baseline at x = 20% to a human: bootstrap over procedures (tau2: tasks), 300 draws.
Expectation: f2 above the baseline on tau2, small on SOPBench where in-procedure difficulty is hidden in the database.
