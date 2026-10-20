# E4, escalation curve (13 Oct 2026)

Script `escalate.py`, results `escalate.json`. Gain in success points over escalating the same share at random, averaged over
agents (all but the strongest) and seeds. To a human = escalated task counted as solved; to the best agent = its observed result.

| scenario | x | baseline | c_lad + gap | f2 | oracle difficulty |
|---|---|---|---|---|---|
| SOPBench new procedures, human | 20% | +0.4 | +2.7 | +2.5 | +7.3 |
| SOPBench new domain, human | 20% | -0.9 | +2.3 | +2.3 | +7.3 |
| tau2 new procedures, human | 20% | -0.2 | +1.0 | +1.3 | +6.3 |
| tau2 new domain, human | 20% | -1.6 | +0.9 | +0.1 | +6.3 |
| SOPBench new procedures, best agent | 20% | -0.0 | -2.0 | -2.0 | -2.7 |
| tau2 new procedures, best agent | 20% | -0.2 | +0.5 | +0.7 | +3.2 |

f2 minus baseline at 20% to a human (points, procedure bootstrap): SOPBench new procedures +2.1 [+0.9, +2.9], new domain
+3.1 [+0.7, +4.6]; tau2 new procedures +1.5 [+0.9, +2.1], new domain +1.7 [+0.8, +2.5].

Findings
- Sending the predicted-hardest 20% to a human gives 1 to 3 points over random escalation with our models, nothing with the
  baseline. The ceiling with the true task difficulty is 6 to 7 points, so we capture about a third on SOPBench.
- Escalating the hardest tasks to the strongest agent does not help on SOPBench, even with the oracle (-2.7): the tasks that are
  hard for one agent are hard for all, so the strong agent fails them too. A router should send predicted-hard tasks to a human or
  to a different kind of executor, not to a stronger model of the same kind.
- The cost axis (price per model) comes from E8, where the cost of every arm was measured.
