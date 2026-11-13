# E2-D: real tasks with the second customer model (pre-registration, 13 Nov 2026)

Written before any episode of this condition. Plan from the 13 Nov instructions (Asanali), run on Tima's Claude
subscription through `claude -p`, $0.

Why. In E2-C the k=2 probes rank families much worse when their customer is Claude Sonnet instead of Claude Haiku
(Spearman with the real failure rate 0.65/0.73 -> 0.36/0.40 for the Haiku/Sonnet agent), but the real tasks were run only
with the Haiku customer. E2-D runs the real tasks with the Sonnet customer, so the effect of a customer mismatch can be
separated from "probes with the Sonnet customer are noisier".

Design. The same 50 families and the same 6 real tasks per family as E1-C (`seedk/selection_seedk.json`), agents Claude
Haiku and Claude Sonnet, customer Claude Sonnet: 600 episodes, `run_claude_tib.py --set real --sim sonnet`, outputs
`runs/real_<agent>_simsonnet.jsonl`. Agent prompt, simulator prompt, bridge and scoring unchanged; only the customer
model changes. Failed episodes (harness errors) are retried by the runner; episodes still missing are reported.

Primary outcome. Per agent, a 2x2 table of Spearman between the family failure rate on k=2 probes and on real tasks, for
(probe customer Haiku or Sonnet) x (real-task customer Haiku or Sonnet), bootstrap over families (2000, seed 20261113),
and paired differences "matched customer minus mismatched": SS - SH (probes Sonnet) and HH - HS (probes Haiku), and the
mean of the two matched cells minus the mean of the two mismatched cells.

Hypothesis (fixed now). With matched Sonnet/Sonnet customers the Spearman is at least 0.55 for both agents (point
estimate). If so, the E2-C drop is explained by the customer mismatch; if not, by noisier probes with the Sonnet customer.

Secondary. Share of real tasks whose outcome differs between the two customers; real success rate of each agent with
each customer; the "all 6 probes passed" families for the four combinations with their real success; real success of the
families that pass "6 of 6" under both probe customers, computed on the real tasks with each customer separately.

Analysis code `analyze_e2d.py`, hashed into `FREEZE_PRE.txt` before the first episode. Anything added after seeing the
results is labelled exploratory.
