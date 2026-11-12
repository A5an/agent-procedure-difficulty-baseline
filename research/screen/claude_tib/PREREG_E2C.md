# E2-C: second customer simulator with Claude agents (pre-registration, 12 Nov 2026)

Written before any episode of this condition. Gemini E2 (`../sim2`) stopped at a third with the 70 USD total; per
Tima's "добьём всё остальное ... через подписку" the same question is asked within Claude: the 300 existing k=2 seeded
probes run again for both Claude agents with Claude Sonnet as the customer instead of Claude Haiku. Simulator prompt,
agent prompt, bridge and scoring unchanged (`run_claude_tib.py --sim sonnet`, outputs `runs/k2_<agent>_simsonnet.jsonl`).

Outcomes as in `../sim2/PREREG.md`, per Claude agent over the 50 families: share of probes whose outcome differs between
the two customers (bootstrap over families); Spearman with the real failure rate (real tasks keep the Haiku customer)
for each customer and their mean, paired differences; the "6 of 6" families under each customer and both, with their
real success. Support for "the customer model matters": disagreement share with a lower bound above 0.10 for at least
one agent, or the "6 of 6" set changes by at least 3 families.
