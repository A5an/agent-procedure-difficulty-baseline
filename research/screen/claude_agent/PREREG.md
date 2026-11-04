# Exp 8: a Claude agent on TauIndianBank, and probes from synthetic past conversations (pre-registration, 4 Nov 2026)

No Gemini: every model runs through `claude -p` on the subscription (`claude_proxy.py`). Written before any number
except the 2-episode pilot (both passed, about 5 minutes per episode). Tima: no paid simulations; Claude runs are fine.

Questions. (1) Do probes predict a non-Gemini agent's success on TauIndianBank as they did for Gemini? (2) The open
problem: without real past conversations, can a strong model write "past conversations" itself, and do probes seeded
with them predict as well as probes seeded with real ones?

Setup. The 20 families of P0-2, 6 real tasks each. Agent Claude Sonnet 5.5, user simulator Claude Haiku 5.5, harness,
policy and grading of P0-2. Probe writer Claude Opus 5.5 (another model than the agent), the P0-2 prompt, candidate
filter and gold-action execution check, 6 probes per family for each of three kinds:
- DOC: documentation only (P0-2 prompt);
- SEED: plus the two real seed tasks of Exp 3 as past conversations;
- PSEUDO: plus two past conversations that Opus first writes itself from the family's one-line purpose, the policy,
  the tools and candidate customers (customer script + correct tool calls, checked by executing the calls), never
  seeing a benchmark task.

Measures per probe kind, over families: Spearman rho with the real success rate, Brier skill against the base rate
(leave one family out), calibration (ECE, 5 bins), and the delegation rule (predicted success >= 0.9: share of cases
delegated and real success among them).

Hypotheses: H1 SEED rho >= 0.4 with the lower bound above 0. H2 PSEUDO has higher Brier skill than DOC (paired
bootstrap over families, interval of the difference above 0). Descriptive: PSEUDO vs SEED; Sonnet vs the Gemini agents.
Order: real episodes; probe generation; probe episodes. If subscription limits stop the runs, the analysis uses the
families completed for every set and says so.
