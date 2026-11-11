# Diagnostic test: what information makes success predictable (plan, written 11 Nov 2026 before any run)

Copy of the design of Table 1 in Agent-as-a-Router (arXiv:2606.22902): one LLM, only the information in its prompt
changes. There the LLM picks a model; here it gives P(success) of an agent on a case, because our decision is
agent vs human.

Data: SOPBench. Inside each procedure with >= 2 cases, half of the cases (seed 0) are "history", the rest are test
(at most 6 test cases per procedure). Four agents: gpt-5 fc, gemini-2.5-flash fc, claude-3-7-sonnet fc, gpt-4o-mini fc.
Predictor: Claude Haiku through `claude -p` (subscription; Gemini billing is off). One call = the test cases of one
procedure (up to 5), answer = probability for each case and agent.

Prompt variants:
- V0 Vanilla: case statements only.
- V1 +Category: adds domain, procedure name and the names of the other procedures of the domain.
- V2 +Stats of similar procedures: adds history success counts of each agent on the OTHER procedures of the domain.
- V3 +Stats of this procedure: V2 plus the history counts of the procedure itself.
References without LLM: BaseRate (agent's history rate), DomainRate (other procedures of the domain), ProcRate
(history of the procedure, shrunk to BaseRate with weight 2), Oracle.

Metrics on test cases, mean over the four agents: AUC, Brier skill against BaseRate, success among the half of cases
with the highest prediction. Bootstrap over procedures.

Expected if the paper's picture holds: V0 = V1 (labels add nothing), a jump at V2/V3, V3 close to ProcRate.
