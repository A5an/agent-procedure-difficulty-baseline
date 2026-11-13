# E3 reconnaissance: where more procedures could come from (12 Nov 2026, no runs)

Handoff E3 asks for about 100 procedures with at least 6 real tasks each, a local resettable environment and scoring
without an LLM judge, so that a 0.90 target can be qualified. Counted from local copies; "procedure" for the tau2
domains = tasks with the same set of gold action names (the benchmarks have no procedure field).

| source | procedures | with >= 6 tasks | >= 8 (6 + 2 seeds) | notes |
|---|---|---|---|---|
| TauIndianBank | 50 | 50 | 50 | all 50 already used; nothing left |
| SOPBench-fixed | 70 | 37 | 28 | deterministic grader, local; but one customer message, no conversation, so a probe is a generated case, not a dialogue |
| tau2 airline | 21 | 3 | - | 50 tasks |
| tau2 retail | 50 | 4 | - | 114 tasks |
| tau2 banking_knowledge | 20 | 5 | - | 97 tasks; v1.0.1 fixed its grading (release notes) |
| tau2 telecom | 1194 | 1 | - | 2285 tasks built from issue x subtask x persona combinations, dual control (the user acts on a phone) |

Reading: no single conversational benchmark at hand adds the ~50 procedures needed. The realistic options are
(a) SOPBench-fixed, 28 procedures with 6 evaluation + 2 seed cases, as a non-conversational second benchmark with probes
written as new cases from two seed cases; (b) the 12 multi-task procedures of airline, retail and banking_knowledge
together, which are too few on their own. Neither gets to ~100 procedures; E3 as specified would need a new benchmark
or synthetic families. Nothing was run.
