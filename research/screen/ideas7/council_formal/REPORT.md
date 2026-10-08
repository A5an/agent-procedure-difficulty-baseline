# Council member `formal` (formal methods, decision tables, process-model metrics), 6 Oct 2026

Saved by the main session. Read the data, no experiments, sources from memory (unverified).
F1 partial evaluation of the compiled program on the case params with forking on each internal_check (verdict determined
by params, number of unresolved record checks, logit_perform from per-check-name pass rates estimated on training splits);
F2 semantic decision-table metrics (rules, predicates, distinct refuse reasons as prime implicants, order-dependent rows);
F3 data-flow metrics (def-use chain depth between tool calls, derived quantities, comparisons between outputs);
F4 process-model complexity (Cardoso control-flow complexity, Mendling connector mismatch, negative polarity, disjunctions);
F5 path asymmetry (paths to perform vs refuse, expected position of the first failing check, text vs code order tau);
F6 tau2 tool-to-ID dependency graph (hand-mapped IDs required and produced per tool, lookup depth per write tool, ID gap
args, aggregated over plan tools); F7 tool-schema complexity (args, nesting, enums, format constraints, cross-argument
constraint words); F8 boundary margin of numeric comparisons between case params and constants.
Top 3: F1 (+0.03 to +0.08 guess on SOPBench; same idea as agent msg_facts), F6 (+0.02 to +0.05 on tau2), F2+F5 (+0.01 to +0.03).
