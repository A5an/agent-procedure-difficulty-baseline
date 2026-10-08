# Council member `cog` (cognitive psychology of procedure following), 6 Oct 2026

Saved by the main session. No code, sources from memory (unverified).
Main point: policy-only features are procedure-level and capped on SOPBench case level; skips need an easy-to-skip
condition AND a case where that condition is the one that fails, so cross condition-level properties with case evidence.
F1 condition packing density (conditions per sentence, and/or chains, list vs prose); F2 guard distance and ordering
(distance between condition and guarded action, order mismatch text vs code, conditions stated after the action, scatter
across sources); F3 simultaneous load (max conjunction width, nesting, entities, open conditions at the write);
F4 interference among similar conditions (near-duplicate thresholds, fan on the same field); F5 polarity and exceptions
(negations, unless, prohibitions, failure means do nothing); F6 user assertion and action-readiness (share of target
arguments already in the message, conditions the user presupposes, fields mentioned, pressure); F7 verification chain depth,
unobservable conditions, tool confusability; F8 case-weighted skip hazard = sum over conditions of p_fail(c | message) x
skip_propensity(c) (existing written-condition skip model).
Top 3: F8 (+0.02 to +0.04 guess), F6 (+0.01 to +0.03), F3+F2 (procedure level). Cheap tests on the 1,365 synthetic cases.
