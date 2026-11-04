# Can a model tell when documentation probes will be wrong? (pre-registration, 4 Nov 2026)

No Gemini; Claude Sonnet via claude -p. For each of the 50 TauIndianBank families Sonnet reads the one-line purpose,
the policy and the tool list (never a benchmark task) and rates 0 to 10 how much a correct handling of this request type
depends on details that the purpose and the policy leave unspecified, with a list of the gaps. Target: the error of
documentation probes, |doc probe failure rate - real failure rate|, per family, averaged over g38 and flash, and the
gain from seeded probes (doc error minus seeded error). Hypothesis: Spearman between the gap score and the doc-probe
error above 0 (bootstrap lower bound above 0).
