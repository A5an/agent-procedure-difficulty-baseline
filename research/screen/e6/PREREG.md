# E6, second pass: pass^k with the best model and recalibration (pre-registration, 13 Oct 2026)

First pass (`../targets/REPORT.md`, 4 Oct) used the baseline P only: on tau2 independent attempts were as good or better, on
banking beta-binomial won because the baseline P was too low; recalibrating P first was left open.

This pass, tau2 new procedures and new domain (the only scenarios with repeated attempts and our trained models), out-of-fold P of
the baseline (`adele`) and of the best model (`f2_all_fc`) from `../ideas10/fc/out`. Variants: raw P or recalibrated P (one logit
shift fitted by maximum likelihood on the other folds' cells of the same seed); independent attempts (p^k) or beta-binomial (phi
fitted on the other folds with the same P). Metric: Brier of pass^k against the share of all-success k-subsets, k = 1 to 4.
Paired bootstrap over tasks (2,000). Main comparison: f2 recalibrated beta-binomial against the plan's default, baseline raw
independent. Expectation: recalibration matters more than the dispersion term; f2 lowers Brier because it ranks better.
Banking has no trained f2, so it is not in this pass.
