# c_lad and f2 on SOPBench with the grader fixed (11 Nov 2026)

No model calls. The 28 agents of the baseline repo, every published episode re-scored with `fixsop.fixed_eval`
(`build_responses.py`; 1032 of 23 240 labels change: 1025 to success, 7 to failure), four data variants (`make_data.py`),
full-data IRT refit per variant, then the unchanged c_lad and f2 feature sets and Ladder model under the baseline protocol
(`run.py`, harness.run_all, new_procedures 5x5 folds, new_domain 7 folds). Statistic as in rounds 3-10: pooled
within-cell Spearman between predicted case difficulty and the variant's own IRT difficulty, bootstrap over procedures
(`score_fixed.py`, `score_fixed.txt`; reference on the old runs `score_ref.py`).

Variants: O806 / A806 = published / fixed labels, the 24 broken cases dropped. Oclean / Aclean = also dropped the 75
cases with message or credential defects and 16 refuse cases decided only by an unwritten condition (716 cases).

| | adele (baseline) | c_lad | f2 |
|---|---|---|---|
| procedure level, published labels (O806, new_proc / new_domain) | 0.22 / 0.26 | 0.44 / 0.45 | 0.43 / 0.46 |
| procedure level, fixed labels (A806) | 0.63 / 0.65 | 0.68 / 0.68 | 0.72 / 0.72 |
| case level, published labels (O806) | 0.13 / 0.11 | 0.25 / 0.22 | 0.27 / 0.22 |
| case level, fixed labels (A806) | 0.09 / 0.13 | 0.26 / 0.24 | 0.29 / 0.26 |
| procedure level, clean subset published / fixed (Oclean, Aclean; new_domain) | 0.20 / 0.61 | 0.41 / 0.65 | 0.42 / 0.65 |

With the grader fixed, ranking procedures by difficulty from the documentation alone works: rho about 0.65 to 0.72 for
every method, against 0.2 to 0.46 before. The cause is visible in the labels: simple lookup procedures
(hotel/show_available_rooms, hotel/find_booking_info, get_product_details, get_provider_details) went from 0 to 0.3
success to 0.96 to 1.00; under the published grader they looked like the hardest procedures, and every documentation
feature called them easy. At the case level nothing changes (c_lad 0.24 to 0.26, f2 0.26 to 0.29): which request inside
a procedure fails depends on the data at run time.

c_lad and f2 against the ADeLe baseline, paired: case level still +0.11 to +0.20 (f2 intervals above 0 in both schemes
on A806); procedure level the gap shrinks from +0.19 to +0.22 to +0.03 to +0.09 (intervals include 0). Once the target
is clean, the baseline's ADeLe levels already rank procedures well; the extra features mostly help inside procedures.

Limits: the target is population difficulty over 28 agents of mixed strength; for one strong agent alone failures are
too rare (2.4% on SOPBench-fixed) to rank. The O and A variants are separate runs, so the O-to-A change has no joint
interval, but the procedure-level intervals do not overlap for adele and c_lad. Exploratory, no pre-registration.

## Success-probability metrics (`success_metrics.py`, `success_metrics.txt`)

The Ladder models output P(agent succeeds) per agent and case (agent ability minus predicted difficulty). Skill is the
Brier skill against the `constant` predictor, which knows each agent's ability but treats every task as average;
`oracle` uses the true full-data difficulty (ceiling). Seed-averaged out-of-fold predictions, bootstrap over procedures.

| agent x procedure success rate, Brier skill (new_procedures / new_domain) | adele | c_lad | f2 | oracle |
|---|---|---|---|---|
| published labels (O806) | +0.01 / -0.01 | +0.21 / +0.20 | +0.18 / +0.18 | +0.47 / +0.52 |
| fixed labels (A806) | +0.21 / +0.19 | +0.24 / +0.24 | +0.27 / +0.26 | +0.54 / +0.57 |

Mean absolute error of the predicted agent x procedure success rate: f2 0.22 on fixed labels, constant 0.24, oracle
0.16. Per agent x case: Brier skill +0.05 to +0.09 for c_lad and f2 under both labels (oracle +0.34 to +0.40), AUC 0.72
to 0.75 against 0.70 to 0.72 for agent ability alone. The models almost never predict 0.9 or more (0 to 2% of agent x
procedure cells; oracle 14% at 96% real success): the ridge shrinks difficulties toward the mean, so as they stand they
cannot pick out procedures safe to hand over.

Reading: with the grader fixed, the success-rate forecast per agent and procedure improves (f2 +0.18 to +0.27, about half
of the oracle's +0.55), and even the ADeLe baseline goes from 0 to +0.20. Per case it stays small.

## Recalibration and a hand-over rule (`calibrate.py`, `calibrate2.py`, `calibrate*.txt`)

Fitted on the other folds only (other procedures, or other domains). (a) Logistic map per agent x case, agent part and
task part with separate slopes; (b) binomial logistic per agent x procedure; (c) a hand-over threshold on the raw
predicted rate, chosen on the other folds as the lowest rate whose cells there succeed at least 90%, then applied to the
held-out fold.

Recalibration does not help: mean absolute error per agent x procedure stays 0.22 (c_lad, f2) and skill against an
equally recalibrated constant is +0.10 to +0.16 (+0.33 to +0.54 for the oracle). The stretch learned on some procedures
does not carry to others.

The threshold rule, fixed labels (A806, new_procedures / new_domain): c_lad hands over 7% / 5% of agent x procedure
pairs at 0.885 / 0.875 real success, f2 6% / 2% at 0.856 / 0.784, adele 3% / 1% at 0.858 / 0.686; oracle 22% / 21% at
0.898 / 0.902. Clean subset (Aclean): c_lad 10% / 8% at 0.884 / 0.846. Published labels (O806): c_lad 2% / 0% at 0.843 /
0.43, adele 0%. So on the fixed benchmark the documentation method can pick a small set of safe pairs at close to the
target (about a third of what perfect knowledge of difficulty would pick); on the published labels it could not.
