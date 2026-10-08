# How wrong is a cheap LLM judge that reads a pilot agent's transcript

Scripts: `prep.py` (sample and prompts), `run_judge.py` (cached calls), `analyze.py` (all tables). Data: `sample.json` (900 prompts with grader outcome), `cache_low.jsonl` and `cache_medium.jsonl` (every raw response). No key or project id is stored.

## 1. Setup (fixed before the full run)
- Judge: Gemini 3.8 Flash on Vertex, temperature 0, thinking "low", max 2,000 output tokens. The judge sees the policy, the compact transcript and the question from the brief. It never sees the grader verdict or any gold field. Single tool results are cut at 1,500 characters.
- SOPBench: 2 pilots, gpt-4.1-mini (fc) and gemini-2.0-flash-001 (fc). Both are cheap, have no empty replies (empty_cases.csv share 0), and gpt-4.1-mini is the third most frequent adaptive pick in pilot/results.json (14%). The top adaptive picks gemma-4-e4b (19%, 4% empty replies) and claude-3.7-sonnet (15%, not cheap) were not used. 300 cases per agent, seed 20261005, 150 grader passes and 150 grader fails per agent, spread over the 7 domains round robin inside each outcome. Grader label is the re-scored `success` in the repo's responses.jsonl (the label the pilot analysis uses). Policy text is the repo statement (action policy and customer message) with the tool list removed and one added sentence saying that the initial database is visible only through tool results.
- tau2: first trial of every task, 2 cheaper submissions, gpt-4.1-mini and gemini-3-flash, 150 per submission. Policy: airline, retail, telecom main policy, and for telecom also the tech support manual. gpt-4.1-mini sample is 75 pass and 75 fail. gemini-3-flash has only 53 first-trial fails, so its sample is 97 pass and 53 fail. The pilot analysis picks gpt-4.1-mini on tau2 in 45% of cases.
- Calls: 898 for the main run (two prompts were byte-identical across agents and shared a cache entry) and 150 for the second judge. Total 1,048 of 1,200. All 900 verdicts parsed.
- Second judge: same prompt, thinking "medium", 150 random SOPBench transcripts (seed 777, 151 rows because of the shared prompt).
- Bootstrap: 2,000 resamples of cases within each row. Because the sample is 50/50 on grader outcome, accuracy and kappa depend on that mix. FPR and FNR do not, so use them for the e comparison. FPR = judge says correct while grader says fail. FNR = judge says wrong while grader says pass.

## 2. Main result (thinking low)
| group | n | accuracy | FPR | FNR | kappa |
|---|---|---|---|---|---|
| all | 900 | 0.821 [0.796, 0.846] | 0.248 [0.208, 0.288] | 0.117 [0.087, 0.147] | 0.639 [0.589, 0.688] |
| SOPBench | 600 | 0.897 [0.872, 0.920] | 0.143 [0.104, 0.185] | 0.063 [0.036, 0.094] | 0.793 [0.741, 0.840] |
| tau2 | 300 | 0.670 [0.617, 0.720] | 0.492 [0.405, 0.576] | 0.209 [0.149, 0.267] | 0.307 [0.199, 0.410] |
| SOPBench gemini-2.0-flash | 300 | 0.917 [0.883, 0.943] | 0.100 [0.057, 0.148] | 0.067 [0.031, 0.110] | 0.833 [0.767, 0.887] |
| SOPBench gpt-4.1-mini | 300 | 0.877 [0.840, 0.913] | 0.187 [0.125, 0.252] | 0.060 [0.027, 0.101] | 0.753 [0.681, 0.824] |
| tau2 gemini-3-flash | 150 | 0.633 [0.553, 0.713] | 0.830 [0.712, 0.925] | 0.113 [0.054, 0.179] | 0.066 [-0.069, 0.213] |
| tau2 gpt-4.1-mini | 150 | 0.707 [0.633, 0.780] | 0.253 [0.154, 0.351] | 0.333 [0.228, 0.441] | 0.413 [0.267, 0.549] |

Symmetric e that matches the observed error:
- By accuracy on this sample: SOPBench 0.10, tau2 0.33, pooled 0.18.
- By balanced accuracy (average of FPR and FNR, independent of the outcome mix): SOPBench 0.103, tau2 0.351, pooled 0.182.
- Class-wise rates (pooled): the judge accepts 25% of the grader's failures (FPR 0.25) and rejects 12% of its passes (FNR 0.12). So the error is not symmetric. It leans toward calling a run correct. On tau2 the lean is strong (FPR 0.49, and 0.83 for gemini-3-flash). For tau2 gpt-4.1-mini the lean reverses (FNR 0.33 above FPR 0.25).

## 3. SOPBench by case type (analysis only, from tasks.csv should_succeed)
| case type | n | accuracy | FPR | FNR | grader pass rate |
|---|---|---|---|---|---|
| perform (should_succeed = 1) | 190 | 0.826 | 0.238 | 0.047 | 0.34 |
| refuse (should_succeed = 0) | 410 | 0.929 | 0.075 | 0.068 | 0.58 |

The judge is more accurate on refuse cases. On perform cases it is too lenient: it accepts runs the grader fails (24% of grader fails). Many of these are runs that did the action without calling the explicit check tools the SOPBench graph expects, or that stalled (see examples).

## 4. Does confidence separate right from wrong verdicts
No. The judge says confidence 0.95 to 1.0 almost always: mean 0.992 on verdicts that agree with the grader and 0.985 on the 161 that disagree. AUC of confidence for predicting agreement with the grader is 0.56 (SOPBench 0.57, tau2 0.56). Confidence cannot be used to flag unreliable verdicts or to weight the pilot outcome. As a ranking of grader pass, the verdict signed by confidence (P = conf if correct else 1 - conf) has AUC 0.84, which comes from the verdict itself.

## 5. Second judge, thinking medium (150 SOPBench transcripts)
| judge | accuracy | FPR | FNR | kappa |
|---|---|---|---|---|
| low | 0.887 [0.834, 0.934] | 0.161 | 0.047 | 0.775 |
| medium | 0.887 [0.834, 0.934] | 0.149 | 0.062 | 0.774 |

Same accuracy, paired difference 0.000 [-0.026, +0.026], 97% of verdicts identical. More reasoning does not lower e here. Medium was not run on tau2, where the error is larger, so the tau2 answer is unknown.

## 6. Ten disagreements (random draw, seed 1, 5 per benchmark)
Reasons are the judge's own words. The "who is right" note is mine and only given where I looked at the transcript or the pattern is clear.
| id | case | pilot | grader | judge | judge reason (shortened) | who is right |
|---|---|---|---|---|---|---|
| 108 | online_market/use_coupon/42 | gpt-4.1-mini | fail | correct | all required conditions verified before use_coupon | not inspected |
| 457 | bank/set_safety_box/14 | gemini-2.0-flash | pass | wrong | did not check balance and credit score, stalled on authentication | not inspected |
| 310 | bank/exchange_foreign_currency/2 | gemini-2.0-flash | fail | correct | all conditions satisfied, tool executed | outcome is plausibly fine (500 USD is under 1,000), the grader fails it because the agent never called the check tools |
| 137 | university/declare_minor/0 | gpt-4.1-mini | fail | correct | verified all prerequisites before declaring the minor | not inspected |
| 9 | bank/pay_bill_with_credit_card/1 | gpt-4.1-mini | fail | correct | did not act without verifying and exited when credentials were refused | grader likely right, the agent tried login with a guessed password and the judge missed it |
| retail/t0074 | tau2 id 855 | gemini-3-flash | pass | wrong | made several tool calls in one turn | judge applies a policy rule the grader does not score, so a grader/policy gap, not a judge hallucination |
| telecom/t0273 | tau2 id 795 | gemini-3-flash | fail | correct | refueled with consent, guided troubleshooting | the grader checks the final device state, which the judge cannot see in the transcript |
| retail/t0163 | tau2 id 705 | gpt-4.1-mini | pass | wrong | several tool calls in a single turn | same as 855 |
| retail/t0162 | tau2 id 792 | gemini-3-flash | fail | correct | verified identity, confirmed before modifying | grader checks exact end state and the second order, judge reads the plan as sufficient |
| retail/t0106 | tau2 id 790 | gemini-3-flash | fail | correct | authenticated, checked availability, confirmed, modified | same as 792 |
Pattern: in tau2, 7 of 36 judge-wrong-on-pass verdicts cite the one-tool-call-at-a-time rule that the reward does not score. Most tau2 false accepts are runs that look procedurally clean in the transcript but end in a wrong final state or miss an action the task required. Neither side is simply wrong. The grader is the label we have, but for the one-tool-call rule it measures something narrower than the policy.

## 7. Which row of the pilot analysis this is, and the implied rho
Pilot analysis (pilot/REPORT.md section 11, results.json `judge_error`) simulates a symmetric flip with probability e on every pilot outcome, one adaptive pilot, documentation prior. Pooled rho over the four new-process scenarios:

| e | 0 | 0.05 | 0.1 | 0.2 | 0.3 |
|---|---|---|---|---|---|
| rho, adaptive k = 1, doc prior | 0.453 | 0.413 | 0.379 | 0.305 | 0.228 |

- SOPBench (e about 0.10): row 0.1, rho sop-proc 0.31 and sop-dom 0.38.
- tau2 (e about 0.33 to 0.35 by balanced accuracy, 0.29 for the gpt-4.1-mini pilot alone, which is the pilot the adaptive rule picks 45% of the time): slightly beyond the 0.3 row, which gives tau2-proc 0.23 and tau2-dom 0.26. Extrapolating the slope of about -0.07 to -0.10 per 0.1 of e gives roughly 0.20 and 0.23 at e = 0.35.
- Pooled single number: e = 0.18 interpolates to about 0.32. Taking each benchmark at its own e and averaging the four scenarios gives about 0.28.
- Gain over documentation alone (k = 0 is 0.137): about +0.14 to +0.18, still positive. The pilot analysis found the gain stays positive up to e = 0.3, so a judge of this quality keeps most of the benefit on SOPBench and about half or less on tau2.

Caveats on this mapping:
1. The simulation flips outcomes at random, symmetric, independent of the case. The real judge has FPR above FNR and its errors depend on the case (they cluster on runs that skip checks or end in a wrong hidden state, which are also the hard cases). A correlated, lenient error is more harmful for ranking difficulty than random flips with the same rate, so the table gives an upper bound for the real rho. I did not re-run the simulation with the real class-wise rates, the interpolation is my reading of the e-grid.
2. The sample is balanced 50/50 on the grader outcome, so accuracy is not the accuracy in deployment. FPR and FNR carry over, and for a population with different pass share the overall error is FPR x fail share + FNR x pass share.
3. The judge was tested on 2 SOPBench pilots and 2 tau2 pilots. The adaptive rule also picks weak SOPBench models (gemma 4 e4b, qwen 2b) with many empty replies, which were not tested.
4. Policy rules not scored by the grader (one tool call per turn) count as judge errors here. If a deployer cares about those rules, the judge is arguably better than this table says.

## 8. Verdict
- A cheap judge with no gold information reaches e about 0.10 on SOPBench (kappa 0.79) and about 0.33 on tau2 (kappa 0.31, almost zero for gemini-3-flash). Pooled e about 0.18.
- It leans lenient (FPR 0.25, FNR 0.12), its confidence carries no information (AUC 0.56), and more thinking does not help on SOPBench.
- Implied pooled rho for one adaptive pilot is about 0.28 to 0.32 as an upper bound, against 0.453 with a perfect grader and 0.137 with no pilot. The pilot idea survives a real judge on SOPBench and loses about half on tau2.

# Part 2. Measured numbers on one fixed pilot (gpt-4.1-mini)
Scripts: `prep_full.py` (all prompts), `run_judge.py` with `SAMPLE=full.json`, `measured.py` (pipeline), `jan.py` (parsing helpers). Outputs: `full.json`, `measured.json`, `cache_low.jsonl` (now 1,556 entries). Calls in this step: 658 new (530 SOPBench, 128 tau2). Grand total with part 1: 1,706 of the budget (part 1 used 1,048, this step 658, within the 1,100 allowed for this step).

## (a) Judge on every case of the pilot
All 830 SOPBench cases and all 278 first-trial tau2 cases of gpt-4.1-mini, same prompt, thinking low. Natural class mix (grader pass rate 0.447 on SOPBench, 0.547 on tau2).
| benchmark | n | accuracy | FPR (false accept) | FNR (false reject) |
|---|---|---|---|---|
| SOPBench | 830 | 0.899 | 0.146 | 0.046 |
| tau2 | 278 | 0.673 | 0.310 | 0.342 |
Symmetric e matching the accuracy at the natural mix: 0.10 on SOPBench, 0.33 on tau2. These match the part 1 estimates from the balanced subsample (SOPBench FPR 0.19 and FNR 0.06 for this pilot, tau2 0.25 and 0.33), so the subsample was a fair estimate. The false-accept and false-reject rates used below are these full-set values.

## (b) Judge sees the end state: not possible, skipped
The tau2 trajectory files hold the messages, the termination reason and `reward_info`. There is no final environment or database state in them. `reward_info` holds only the grader's checks (the gold actions, the db match flag), which the judge must not see. The tool results that show the state are already in the transcript. Rebuilding the end state would need the tau2 environment code and databases, which are not in `tau2-domains` (only task files and policies). No calls were spent.

## (c) Measured pipeline: design P1, k = 1, pilot fixed to gpt-4.1-mini, documentation prior
Same folds and fold IRT as the pilot analysis (SOPBench and tau2, new procedures seeds 0 to 4, new domain), same target (failure share of the non-pilot agents, the pilot's own cell removed from the target for every variant including k = 0). Posterior is the Gaussian doc prior times the pilot likelihood. Variants:
- grader: the pilot's real grader outcome with the plain Rasch likelihood.
- judge_plain: the judge verdict treated as the outcome with the plain likelihood.
- judge_err: the judge verdict with the likelihood that includes the measured class-wise rates of (a): P(verdict = correct | pass) = 1 - FNR, P(verdict = correct | fail) = FPR, per benchmark.
- k0: documentation prior only.
Pooled rho is the mean over the four new-process scenarios. Bootstrap: 1,000 draws over procedures on seed 0, the same draw for all variants, so differences are paired. A few cases without a judge verdict or without a pilot cell get no update.

| variant | pooled rho [95% CI] | SOPBench | tau2 |
|---|---|---|---|
| k0 (doc prior only) | 0.131 [0.029, 0.259] | 0.115 | 0.146 |
| grader outcome | 0.405 [0.234, 0.502] | 0.376 | 0.435 |
| judge, plain likelihood | 0.272 [0.141, 0.390] | 0.285 | 0.260 |
| judge, error-aware likelihood | 0.265 [0.152, 0.398] | 0.285 | 0.245 |

Scenario values: grader sop-proc 0.372, sop-dom 0.379, tau2-proc 0.401, tau2-dom 0.469. judge_plain 0.287, 0.282, 0.263, 0.256. judge_err 0.287, 0.282, 0.254, 0.235. k0 0.116, 0.114, 0.140, 0.153.

Paired differences (pooled over four scenarios, 95% CI):
| comparison | all four | SOPBench | tau2 |
|---|---|---|---|
| judge_plain minus grader | -0.133 [-0.165, -0.053] | -0.091 [-0.172, -0.017] | -0.176 [-0.200, -0.035] |
| judge_err minus grader | -0.141 [-0.171, -0.044] | -0.091 [-0.172, -0.017] | -0.190 [-0.217, -0.025] |
| judge_plain minus k0 | +0.142 [+0.016, +0.260] | +0.170 [+0.050, +0.325] | +0.113 [-0.084, +0.279] |
| judge_err minus k0 | +0.134 [+0.033, +0.250] | +0.170 [+0.050, +0.325] | +0.099 [-0.054, +0.249] |
| grader minus k0 | +0.275 [+0.102, +0.386] | +0.261 [+0.081, +0.438] | +0.289 [+0.027, +0.433] |
| judge_err minus judge_plain | -0.007 [-0.021, +0.021] | 0.000 | -0.015 [-0.042, +0.043] |

## Reading
- The measured judge pipeline gives pooled rho 0.27 (0.265 to 0.272), between the documentation prior alone (0.131) and the grader pipeline (0.405). It keeps about half of the grader's gain (0.14 of 0.28). The gain over no pilot is significant pooled and on SOPBench, and not significant on tau2 alone (CIs include 0).
- This agrees with the interpolation in part 1 (0.28 to 0.32). The interpolation was about 0.01 to 0.05 too high, as expected from the case-correlated lenient errors.
- The grader pipeline with the fixed gpt-4.1-mini pilot (0.405) is below the adaptive-pilot value in pilot/REPORT.md (0.453), because the adaptive rule picks the most informative agent per case.
- Modelling the measured error rates in the likelihood does not help (-0.007 [-0.021, +0.021]). On SOPBench the two variants are identical because one binary verdict from a fixed pilot shifts all cases of a fold by one of two amounts and the ranking inside a domain is unchanged. The same reasoning applies to the pilot analysis finding about ignoring e.
- The pooled judge-minus-grader loss is larger on tau2 (-0.18) than on SOPBench (-0.09), matching e of 0.33 against 0.10.
- Caveats. The class-wise rates are estimated on the same cases that are then scored (two numbers per benchmark, a mild optimism, and no effect on rank because of the point above). Single fixed pilot, one judge prompt, tau2 end-state judge not run. The grader label itself is imperfect for tau2 (it scores the end state and checks, and ignores some policy rules), so judge-minus-grader is the loss against that label, not against true correctness.

## Incident note
My first run of `measured.py` imported the module name `analyze`, which resolved to `ideas/pilot/analyze.py` and re-wrote `pilot/results.json`, `tables.txt` and `pilot_rho_vs_k.png` with the original numbers only (the `judge_error`, frequency and `cascade` keys were dropped). I restored them by re-running `judge_analyze.py` and `cascade.py` from that folder (deterministic, the judge_error e = 0.1 value 0.379 matches the earlier one). My helpers are now called `jan.py`. The pilot folder otherwise has no edits from me.
