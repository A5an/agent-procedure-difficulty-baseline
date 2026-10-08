# Emulated pilot, feasibility of the first half (agent `emulate`, 5 Oct 2026)

Saved by the main session from the agent's final message (the agent could not write files itself).
139 LLM calls (gemini-3.8-flash on Vertex), no agent episodes.

## Summary
1. Case generation works. With the full schema and format examples in the prompt, 95% of 400 synthetic cases load and
   run. Code gives the true perform or refuse label, and 35% of cases must be performed, against about 35% in the real benchmark.
2. The key step is untested. No agents were run, so we do not know whether a cheap agent's success on synthetic cases
   ranks procedures like the real difficulty does.
3. Ceiling from public data. If synthetic cases were as good as real ones, a good cheap agent would reach procedure-level
   rho 0.5 to 0.6, against 0.23 for the baseline and 0.41 to 0.52 for policy-reading methods. Expected with synthetic cases: 0.3 to 0.5.
4. Cost of the full test about 48 USD and about 3 hours. A smoke run on the 10 tested procedures about 0.8 USD.
5. tau2 variant: skip for now (no procedure unit, no databases locally, judge errs 33%). About 10 to 18 USD later.
6. Main caveat: SOPBench's own cases are written by gpt-4.1-mini from the policy and schema, then checked by code. We repeat
   that recipe, so this is the friendliest possible test and says little about real company data.

## Validation
Procedures (10 in 7 domains): bank.pay_loan, bank.apply_credit_card, dmv.schedule_test, dmv.renew_vehicle,
healthcare.schedule_appointment, healthcare.submit_claim, hotel.book_room, library.borrow_book, online_market.return_order,
university.enroll_course. Prompt: the agent's exact policy text for the most common constraint variant, plus a database
schema made by code from real cases (v3: union of all real databases of the domain). Ground truth: gt.py uses the env's own
check functions, reproduces the stored label on all 197 real cases of these procedures and 815 of 830 overall.

Real-data bug: the 15 disagreements are mostly bank.cancel_credit_card and bank.pay_bill_with_credit_card. In
bank.cancel_credit_card all 17 real databases store credit_cards as a list where the code expects a dict, so the environment
raises. Exclude these two procedures from any SOPBench run.

v3, 400 cases, 40 per procedure:

| Check | Result |
|---|---|
| Loads into the environment | 100% |
| Loads and the goal tool runs | 95% |
| LLM's own label matches the code label | 98.2% |
| Perform share by code | 35.4% (real 37.6% in these 10 procedures, 34.7% overall) |
| Failing policy conditions vs real | Spearman 0.84 over 76 procedure-condition pairs |
| Distinct failure patterns at real sample size | 47.2 vs 65 real (ratio 0.73) |

Invalid cases are almost all online_market.return_order (ISO timestamps where the env parses a plain date). All 7 LLM label
errors are in dmv.schedule_test (or-branch misread). The LLM does not reproduce procedure-specific perform mixes (Spearman
0.27 over 10 procedures). Types-only schemas gave 54 to 75% valid, format examples 87%, domain-wide schema 95%. Synthetic
cases in SOPBench task format: synthetic_tasks/ (379 valid). Per-procedure tables: tables.md.

## Ceiling from public outcomes (ceiling.py)
Real cases resampled per procedure as a stand-in for synthetic ones. Pair-weighted Spearman inside domain against mean IRT
difficulty, 70 procedures.

| Pilot agent | All real cases | 10 cases per procedure |
|---|---|---|
| gpt-4.1-mini | 0.65 | 0.58 |
| gemini-2.5-flash | 0.54 | 0.53 |
| qwen3.5-4b | 0.59 | 0.53 |
| gemma-4-e4b | 0.58 | 0.53 |
| gemini-2.0-flash | 0.44 | 0.38 |
| gpt-4o-mini | 0.34 | 0.30 |

Leave-one-out targets give 0.56 to 0.69 for the first four. Reweighting to a fixed 35% perform share changes little. Which
agent you pick matters more than how many cases you run.

## Design of the full experiment
70 procedures (drop the two broken bank ones), 239 policy variants, one generation call per variant gives 10 cases, keep up
to 6 valid, about 1,434 cases. Agents: gemini-2.5-flash (in the public data, so synthetic vs real success of the same agent
can be compared) and gemini-2.5-flash-lite, gpt-4.1-mini optional. Temperature 0, max_tokens 4096, one attempt, scripted
user, SOPBench gold grader (works unchanged because every case of a constraint variant shares the same directed_action_graph).
Estimate per procedure: 0.35 x perform-class failure rate + 0.65 x refuse-class failure rate, compared with mean IRT
difficulty, bootstrap over procedures. Pre-registered rule: rho >= 0.40 with the interval lower bound above 0.23.
Code: Vertex route from e8_router.patch plus model names in swarm/constants.py, about 40 lines for the variant loop and 80
for the analysis. run_simulation.py takes --data_dir. About 18,000 calls, about 3 hours with 8 workers.

| Item | USD |
|---|---|
| Generation, 239 calls (measured 0.0187 each, doubles after 1 Jan 2027) | 4.4 |
| flash-lite episodes (1,434 at 0.0051) | 7.4 |
| gemini-2.5-flash episodes (1,434 at 0.0199) | 28.5 |
| Main run with 20% margin | about 48 (33 to 72) |
| Minimal run (modal variant only, flash-lite) | about 13 |
| Strong run (adds gemini-3.8-flash, 2 attempts) | about 233 |

Episode costs are estimates from logged token counts with 500 assumed thinking tokens per call. Only generation cost is
measured. Smoke run S0 first: 100 cases of the 10 procedures on flash-lite, about 0.8 USD, and check that refuse cases are
not scored correct by empty answers.

## Risks
1. Optimistic test because SOPBench's generator and ours follow the same recipe.
2. The perform/refuse mix is a design choice, a company would supply its own traffic mix.
3. Synthetic cases cover 73% of real failure patterns at equal size and mislabel some or-branch logic.
4. Real company schemas may be messier.
5. 68 procedures in 7 domains, wide intervals.
6. gemini-3.8-flash prices double on 1 Jan 2027, gemini-2.5 may be retired, gpt-4.1-mini price unverified.
7. Public gemini-2.5-flash outcomes came from AI Studio with max_tokens 512, our run would use 4096. Control run on about 30
   real cases (about 0.6 USD).
