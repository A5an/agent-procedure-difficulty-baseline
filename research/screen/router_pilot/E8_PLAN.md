# E8 plan: step-by-step routing pilot on SOPBench

Date: 4 Oct 2026. Nothing was run. No agent episodes, no paid calls. All numbers below come from reading code, counting characters in the existing SOPBench logs and reading price pages.

Files in this folder: `e8_router.patch` (sketch, applies cleanly and compiles on a scratch copy, the original SOPBench copy was not touched), `sample150.csv` (the 150 cases), `tokens.py`, `cost.py`, `predict.py`, `make_sample.py` (the scripts behind every number), `episode_stats.json`, `cost_arms.json`.

## 0. Short version for the supervisor

1. The router is cheap to build. About 60 added lines in 5 files, no change to the environment or the grader.
2. The pilot costs about 29 USD for one attempt of three arms (range 18 to 52 USD) and about 87 USD for three attempts (range 53 to 155 USD). About 3,100 model calls and 18.6M input tokens for one attempt. About 8 hours if run one call at a time, about 1.2 hours with the 7 domains in parallel.
3. Before spending that, one finding from the logs changes what the pilot can show. The "gpt-5 rarely performs the final action" result is probably a truncation artifact, not a weakness at the final-action step. See section 1.5. A 5 USD diagnostic (E8-0) should run first.
4. With 150 cases the pilot can only detect a difference of roughly 7 to 12 percentage points between arms. The only pair of Vertex models with logged baselines (gemini-2.5-flash and gemini-2.5-pro) is predicted to gain about 1 point, so that pair is a test of the machinery and a negative control, not a real test of the router idea.

## 1. Feasibility

### 1.1 Where the model is called

Everything runs through one loop. Paths are inside `prototype/data/external/SOPBench-d2622008/`.

- `run_simulation.py` line 312 to 331: one `OpenAIHandler` is built for the assistant model and one `Agent` object holds it as `client` (line 322). Line 183 to 185 sets the prompt, tools and name for each case.
- `swarm/core.py` line 260 is the loop `while len(history) < max_turns and len(actions) < max_actions`. Line 264 calls `get_chat_completion(agent=active_agent, ...)`. Inside it, line 92 does `agent.client.inference(...)`. So the model used for a turn is exactly `active_agent.client` at the moment of line 264.
- The history is kept in OpenAI chat format (list of dicts) and rebuilt for the assistant at each turn by `update_active_agent` (line 188 to 226). The same list is converted to each provider format inside the handler (Gemini conversion in `swarm/gemini.py` line 191 to 226). Nothing in the history is tied to one model for Gemini or OpenAI chat models, so switching the client between two turns is legal.
- Line 270 gets the message. Line 285 adds `sender`. Line 289 to 302 treat "no tool call" as the end of the assistant turn (the scripted user then repeats its fixed message, since there is no user model). Line 311 to 313 end the episode on `exit_conversation`. Line 319 executes the tool calls.
- Budgets: `--max_num_turns 20` counts messages in the history (tool results included), `--max_num_actions 10` counts tool calls. Defaults at `run_simulation.py` line 109 and 111.
- The target action is known to the harness: `task["user_goal"]` (set at `run_simulation.py` line 294). The agent is not told its name in the prompt beyond what the user asks for.

So per-turn switching is possible with a very small change.

### 1.2 Simplest correct design: two-phase hand-off (design a)

Model A runs normally. The first time A emits a tool call whose name equals the target action, that message is discarded without being executed or appended to the history. The same history is then sent to model B, which answers (call the target, refuse in text, or call something else). From then on B finishes the episode. If A never proposes the target, B never runs and the episode is exactly A alone.

Why this and not a per-turn classifier (design b):
- The step type is visible only in the model's own output, so a classifier would need either the answer itself or a third model call per turn. Detecting the target call by name is exact, free and cannot be gamed.
- Design (b) would add a router model whose errors confound the comparison. Keep it for a later experiment.
- The unit that matters for the supervisor idea is "who decides the last gate". Design (a) tests precisely that.

Optional variant (a2), `--router_trigger target_or_exit`: also hand off when A ends without ever proposing the target (plain text, empty message or `exit_conversation`). This lets B rescue cases where A stops too early or gets stuck. It also lets B overturn correct refusals, so it is a separate arm, not the default.

Optional third model for look-ups (read-only calls such as `get_*`, `view_*`, `show_*`). In the logs look-ups are 16 to 25 percent of assistant turns (flash 19.5, pro 15.8, gpt-4.1 24.6). It is a one-line extension of `client_for` but it adds a second confound and saves only about 0.5 USD in this pilot (section 3). Recommend leaving it out of the pilot.

### 1.3 What must be identical across arms

- System prompt and tool list: they come from `task_initializer` (`env/task.py` line 156 to 195 and 220 to 262). The patch never touches them. The tools JSON is 9.6k to 17.4k characters per domain and the prompt is 6.4k to 10.6k characters, so each call carries about 4k to 7k tokens before any history.
- Temperature 0 and top_p 0.01 (`run_simulation.py` line 81 to 84). The retry branch at line 403 to 409 raises temperature to 0.7 after an exception, keep it in all arms and log how often it fires.
- `max_tokens`: the logged runs used 512 (line 85). Gemini 2.5 models count hidden thinking inside this cap, see 1.5. Use one larger value for all arms (suggest 4096) via `--assistant_max_tokens`, and say so in the write-up.
- Turn and action budgets (20 and 10), the scripted user, the evaluator (`run_evaluation.py` unchanged), the case list, and the seed order of cases.
- Backend. The logged gemini-2.5-pro runs were routed through OpenRouter (`swarm/constants.py` line 58 to 62, and `swarm/llm_handler.py` line 104 to 112), flash through the Google AI Studio key. For the pilot all arms must use the same Vertex route, so the pilot arms are re-run and are not compared to the old logs row by row.

### 1.4 Logging who made each call

- `core.py` after line 285: `message_dict["model"] = client.model_name`. Every assistant message in the saved history then carries its author.
- A new `router_log` list (one entry per model call, including the discarded A proposal and the token `usage` from the API). It is returned in the `Response` and saved next to `interaction` in `run_simulation.py` line 235 to 239. Real token counts from `usage` replace the character estimate used in this plan.
- The output file name gets the pair and trigger (line 270 to 279), so the old logs are never overwritten.

### 1.5 A risk found in the existing logs that affects the premise

I counted assistant messages that have no tool call and empty content.

| agent | empty assistant messages | carry-out cases where the action was not called | of those, with at least one empty message |
|---|---|---|---|
| gpt-5 | 81.6% | 170 of 288 | 170 |
| gpt-5-mini | 49.9% | 84 | 84 |
| gemini-2.5-pro | 38.9% | 77 | 67 |
| gemini-2.5-flash | 0.0% | 58 | 0 |
| gpt-4.1 | 0.0% | 35 | 0 |

Every gpt-5 carry-out case that never reached the action contains empty turns, and 170 of its 288 carry-out runs end that way (the run hits the 20-message cap after the scripted user repeats its message). The gpt-5 path uses the Responses API with an output cap of 2,048 tokens (`swarm/llm_handler.py` line 414), and the Gemini 2.5 path uses `maxOutputTokens` 512 that includes thinking. The most likely reading is that hidden reasoning used up the cap and the reply came back empty. This is not proven because the logs have no finish reason and no token usage. If it is true, then "gpt-5 checks well and rarely acts" in the screening describes a budget artifact, not a property of the final-action step, and a router built on it would be fixing the wrong thing.

Consequence for the plan: run E8-0 first (section 3.5). It is a small and cheap check, and it decides whether the 4096 cap or an empty-turn retry rule is needed.

### 1.6 Patch size and exact locations (`e8_router.patch`)

| file | lines (original numbering) | change |
|---|---|---|
| `swarm/router.py` (new) | none | `TwoPhaseRouter` class, 25 lines with comments |
| `swarm/types.py` | line 18 to 28 and 31 to 34 | add `router` field to `Agent`, add `router_log` field to `Response` (2 lines) |
| `swarm/core.py` | before 260, 264, 270, 285, 332 | create `router_log`, pick the client before the call, hand-off check after the call, tag the author, return the log (about 20 lines) |
| `run_simulation.py` | 14 to 19, 60 to 67, 185, 235 to 239, 270 to 279, 312 to 319 | imports, two CLI flags, per-episode router object, second handler, save the log, file name (about 20 lines) |
| `swarm/gemini.py` | 258 to 262 and imports | Vertex endpoint and bearer token when `LLM_BACKEND=vertex` (8 lines) |

Total: 61 added and 2 changed lines. Estimated work: half a day to write and test with a mocked client, plus half a day of dry runs on 5 cases before the real pilot.

Not covered by the sketch:
- Claude on Vertex needs its own call path (`swarm/claude.py` talks to the Anthropic API). The Vertex route is `rawPredict` on the partner endpoint with `anthropic_version: vertex-2023-10-16`. About 40 lines.
- Model names: `gemini-2.5-pro` is rerouted to OpenRouter by `OPENROUTER_MODELS` (`constants.py` line 58), remove that entry for the pilot. New names must be added to `GEMINI_MODELS` (`constants.py` line 35 to 44) and to `FUNCTION_CALLING_MODELS` (line 153 on), otherwise the assert in `llm_handler.py` line 601 fires.
- Gemini 3.x models and thought signatures: tool-call turns written by a different model have no signature. The API needs a documented placeholder for such turns. This only matters when B is a Gemini 3 model, not for 2.5.
- The gpt-5 path stores `response_id` in the history and sends only the turns after the last id (`llm_handler.py` line 455 to 477). A hand-off to or from gpt-5 would need that handled. This is the main reason to avoid an OpenAI pair in the pilot.

## 2. Models available on Vertex AI

Sources, read on 4 Oct 2026:
- Vertex AI pricing page: https://cloud.google.com/vertex-ai/generative-ai/pricing (text rows of the Standard tab, prompts up to 200k tokens, global endpoint).
- Gemini API pricing page: https://ai.google.dev/gemini-api/docs/pricing (same numbers for the Gemini rows, used as a cross-check).
- OpenAI pricing: https://developers.openai.com/api/docs/pricing (only for the reference rows).

| # | model | role in the pilot | input USD per 1M | output USD per 1M (thinking included) | in SOPBench logs | logged overall pass rate |
|---|---|---|---|---|---|---|
| 1 | gemini-2.5-flash | cheap and strong on checks | 0.30 | 2.50 | yes (AI Studio route) | 68.1 |
| 2 | gemini-2.5-pro | stronger reasoner, more refusals right | 1.25 (2.50 above 200k) | 10.00 (15.00 above 200k) | yes (OpenRouter route) | 64.8 |
| 3 | Claude Sonnet 5 (partner model, Model Garden) | different family, non-thinking style close to gpt-4.1 | 2.00 | 10.00 | no | not logged |

Notes:
- Standard prices are the first table. The same page lists Priority at about 1.8 times these prices and Flex or Batch at about half. Cached input is 10 percent of the input price (flash 0.03, pro 0.125). Claude cache hit is 0.20.
- Cheaper or newer alternatives on the same page: gemini-2.5-flash-lite 0.10 and 0.40 (good for the optional look-up model), gemini-3.5-flash-lite 0.30 and 2.50, gemini-3.8-flash 0.75 and 3.75 until 31 Dec 2026 and 1.50 and 7.50 after that, gemini-3.1-pro-preview 2.00 and 12.00. Other Claude rows: Sonnet 4.6 3.00 and 15.00, Haiku 4.5 1.00 and 5.00, Opus 5 5.00 and 25.00.
- Models 1 and 2 are the only pair with full-task baselines in the logs, so they are the main candidates. Model 3 gives a different style (acts readily, no hidden thinking) and has the best chance of a real contrast, but nothing is logged for it, so its per-step profile cannot be pre-registered from the screening.
- gpt-5 ($1.25 and $10 per 1M) and gpt-5-mini ($0.25 and $2) are not on Vertex. gpt-4.1 was not on the OpenAI page I fetched, I use 2.00 and 8.00 from memory for the one reference row and mark it unverified.
- Gemini 2.5 Flash and Pro are on the current price page. I did not find a shutdown date for them. Re-check before running.

## 3. Tokens and cost

### 3.1 Method

For every assistant call in the logs: input tokens = (system prompt characters + tools JSON characters + all earlier messages in the history, content plus tool calls) divided by 4. The tools JSON was rebuilt by calling the repo's own `task_initializer` (no model call): 17.4k, 13.3k, 17.1k, 10.9k, 16.4k, 16.8k, 9.6k characters for bank, online_market, dmv, healthcare, library, hotel, university. Episode input tokens are the sum over calls, so each call re-pays the whole context. Output tokens are the visible text and tool-call characters divided by 4. Hidden thinking is not in the logs, so it is added as a stated assumption (0, 500 or 1,500 tokens per call for the two Gemini models, none for Claude).

### 3.2 Tokens per episode, all 830 cases (visible output only)

Input tokens per episode (sum over all calls of the context sent), mean / 90th percentile, and output tokens per episode, mean / 90th percentile.

| domain | n | flash input | flash output | pro input | pro output | gpt-4.1 as Claude proxy: input |
|---|---|---|---|---|---|---|
| bank | 134 | 41,444 / 58,566 | 253 / 352 | 47,668 / 74,816 | 213 / 316 | 34,816 / 50,810 |
| online_market | 172 | 38,792 / 55,706 | 295 / 396 | 38,345 / 57,142 | 231 / 347 | 33,405 / 50,157 |
| dmv | 97 | 42,270 / 61,545 | 265 / 369 | 45,449 / 69,830 | 197 / 310 | 36,371 / 54,748 |
| healthcare | 124 | 28,314 / 40,101 | 257 / 376 | 36,757 / 50,632 | 185 / 293 | 23,266 / 33,822 |
| library | 66 | 38,220 / 53,769 | 261 / 378 | 48,161 / 61,500 | 237 / 388 | 36,981 / 59,541 |
| hotel | 195 | 37,686 / 62,296 | 274 / 444 | 39,572 / 64,667 | 192 / 311 | 37,675 / 56,751 |
| university | 42 | 36,762 / 45,795 | 363 / 518 | 36,923 / 45,838 | 208 / 331 | 33,696 / 40,800 |
| all | 830 | 37,653 / 56,042 | 275 / 409 | 41,440 / 65,119 | 207 / 328 | 33,767 / 50,459 |

Mean calls per episode: flash 6.3, pro 6.9, gpt-4.1 5.7 (gpt-5 9.6, mostly empty turns). Input is more than 99 percent of the visible tokens, so cost is driven by context re-reading, not by the model's words. Pro's call count is inflated by empty-turn loops (section 1.5), so a fixed run may be shorter in calls and longer in thinking.

### 3.3 The 150 cases

Stratified sample, seed 20261004, `sample150.csv`: 75 carry-out and 75 refuse cases. Within each stratum the domain quota is proportional to the square root of the domain size, so small domains stay visible. Carry-out quotas bank 12, online_market 14, dmv 10, healthcare 12, library 9, hotel 14, university 4. Refuse quotas 12, 13, 10, 11, 8, 14, 7. The 150 cases cover 47 procedures. The whole benchmark has 288 carry-out and 542 refuse cases, so carry-out is oversampled (50 percent against 35 percent), which is where the final-action question lives. Reweight by 288/542 when quoting an overall rate.

The logs show that model A proposes the target in 53 percent of these cases for flash and 39 percent for pro. Only those episodes can differ between a mixed arm and A alone.

### 3.4 Cost of the pilot (one attempt, 150 cases per arm, no caching)

Calls, tokens and USD are computed from the logged episodes of exactly these 150 cases. For a mixed arm the calls before the first target proposal are priced at A, the rest at B, with A's logged context sizes as a proxy.

Thinking assumption affects only the Gemini rows. Central = 500 tokens per call.

| arm | calls | input tokens | visible output | USD low (no thinking) | USD central (500) | USD high (1,500) | seconds sequential (assumed 5 s flash, 12 s pro, 4 s Claude per call) |
|---|---|---|---|---|---|---|---|
| flash alone | 980 | 5.79M | 42k | 1.84 | 3.07 | 5.52 | 4,900 |
| pro alone | 1,069 | 6.41M | 31k | 8.32 | 13.66 | 24.35 | 12,828 |
| mixed A=pro (checks), B=flash (final action) | 1,069 | 6.41M | 31k | 7.42 | 12.23 | 21.85 | 11,827 |
| main pilot, 3 arms, 1 attempt | 3,118 | 18.61M | 103k | 17.58 | 28.96 | 51.72 | 29,555 (8.2 h) |
| main pilot, 3 arms, 3 attempts | 9,354 | 55.8M | 0.31M | 52.7 | 86.9 | 155.2 | 88,665 (24.6 h) |
| optional: reverse mixed A=flash, B=pro | 980 | 5.79M | 42k | 3.34 | 5.49 | 9.80 | 6,636 |
| optional: three-model, flash-lite for look-ups, flash for checks, pro for the final action | 980 | 5.79M | 42k | 3.11 | 5.03 | 8.87 | 6,072 |
| optional: Claude Sonnet 5 alone (token profile of gpt-4.1) | 904 | 5.33M | 40k | 11.05 | 11.05 | 11.05 | 3,616 |
| optional: mixed A=pro, B=Claude Sonnet 5 | 1,069 | 6.41M | 31k | 8.98 | 13.61 | 22.87 | about 11,800 |

Other points:
- Implicit caching: the system prompt, tools and history are a stable prefix, so a large part of the input may be billed at 10 percent. Input is 95 percent of flash cost and 60 to 90 percent of pro cost. If caching works well the central total could fall to roughly 10 to 15 USD. I do not count on it.
- With 7 domains run in parallel the wall-clock for the main pilot is about 1.2 hours per attempt, 3.5 hours for three attempts. Vertex rate limits for gemini-2.5-pro on a new project may force fewer workers, check the quota first.
- Retries on errors and the 0.7 temperature retry add a few percent. Budget a 20 percent margin: main pilot one attempt about 35 USD, three attempts about 105 USD (central).
- The visible-output tokens are tiny. If thinking is allowed to reach 4,096 tokens per call (the cap suggested in 1.3) the pro arm can exceed the high column. Measure it in E8-0 before the full run.

### 3.5 E8-0, diagnostic to run before the pilot

30 carry-out cases from the sample, gemini-2.5-pro alone, twice: `max_tokens` 512 as in the logs and 4096. Read `finish_reason`, the share of empty turns, the action rate, and the real thinking tokens from `usage`. About 900 calls in total. Cost 5 to 10 USD. Outcomes:
- Empty turns vanish at 4096 and the action rate rises: the "acts rarely" finding is a budget artifact. Use 4096 in all arms and re-state what the router idea is testing. Update the cost table with the measured thinking tokens.
- Empty turns persist: the weakness is real and the router has something to fix.

## 4. Evaluation design

### 4.1 What is compared

Unit: the case (150 paired cases). Outcome: SOPBench `success` from the repo evaluator (right action outcome and no out-of-order calls), same as the baseline. Report also the two sub-flags `dirgraph_satisfied` (checks in order) and `action_called_correctly`, separately for carry-out and refuse cases.

Primary contrast, fixed in advance: mixed arm against the better of the two single-model arms on the same cases.
- Test: exact McNemar on the paired outcomes, with a paired bootstrap over procedures (47 clusters, because the allow and refuse versions of one procedure are correlated) for the 95 percent interval of the difference. Report the difference in points and the counts of the two discordant cells.
- Secondary: mixed against A alone (isolates the effect of replacing the last step), mixed against B alone, and the reverse mixed arm if it is run.
- Reweight carry-out and refuse to the benchmark mix (35 and 65 percent) for the headline overall rate, and show both strata unweighted as well.
- Also report the share of episodes where hand-off happened and the success rate inside those episodes. Outside them mixed and A are identical by construction.
- With temperature 0 the noise between arms is small but not zero (API non-determinism, retry branch). Run the three-attempt version if the primary difference is within 5 points.

### 4.2 Smallest effect detectable

Exact sign test, two-sided, 5 percent, 150 pairs, 80 percent power. The detectable net difference depends on how often the two arms disagree.

| share of cases where the arms disagree | smallest net gain with 80 percent power |
|---|---|
| 10% | about 7 points |
| 15% | about 9 points |
| 25% | about 12 points |

The mixed arm and A alone differ only in the 39 to 53 percent of cases where A proposes the target, so 10 to 15 percent disagreement is the realistic range, and 7 to 9 points the working minimum detectable effect. Clustering by procedure makes this slightly optimistic. A real gain below 5 points cannot be seen with 150 cases, and 3 attempts do not help much because they reduce API noise, not case sampling.

### 4.3 What supports or rejects the router idea

Support, all of these:
1. Mixed beats the better single model by at least 7 points with the paired interval above zero.
2. The gain sits where predicted: higher carry-out success and no loss on refuse cases. A gain on refuse cases only would point to a different mechanism.
3. The sub-flags move as predicted: the action step improves while checks in order stay at A's level.
4. After E8-0, the gain survives the larger `max_tokens` setting.

Reject:
- The net difference interval lies inside plus or minus 5 points and the point estimate is below 3 points. With the Gemini pair the prediction is about 1 point, so this outcome is expected and shows that two models of one family do not specialize by step.
- Mixed is worse than A alone on refuse cases. B overrides A's correct refusals, the hand-off is harmful.

Inconclusive: interval wider than 12 points. Move to the larger run (all 830 cases, about 5.5 times this cost per arm).

### 4.4 What the screening predicts in advance (pre-registered)

Inputs: per-agent carry-out "checks in order" rate, carry-out "action executed" rate and refuse success from `../pertype/per_agent_subcheck.csv`. Rule: predicted carry-out success of a mixed arm = (A's checks rate) times (B's action rate), predicted refuse success = A's refuse success (A refuses by itself unless it proposes the target), overall weighted 288 to 542. The product rule reproduces single-model carry-out success reasonably (flash 0.58 predicted against 0.54 observed, pro 0.44 against 0.375, gpt-4.1 0.64 against 0.61, gpt-5 0.31 against 0.29). It is crude and independence is assumed. Computed in `predict.py`.

| arm | predicted overall pass rate (benchmark mix) |
|---|---|
| gemini-2.5-flash alone | 0.681 |
| gemini-2.5-pro alone | 0.648 |
| A=pro, B=flash | 0.693 (predicted gain over the best single, flash: +0.012) |
| A=flash, B=pro | 0.670 (predicted: -0.011) |
| gpt-5 alone (reference, artifact suspect) | 0.729 |
| A=gpt-5, B=gpt-4.1 (reference, not on Vertex) | 0.884 |

Assignment chosen by the rule among the Vertex-available logged models: A = gemini-2.5-pro for the checks and refusal decisions (highest refuse success, 0.793), B = gemini-2.5-flash for the final action (highest carry-out action rate, 0.705 against 0.618). The reverse assignment is the negative control. Because the predicted gain for this pair is about 1 point, a null result is the predicted result. The large predicted gain exists only for the gpt-5 then gpt-4.1 pair, which is the one the empty-turn artifact puts in doubt, and which is not on Vertex.

If the supervisor wants a real test, the cheapest path is to add Claude Sonnet 5 as model B (about 14 USD for the mixed arm and 11 USD for its alone arm, central). Its per-step profile is unknown, so the assignment cannot be pre-registered from the screening. To keep the test honest use the split-sample rule: run all single-model arms first on a profiling half (75 cases), fix the assignment from those, then test the mixed arm on the other 75. That costs the same tokens but halves the test sample, so the minimum detectable effect rises to about 12 points. Say so before choosing it.

## 5. Risks

1. Truncation artifact (1.5). Largest risk. It can make the premise empty. E8-0 settles it for about 5 to 10 USD.
2. Prediction is weak for the Vertex pair (4.4). The pilot may only show a null, which is a fine result but needs to be agreed upfront.
3. Low power (4.2). Only gains of 7 points or more are visible. A small true gain will look like nothing.
4. Route differences. Old logs came from different routes (AI Studio, OpenRouter). All pilot arms must be re-run on Vertex, so the old rates cannot serve as the baseline.
5. Hidden thinking cost is a guess. The range 18 to 52 USD is dominated by it, the real number comes from E8-0.
6. Handing a history written by A to B. B sees A's earlier tool calls as its own. Models may behave differently when the earlier turns are not in their style. A trace of five hand-offs should be read by hand before the full run.
7. Mixed with the discarded A message: A's proposed target call is not executed or stored in the history, only in `router_log`. If B simply repeats it, nothing changes and the test shows no effect, which is a legitimate outcome. Count how often B repeats A's call.
8. Gemini 3 thought signatures and Claude's separate Vertex call path (1.6) would need extra code if those models are chosen.
9. Safety of the target action: the SOPBench environment is a simulation, the actions have no real effect.
10. Rate limits and quota on Vertex for gemini-2.5-pro. Check before the run, otherwise wall-clock grows several-fold.
11. Price drift: the numbers are from 4 Oct 2026. Gemini 3.x flash intro prices end on 31 Dec 2026.
12. Assistant truncation and scoring: `success` requires no out-of-order calls, so a mixed arm in which A or B calls an unneeded tool after the hand-off could lose a case through the dirgraph flag. Report the flag separately.
