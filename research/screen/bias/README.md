# Forecaster bias checks (20 Oct 2026, post hoc, no new LLM calls)

Question (from Tima, after a friend's remark): does an LLM forecaster favour agents of its own family, and is the direct
rating D just text length?

Inputs: ideas10/panel/panel_scores.csv (direct question D to six models: Sonnet 5.5, Opus 5.5, Haiku 4.5,
Gemini 2.5 Pro, Gemini 3.8 Flash, Gemini 3.5 Flash), repo responses.jsonl and features/length.csv.

`family.py`: per agent, within-domain Spearman between the forecaster score and that agent's failure rate per task,
averaged by forecaster x agent family. Interaction = (Claude forecasters on Claude agents minus on Gemini agents) minus
(Gemini forecasters on Claude agents minus on Gemini agents), bootstrap over tasks (300 draws).
- tau2 (15 agents: 6 GPT, 3 Claude, 3 Qwen, 2 Gemini, 1 GLM): interaction +0.014 [-0.037, +0.063]. Every forecaster
  ranks GPT agents best and Gemini agents worst. No own-family advantage.
- banking (12 agents, 4 Claude, no Gemini): Claude forecasters predict Claude agents slightly worse than the others
  (-0.03 to -0.05).
- Caveat: few agents per family (2 Gemini on tau2), per-agent rho is noisy.

`length.py`: within-domain Spearman of D (Sonnet) with difficulty, and partial Spearman given log length.
- tau2: D 0.574, length 0.231, D vs length 0.295, D given length 0.519.
- banking: D 0.531, length 0.532, D vs length 0.377, D given length 0.421.

Outputs: family_out.txt, length_out.txt. Run with the repo venv: `python -I family.py`.
