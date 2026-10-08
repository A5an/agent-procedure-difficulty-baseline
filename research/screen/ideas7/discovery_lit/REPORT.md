# How multi-agent AI systems make discoveries, and a recipe for us (agent `discovery_lit`, 6 Oct 2026)

Saved by the main session from the agent's final message. Web research, sources fetched 6 Oct 2026 unless noted.

Claim "GPT solved the 10 million Navier-Stokes problem": mostly false or exaggerated. Clay prize is 1 million. OpenAI
announced on 8 Sep 2026 a proof of finite-time blow-up for the FORCED case (Clay statements C and D), not the zero-force
global smoothness question (A and B). Unreleased internal model, about 10,000 concurrent agents, about 88 h plus 17 h Lean
formalisation (secondary sources, openai.com returned 403). Clay still lists the problem as unsolved, rules require
publication, two years and community acceptance. Priority dispute with Buckmaster and Alpoge (forced Euler).
Sources: en.wikipedia.org Navier-Stokes existence and smoothness; implicator.ai; xenospectrum; manlius substack; claymath.org rules.

Claim "Claude found something related to DNA": true with caveats. anthropic.com/news/claude-discovers-novel-enzyme-system
(23 Sep 2026): about 950 Claude agents, 21 h, about 210M tokens searched sequence databases, found a CRISPR-like system of
array-associated reverse transcriptases (ART) in bacteriophages; function unknown, wet lab by humans, preprint.

Systems and loops: AlphaEvolve (arXiv 2506.13131; LLM ensemble evolves code, automatic human-written evaluator, cascade,
MAP-Elites + islands, thousands of samples); FunSearch (Nature 2023, islands, millions of samples, only programs that run
and score advance); Google AI co-scientist (Gemini agents generate, debate, Elo tournament, wet-lab validation); Sakana AI
Scientist (LLM reviewer about 69% balanced accuracy, agent once edited its own timeout script); OpenAI Erdos results
(rediscoveries in Oct 2025, Lean-verified #728 in Jan 2026); Anthropic ART (worker, supervisor, curator, editor Claude Code
agents, pre-registered promotion filters, LLM-judge tournament, humans validate); ARIS arXiv 2605.03042 (Claude Code or
Codex executor with cross-family reviewer and audit cascade); GEAR arXiv 2605.13874 (population search over experiments).
Common pattern: cheap trusted automatic evaluator, many cheap proposals filtered hard, humans define the evaluator and
validate, evaluator must be outside the agents' write access.

Recipe for us: candidates = small feature extractors or prompts with a fixed interface and no label access; evaluator =
locked protocol script with a cascade (sanity, small inner CV, full inner CV); 20 to 40 live candidates in 3 to 4 islands,
10 to 20 children per island per round, 8 to 15 rounds, keep by score plus novelty (low correlation with existing
features); nested splits grouped by domain or procedure; a frozen held-out benchmark; at most 3 finalists to one blind
test; audit pass for leakage. Risks: Goodhart on the metric, multiple comparisons, leakage, small data, novelty claims.
