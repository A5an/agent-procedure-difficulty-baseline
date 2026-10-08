# PREREG forecaster panel (round 10, 8 Oct 2026, before any panel call)
After the ablation: a direct question "with what chance will a typical 2024-2025 agent fully succeed" to Sonnet works as
well as the pre-mortem, and Opus and Sonnet have different blind spots (their average beat each on tau2 and banking).
Question: does a panel of different models, averaged, rank better than Sonnet alone, and does any model size suffice?

Prompt: exactly the D prompt of ablate/ablate.py (d_tau for tau2 and banking, d_gen for TAC, MCPMark, DrafterBench).
Models: Haiku 4.5, Opus 5.5 (claude -p), Gemini 2.5 Pro, Gemini 3.5 Flash, Gemini 3.8 Flash (Vertex, temperature 0),
plus the existing Sonnet 5.5 D. Opus only on tau2, banking, TAC and MCPMark (cost).
PANEL = mean of within-benchmark z-scores of all available models' logit(1 - p_success).
Primary: PANEL vs Sonnet D on TAC and MCPMark (fresh), paired bootstrap. Secondary: every model alone on every
benchmark; PANEL on tau2 (old and new agent population) and banking; PANEL under the protocol cells on tau2.
One run per model, no prompt changes.
