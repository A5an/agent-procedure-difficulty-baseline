# Data in this repository

The MIT license in `LICENSE` covers the code in `src/`. The files in `data/` are derived from
third-party releases and keep their licenses:

| files | derived from | license |
|---|---|---|
| `data/sopbench/*` | SOPBench task data and released agent runs, github.com/Leezekun/SOPBench (commit d2622008) | CC BY 4.0 (SOPBench authors) |
| `data/tau2/*`, `data/tauk_banking/*` | tau2-bench task files, github.com/sierra-research/tau2-bench (commit b7ea9074); outcomes of the public leaderboard submissions in the sierra-tau-bench-public bucket | task text MIT (tau2-bench authors); outcome matrices summarise the public leaderboard results |
| `data/adele_validation/*` | annotations of instances of the ADeLe battery, github.com/Kinds-of-Intelligence-CFI/ADeLe-AIEvaluation | CC BY-SA 4.0 (ADeLe battery) |
| `*/adele_responses.jsonl`, `*/human_time.jsonl`, `data/adele_retest/*` | answers of Gemini 3.8 Flash to the ADeLe rubric prompts and to the human-time prompt, generated for this repository | same terms as the text they annotate |

The third-party code in `third_party/` is included as git submodules at pinned commits and keeps
its own license (both MIT).
