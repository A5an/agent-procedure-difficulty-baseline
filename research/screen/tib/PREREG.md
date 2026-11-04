# P0-2: the MF model on TauIndianBankBench (pre-registration, 2 Nov 2026)

Written before any probe was generated and before any agent episode. Approved by Tima on 2 Nov ("пилот цены и маленький
вариант"), session cap 30 USD on Gemini, all spend through `proxy.py` (ledger `ledger.jsonl`, cap 29.5 USD).
Draft it replaces: `../mf/PREREG_P0-2_DRAFT.md`.

## Question

The MF model (`../mf/REPORT.md`) was built and tuned on SOPBench only. On a benchmark it has never seen, do documentation
probes plus the frozen model predict which procedures an agent fails, and does a router built on that beat the best
constant policy?

## Benchmark

TauIndianBankBench, code commit 1ccb64f (MIT), data from Hugging Face NPCI/tau-indian-banking (fetched 2 Nov). 1,000
tasks in 50 scenario families; one family = one procedure (each family has one "purpose" line and one gold action
pattern). Evaluation unchanged: ACTION + DB, no judge.

## Selection (seed 20261102, `select_tasks.py`)

20 families drawn at random from the 50. 6 benchmark tasks per family drawn at random = the real tasks (120).
Class of a family's unit: 1 if its gold actions contain a state-mutating tool (toolkit flag), else 0.

## Probes (`gen_probes.py`)

Generator gemini-3.8-flash, temperature 0.7. It sees: the family's purpose line, `policy.md`, `agent_instruction.txt`,
the tool schemas, and the full database record of 3 candidate customers (random, never customers used by any task of
the 20 selected families, filtered by the products the purpose line mentions). It never sees any benchmark task. It
writes one task in the benchmark's format (scenario, ordered steps, gold actions with compare_args on writes, one NL
assertion). Validity: every gold action is executed on a fresh environment with that customer logged in; a probe with
any tool error or failure message is dropped. Up to 10 attempts per family to get 6 valid probes. All probes are written
to `probes.json` and its sha256 recorded in `FREEZE.md` before the first real episode.

## Runs (`run.py`)

Agents gemini-3.8-flash and gemini-2.5-flash (the two strongest of the four agents in the MF model that we can run),
temperature 0, max_tokens 8192. User simulator gemini-2.5-flash, temperature 0, thinking off, the benchmark's user
prompt. max_steps 200, max_errors 10, one attempt per task. Both agents on the 120 real tasks and the 120 probes.

Cost pilot first: 5 real tasks per agent (10 episodes, from families outside the 20, so they do not touch the test).
If pilot cost x 480 + generation would exceed the remaining cap, the size drops to 5 real + 5 probes per family before
any test episode, and that is written in REPORT.md.

## Frozen before the first test episode

- MF parameters refitted on all 68 SOPBench procedures (`mf_params.pt`, sha256 in `FREEZE.md`). Agent mapping:
  g38 -> the "g38" agent of the model, flash -> "flash".
- `analyze.py` and the cost settings of `../mf/route.py` (API cost per case taken from this run's ledger).

## Primary outcomes

1. Spearman between MF-predicted and real per-family failure rate, per agent, over the 20 families, 95% bootstrap over
   families. Support: rho >= 0.40 and lower bound > 0 for at least one agent.
2. Router over {g38, flash, human} per family, F = 20, human $5 at 0.99: realized loss minus the best constant policy,
   bootstrap over families. Support: upper bound < 0.
3. Log-loss of MF vs the raw probe rate and the constant (SOPBench-level) on the real tasks.

Reported alongside, not used for the verdict: the raw probe rate's rho; MF refitted on this benchmark with leave-one-
family-out (agents' level learned from the other 19 families); split-half reliability of the real target (it will be
low with 6 tasks per family, which caps rho); probe validity rate; cost per episode.

## Known weaknesses, stated up front

6 real tasks per family is a noisy target; 20 families give wide intervals. The benchmark's own tasks are LLM-written
from templates, so probes written from the purpose line are a friendly test, as on SOPBench. The purpose line lists the
steps, which is more than a real SOP title would. The MF agent levels were learned on another harness.
