# Memory for a new procedure: result (11 Nov 2026)

Plan `PREREG.md`, code `memory.py`, output `memory.out`, per-procedure numbers `results.json`. No LLM calls, $0.
SOPBench, 28 target procedures (>= 8 cases), 28 agents, 30 probe draws. A probe is a real held-out case of the
procedure. Brier is on the test cases; a forecast equal to the true test rate would score 0.157 (noise floor),
the agent's average rate over other procedures (Prior) scores 0.216.

## 1. No probes at all

| Memory content | Prior | ProcMemory | CaseMemory |
|---|---|---|---|
| all other procedures (in-domain) | 0.2161 | 0.1956 | 0.1981 |
| only other domains (new domain) | 0.2221 | 0.2150 | 0.2274 |

In-domain, ProcMemory is better than Prior by 0.0205 [0.0123, 0.0289]. On a new domain the difference shrinks to
0.0071 [0.0027, 0.0129], and CaseMemory is not better than Prior (-0.0053 [-0.0165, +0.0061]).

## 2. How many probes Memory replaces

Brier with n probes, prior weight m = 4:

| n probes | Prior + probes (in-domain) | ProcMemory + probes (in-domain) | Prior + probes (new domain) | ProcMemory + probes (new domain) |
|---|---|---|---|---|
| 0 | 0.2161 | 0.1956 | 0.2221 | 0.2150 |
| 1 | 0.2086 | 0.1951 | 0.2125 | 0.2079 |
| 2 | 0.2030 | 0.1938 | 0.2057 | 0.2025 |
| 4 | 0.1967 | 0.1915 | 0.1982 | 0.1964 |

In-domain: Memory with 0 probes (0.1956) is about as good as 4 probes without Memory (0.1967), and Memory with
2 probes beats 4 probes without Memory by 0.0029 [0.0005, 0.0053]. So inside a known domain Memory is worth about
4 probes per procedure. New domain: Memory with 0 probes is about 1 probe (0.2150 vs 0.2125), and Memory + 2 probes is
worse than 4 probes (-0.0043 [-0.0060, -0.0026]); the gain of Memory at equal n stays positive but small
(0.0018 to 0.0046).

CaseMemory with probes added as ordinary memory entries works in-domain (same level as ProcMemory + probes) and
breaks on a new domain (0.2735 with 1 probe): the few probes are outvoted by unrelated neighbours from other domains.
A router must treat the procedure's own probes separately from the memory, not as more entries in one store.

## 3. Stream (memory starts empty)

Brier by number of procedures already stored (40 random orders):

| stored | Prior | ProcMemory | Prior + 2 probes | ProcMemory + 2 probes |
|---|---|---|---|---|
| 0-4 | 0.2564 | 0.2645 | 0.2250 | 0.2287 |
| 5-9 | 0.2217 | 0.2238 | 0.2023 | 0.2041 |
| 10-19 | 0.2228 | 0.2181 | 0.2064 | 0.2049 |
| 20-34 | 0.2154 | 0.2037 | 0.2021 | 0.1978 |
| 35-65 | 0.2158 | 0.1977 | 0.2019 | 0.1949 |

Memory starts to help after about 10 stored procedures and gives its full gain after about 35.

## Reading

Same picture as the kNN baseline in the paper (good in-distribution, collapse on OOD): similarity between procedures
carries information about agent success only inside a domain. For the router: Memory is a way to cut probe cost for
new procedures in a domain where other procedures were already checked; for a new domain probes are still needed.

## Limits

- One benchmark. SOPBench procedures of a domain share tools and policy style, which favours Memory.
- Probes are real held-out cases, the best case for probes; generated probes are noisier, which would make Memory
  look relatively better.
- The stream assumes the true outcomes of every finished procedure are stored (full feedback). In deployment the
  outcomes of procedures given to a human are not observed.
- Many paired differences were computed; they are not corrected for multiple comparisons.
- m = 4 was the best of the three prior weights for Prior + probes, m = 8 for ProcMemory + probes; the tables use
  m = 4 for both.
