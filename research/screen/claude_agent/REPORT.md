# Exp 8: Claude agent on TauIndianBank, probes from synthetic past conversations (4 Nov 2026)

Plan `PREREG.md`, deviations `DEVIATIONS.md`. All models through `claude -p` on the subscription: agent Sonnet 5.5,
user simulator Haiku 5.5, probe writer Opus 5.5. Stopped by Tima before the end to spare subscription limits: 120 of
120 real episodes, 114 of 120 documentation probes, 98 of 120 synthetic-conversation (PSEUDO) probes; the seeded set was
stopped early (13 episodes) and is not analysed. Analysis on the 17 families with at least 4 real and 3 probe episodes
of each set (`analyze8.py`, `analyze8.out`).

| | real | DOC probes | PSEUDO probes |
|---|---|---|---|
| mean success | 0.52 | 0.85 | 0.62 |
| rho with real success | | 0.70 [0.39, 0.82] | 0.59 [0.16, 0.89] |
| Brier skill vs base rate | | -0.04 | +0.27 |
| mean abs error of the percentage | | 0.32 | 0.21 |
| delegate at p >= 0.9: cases / success | | 65% / 73% | 47% / 73% |

H1 (seeded) not run. H2 (PSEUDO higher Brier skill than DOC): +0.31 [-0.07, +0.82], not supported at 95%, direction
as expected.

Reading. For the Sonnet agent, documentation probes rank the families as well as seeded probes did for Gemini (0.70),
but they are far too easy: Sonnet passes 85% of them and 52% of the real tasks, so their percentages are useless
(skill below zero). Probes from Opus-written past conversations are much closer in level (62% vs 52%) and give a usable
percentage (skill +0.27, the level of real seeded probes for g38), with the ranking slightly worse. Neither set yet
makes the delegation rule safe (73% success among delegated). Small sample (17 families, 3 to 6 probes each); a clear
lead, not a result.
