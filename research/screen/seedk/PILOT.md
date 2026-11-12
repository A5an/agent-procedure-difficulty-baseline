# E1 and E2 pilots (12 Nov 2026)

5 families (hpemi, edgbcarddisamb, sqmand, hpgold, edgbfdcap). No rho on the pilot (PREREG.md). Ledger 9.71 USD of 20.

- E1 generation: 90 valid probes from 92 attempts (k5: one JSON parse failure, one "no fitting candidate"); 2.93 USD,
  about 0.20 USD per family and condition.
- E1 episodes: 180 of 180, 0 errors, all USER_STOP; 4.39 USD (about 0.024 per episode). Mean probe reward
  g38 / flash: k1 0.77 / 0.63, k5 0.77 / 0.67, foreign 0.80 / 0.83.
- E2 episodes (pro simulator on the existing seeded probes): 60 of 60, 0 errors; 2.38 USD (about 0.040 per episode).
  Mean reward g38 0.63, flash 0.77. Outcome differs from the recorded flash-simulator run on 6 of 30 probes (g38) and
  2 of 30 (flash); descriptive only.
- Incident: the Mac slept 00:48-01:13; the three g38 runners hung on dead connections and were restarted at 01:15
  (finished episodes kept, in-flight ones rerun). Nothing else changed.

Projected full runs (remaining 45 families): E1 generation about 26 USD, E1 episodes about 18 USD, E2 about 21 USD;
together about 65 USD. Suggested caps with margin: E1 50, E2 25.

## Full run (12 Nov 2026)

Tima approved the full E1 and E2 runs ("давай оба, е1 и е2") with caps E1 50 USD and E2 25 USD on top of the pilot;
the proxy enforces the sum (cap_usd.txt 84.7 = 9.7 + 75). Correction found right after the approval: the E1 projection
above counted 720 remaining episodes; the right number is 1,620 (3 conditions x 45 families x 6 probes x 2 agents), so
E1 needs about 26 (generation) + 40 (episodes) = 66 USD. E2 and the E1 generation start now; the E1 episodes wait for
a decision on the E1 cap.
12 Nov, later: Tima set a hard total of 70 USD for everything ("максимум 70 долларов в общей сумме за все время").
cap_usd.txt = 70.0. At 26.91 USD spent: k1 generation stopped (28 of 50 families saved), E2 runs paused
(seeded_pro_flash 123/300, seeded_pro_g38 77/300); k5 and foreign generation continue.
12 Nov: Tima: "давай добьем хотя бы первый". E1 episodes for k5 and foreign, both agents, all 50 families; then k1 on
its 28 generated families with what is left under the 70 USD cap. E2 stays at 202 of 600.
