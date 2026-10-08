Cases where the 3 base-prompt programs (t0, two samples at t=1) disagree on whether to call the target action: 0.063 (52/830); disagree on success: 0.051
Mean success of the three programs: 0.770, 0.769, 0.761
All three agree and all fail: 0.211; all three agree and succeed: 0.739
Per procedure (70): Spearman(program disagreement rate, agent difficulty = 1 - mean success of 25 real agents) = 0.19 (p=0.106)
Per procedure: Spearman(program failure rate, agent difficulty) = 0.49 (p=0.000)
Procedures with >=5 cases (44): disagreement vs agent difficulty rho=0.37 (p=0.012)
Case level (830): Spearman(disagreement, agent difficulty) = 0.17 (p=0.000)
Case level within procedure (demeaned): rho=0.08 (p=0.026)