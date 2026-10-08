Agents: all 28 configurations in responses.jsonl. Overall agent success: best 0.753, median(25 real) 0.417, median(28) 0.457

### Variant A (user_known dict) (n=830)
| subset | n | compiled | best agent (25 real) | median (25) | median (28) | best (28) | agents below compiled (25) | (28) |
|---|---|---|---|---|---|---|---|---|
| all | 830 | 0.770 | 0.753 | 0.417 | 0.457 | 0.753 | 25/25 | 28/28 |
| bank | 134 | 0.761 | 0.769 | 0.545 | 0.578 | 0.769 | 24/25 | 27/28 |
| dmv | 97 | 0.918 | 0.876 | 0.660 | 0.691 | 0.876 | 25/25 | 28/28 |
| healthcare | 124 | 0.629 | 0.927 | 0.548 | 0.641 | 0.927 | 14/25 | 14/28 |
| hotel | 195 | 0.656 | 0.708 | 0.215 | 0.233 | 0.708 | 24/25 | 25/28 |
| library | 66 | 0.742 | 0.606 | 0.273 | 0.288 | 0.667 | 25/25 | 28/28 |
| online_market | 172 | 0.924 | 0.895 | 0.430 | 0.439 | 0.895 | 25/25 | 28/28 |
| university | 42 | 0.810 | 0.952 | 0.500 | 0.583 | 0.952 | 22/25 | 22/28 |
| perform (should succeed) | 288 | 0.431 | 0.611 | 0.299 | 0.302 | 0.611 | 17/25 | 20/28 |
| refuse | 542 | 0.950 | 0.856 | 0.485 | 0.522 | 0.963 | 25/25 | 27/28 |

Per procedure (70): compiled success above agent mean (25) on 61, equal 3, below 6. Mean procedure success: compiled 0.726, agents(25) 0.456.
Spearman(compiled success, agent mean) over procedures: 25 configs rho=0.49 (p=0.000); 28 configs rho=0.52 (p=0.000)
Failure types (191 failures): wrong decision: refused a case that must be performed: 147; right decision, required checks not called (dirgraph): 22; wrong decision: performed a case that must be refused: 14; right decision, tool response mismatch (constraint_not_violated): 6; program crashed or timed out: 2
Decision accuracy (action called iff must be performed): 0.804; success among decision-correct: 0.958

### Variant B (parameters extracted from the user message) (n=830)
| subset | n | compiled | best agent (25 real) | median (25) | median (28) | best (28) | agents below compiled (25) | (28) |
|---|---|---|---|---|---|---|---|---|
| all | 830 | 0.764 | 0.753 | 0.417 | 0.457 | 0.753 | 25/25 | 28/28 |
| bank | 134 | 0.739 | 0.769 | 0.545 | 0.578 | 0.769 | 24/25 | 27/28 |
| dmv | 97 | 0.897 | 0.876 | 0.660 | 0.691 | 0.876 | 25/25 | 28/28 |
| healthcare | 124 | 0.629 | 0.927 | 0.548 | 0.641 | 0.927 | 14/25 | 14/28 |
| hotel | 195 | 0.662 | 0.708 | 0.215 | 0.233 | 0.708 | 24/25 | 25/28 |
| library | 66 | 0.727 | 0.606 | 0.273 | 0.288 | 0.667 | 25/25 | 28/28 |
| online_market | 172 | 0.924 | 0.895 | 0.430 | 0.439 | 0.895 | 25/25 | 28/28 |
| university | 42 | 0.810 | 0.952 | 0.500 | 0.583 | 0.952 | 22/25 | 22/28 |
| perform (should succeed) | 288 | 0.406 | 0.611 | 0.299 | 0.302 | 0.611 | 17/25 | 20/28 |
| refuse | 542 | 0.954 | 0.856 | 0.485 | 0.522 | 0.963 | 25/25 | 27/28 |

Per procedure (70): compiled success above agent mean (25) on 61, equal 3, below 6. Mean procedure success: compiled 0.719, agents(25) 0.456.
Spearman(compiled success, agent mean) over procedures: 25 configs rho=0.47 (p=0.000); 28 configs rho=0.52 (p=0.000)
Failure types (196 failures): wrong decision: refused a case that must be performed: 161; right decision, required checks not called (dirgraph): 16; wrong decision: performed a case that must be refused: 12; right decision, tool response mismatch (constraint_not_violated): 5; program crashed or timed out: 2
Decision accuracy (action called iff must be performed): 0.789; success among decision-correct: 0.968

### Variant A + example database in the compile prompt (C) (n=830)
| subset | n | compiled | best agent (25 real) | median (25) | median (28) | best (28) | agents below compiled (25) | (28) |
|---|---|---|---|---|---|---|---|---|
| all | 830 | 0.863 | 0.753 | 0.417 | 0.457 | 0.753 | 25/25 | 28/28 |
| bank | 134 | 0.731 | 0.769 | 0.545 | 0.578 | 0.769 | 23/25 | 26/28 |
| dmv | 97 | 0.897 | 0.876 | 0.660 | 0.691 | 0.876 | 25/25 | 28/28 |
| healthcare | 124 | 0.976 | 0.927 | 0.548 | 0.641 | 0.927 | 25/25 | 28/28 |
| hotel | 195 | 0.769 | 0.708 | 0.215 | 0.233 | 0.708 | 25/25 | 28/28 |
| library | 66 | 0.758 | 0.606 | 0.273 | 0.288 | 0.667 | 25/25 | 28/28 |
| online_market | 172 | 0.988 | 0.895 | 0.430 | 0.439 | 0.895 | 25/25 | 28/28 |
| university | 42 | 0.952 | 0.952 | 0.500 | 0.583 | 0.952 | 24/25 | 26/28 |
| perform (should succeed) | 288 | 0.691 | 0.611 | 0.299 | 0.302 | 0.611 | 25/25 | 28/28 |
| refuse | 542 | 0.954 | 0.856 | 0.485 | 0.522 | 0.963 | 25/25 | 27/28 |

Per procedure (70): compiled success above agent mean (25) on 63, equal 3, below 4. Mean procedure success: compiled 0.798, agents(25) 0.456.
Spearman(compiled success, agent mean) over procedures: 25 configs rho=0.47 (p=0.000); 28 configs rho=0.52 (p=0.000)
Failure types (114 failures): wrong decision: refused a case that must be performed: 51; right decision, required checks not called (dirgraph): 43; right decision, tool response mismatch (constraint_not_violated): 9; wrong decision: performed a case that must be refused: 6; program crashed or timed out: 5
Decision accuracy (action called iff must be performed): 0.927; success among decision-correct: 0.931

### Variant B + example database in the compile prompt (C) (n=830)
| subset | n | compiled | best agent (25 real) | median (25) | median (28) | best (28) | agents below compiled (25) | (28) |
|---|---|---|---|---|---|---|---|---|
| all | 830 | 0.836 | 0.753 | 0.417 | 0.457 | 0.753 | 25/25 | 28/28 |
| bank | 134 | 0.709 | 0.769 | 0.545 | 0.578 | 0.769 | 23/25 | 25/28 |
| dmv | 97 | 0.887 | 0.876 | 0.660 | 0.691 | 0.876 | 25/25 | 28/28 |
| healthcare | 124 | 0.952 | 0.927 | 0.548 | 0.641 | 0.927 | 25/25 | 28/28 |
| hotel | 195 | 0.703 | 0.708 | 0.215 | 0.233 | 0.708 | 24/25 | 27/28 |
| library | 66 | 0.727 | 0.606 | 0.273 | 0.288 | 0.667 | 25/25 | 28/28 |
| online_market | 172 | 0.988 | 0.895 | 0.430 | 0.439 | 0.895 | 25/25 | 28/28 |
| university | 42 | 0.952 | 0.952 | 0.500 | 0.583 | 0.952 | 24/25 | 26/28 |
| perform (should succeed) | 288 | 0.615 | 0.611 | 0.299 | 0.302 | 0.611 | 25/25 | 28/28 |
| refuse | 542 | 0.954 | 0.856 | 0.485 | 0.522 | 0.963 | 25/25 | 27/28 |

Per procedure (70): compiled success above agent mean (25) on 63, equal 3, below 4. Mean procedure success: compiled 0.782, agents(25) 0.456.
Spearman(compiled success, agent mean) over procedures: 25 configs rho=0.46 (p=0.000); 28 configs rho=0.51 (p=0.000)
Failure types (136 failures): wrong decision: refused a case that must be performed: 92; right decision, required checks not called (dirgraph): 28; right decision, tool response mismatch (constraint_not_violated): 8; wrong decision: performed a case that must be refused: 5; program crashed or timed out: 3
Decision accuracy (action called iff must be performed): 0.881; success among decision-correct: 0.949