# Exp 1: progressive autonomy (plan, 3 Nov 2026, written before running)

Question: how many shadow cases does a procedure need before routing it is right, with and without probes as the prior?
Fixed before running: options (agents + human), cost settings (same four as ../mf/route.py), shadow pool = random half
of the procedure's real cases and evaluation = the other half (10 random splits), fixed-m policies m = 0, 1, 2, 4, 8
(SOPBench; capped by the pool) and 0..4 (TauIndianBank), a sequential policy that hands over when
P(agent loss < human loss) >= 0.9, volume V = 50 and 200 cases per procedure for the total cost.
Main comparison: probes + shadow vs shadow only at the same m, and the sequential policy's shadow cases, release rate and
wrong-release rate. SOPBench uses the leave-one-domain-out MF fits of ../mf; TauIndianBank refits MF on the other 19
families (agents' level learned on this benchmark). Code: autonomy.py, results_sop.json, results_tib.json.
