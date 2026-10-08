# Two-stage gold-count feature A, 5-seed replication (agent `fewshot_A5`, 6 Oct 2026)

Saved by the main session from the agent's final message. No LLM calls. PREREG.md before the first run. Leakage assert
on about 55,000 gold lookups, no violation. CIs: paired bootstrap over procedures stratified by domain (optimistic; the
standard evaluate.py CIs include 0 for every A variant).

| column | tau2 new_proc | tau2 new_dom | tau2 mean | vs adele | vs c_lad |
|---|---|---|---|---|---|
| adele | 0.128 | 0.150 | 0.139 | | |
| c_lad | 0.298 | 0.285 | 0.292 | +0.153 | |
| A | 0.434 | 0.467 | 0.451 | +0.312 [+0.102, +0.418] | +0.159 [+0.018, +0.301] |
| A_adele | 0.261 | 0.280 | 0.270 | +0.131 | -0.021 |
| c_lad_A | 0.307 | 0.320 | 0.314 | +0.174 | +0.022 [-0.006, +0.035] |
| A_dm (demeaned targets) | 0.326 | 0.467 | 0.397 | | |
| A_w (writes only) | 0.438 | 0.496 | 0.467 | +0.328 | +0.175 |
| A_all (total actions) | 0.326 | 0.465 | 0.395 | | |
| A_raw (no stage 1) | 0.296 | 0.210 | 0.253 | | |

Mechanism: predicted write count carries the signal (tracks gold writes within domain 0.45 to 0.80). Telecom drives it:
per domain new_dom A airline .50, retail .23, telecom .70 (adele -.01); airline+retail only A .28 vs adele .29 vs c_lad .31.
Telecom has 3 procedures (MMS, mobile data, no service), plan features nearly constant within a family, the family oracle
reaches 0.68. So A mostly sorts 3 telecom families, not tasks inside a family.
SOPBench (gold = directed_action_graph node counts of training cases): A 0.036 / 0.059, c_lad_A 0.237 / 0.205.
Pooled four scenarios: A 0.249 (vs c_lad -0.009), c_lad_A 0.267 (vs c_lad +0.009 [-0.005, +0.017]).
Verdict: replicates on tau2 but is a telecom-family effect, not a general feature. Not worth adding to c_lad.
Files: PREREG.md, a5.py, run.py, score.py, extra.py, mech.py, domrho.py, out_tau2/, out_sop/, score_all.txt.
