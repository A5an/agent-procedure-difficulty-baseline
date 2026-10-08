# PREREG over-checking (round 10, 8 Oct 2026). POST HOC: written after an analysis of SOPBench gold constraints.

Finding that motivated it (execdrivers.py, analysis only, uses gold fields): inside a procedure, policy variants WITHOUT a
login or credential condition are much harder (within-procedure rho of "policy mentions login" with b: -0.34 over all
cases, -0.69 on must-perform cases, -0.28 on must-refuse cases; only bank and dmv vary). Must-perform cases where the
customer gives no credential have mean b 3.20 against 0.82. Reading: agents apply default checks the policy does not ask
for (authentication), cannot satisfy them, and refuse a case that should be performed. This is the mirror image of the
prior-conflict hypothesis (prior/PREREG.md): there the policy asks for checks the assistant would not do; here the
assistant does checks the policy does not ask for.

Features OVER (computed from prior/prior_raw.json, no new LLM calls; universal definitions):
- ov_extra_checks: mean over the 6 naive plans of the number of check or lookup tools not in the policy plan
- ov_auth_extra: share of naive plans calling an authentication-like tool (name contains login, auth, verify, identity,
  password) when the policy plan calls none
- ov_nocred: the request contains no credential word (password, identification, identity, id, pin, passcode)
- ov_auth_gap = ov_auth_extra * ov_nocred
Variants (one harness run, sopbench + tau2, new_procedures + new_domain): o1 = c_lad + GAP + PRIOR + OVER,
o2 = c_lad + GAP + OVER; references g1_clad_gap, pr2_cladgap_prior. Report per domain on SOPBench as well, because the
effect can only show where policies differ in their login rules (bank, dmv). Treat as exploratory whatever the result.
