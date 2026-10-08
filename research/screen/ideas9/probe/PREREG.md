# Pre-registration, ideas9/probe (7 Oct 2026, written during extraction, before any label was touched and before any harness run)

Idea (round-8 `internals` I1/I2, `outdist` OD1): pre-generation hidden-state probe of a local white-box model, level L1 (local model
reads policy + tools + request; no agent run, no generation). Model Qwen3-8B-Base 8-bit MLX. Prompt = cached domain prefix
(policy/role, principles, tool list) + case policy header (SOPBench) + request + fixed cue "Assistant's first step:". Extraction
code extract.py reads no outcome and no IRT value. tau-Knowledge banking: prefix = Rho-Bank policy header + shell instructions +
list of knowledge-base document groups + 3 tool names (the 697 documents themselves are not in the prompt); request = scenario.

## Fixed before looking at any label
- Layers saved (state after k blocks): 12 16 20 24 28 32. PROBE LAYER FOR ALL VARIANTS: 24 (two thirds of depth), last cue token.
  No layer selection from labels. Per-layer within-procedure correlations are reported as a diagnostic only, never used to pick.
- Probe features = 16 principal components of the layer-24 last-token state (PCA fitted on ALL cases of the same benchmark,
  unlabelled texts only, TRANSDUCTIVE: test-fold request texts take part in the PCA; no outcome ever). Columns are
  appended to the ladder's global features g (standardised on training tasks inside the ladder, ridge penalty chosen by its
  inner CV on training tasks).
- Centred variant: before the PCA, subtract the group mean of the state (SOPBench: per procedure; tau2: per domain, since
  every tau2 task is its own procedure; banking: one group). Group means use unlabelled states of the same group (TRANSDUCTIVE).
- Scalars (I2 + OD1): shift_k = ||last_k - ref_k|| for the 6 layers (ref = state at the end of the policy-only part of the same
  prompt), nll_cond = mean NLL of the request tokens given the policy prefix, pmi = nll_unc - nll_cond (nll_unc: same request
  after a bare "### Conversation" header, no policy). 8 columns, raw (not centred).
- Length control: log(request tokens) and log(policy-header tokens) as 2 columns.

## Variants (ladder lad2 built exactly as ideas8/textcount/run.py builds t0_clad: ADeLe + PLAN + CMP)
- t0_clad: reference, must reproduce 0.258.
- p0_len: t0_clad + length control (2 cols). Control: shows what length alone adds.
- p1_probe: t0_clad + 16 PCs of the layer-24 last-token state. PRIMARY HYPOTHESIS: p1 > t0_clad on the pooled four-scenario
  rho (paired bootstrap CI over procedures excludes 0), and p1 > p0_len.
- p3_scalars: t0_clad + 8 scalars (shift, nll_cond, pmi).
- p4_centred: t0_clad + 16 PCs of the procedure/domain-centred layer-24 last-token state.
(adele + probe variant dropped to keep four.) One harness run, no reruns with changed features; crash reruns allowed.
Everything is reported. Harness scoring = copy of ideas8/textcount/score.py (paired CIs vs adele and vs t0_clad), plus SOPBench
procedure level.

## Banking (fresh benchmark, never used to choose anything)
No PLAN/CMP features exist for banking, so c_lad cannot be built there. (a) Within-domain Spearman with b of: probe-ridge score
(below), scalars, length. (b) Harness on tauk_banking random_cases with q0 = ladder(ADeLe), q1 = q0 + probe PCs, q2 = q0 + length
(secondary, 97 cases, expected to be noisy).

## Diagnostics (not used for any selection)
- Within-procedure Spearman with b (b higher = harder) on SOPBench, pooled as the mean over procedures, for each scalar, each
  length variable, PC1..PC4 of the centred state, per layer for the raw centred state mean-request vs last-token PCs.
- Probe score = ridge on PCA(64) of the centred layer-24 state, target b, trained leave-one-procedure-out (SOPBench), leave-one-domain-out
  (tau2), and trained on SOPBench+tau2 then applied to banking. Alpha fixed on training data by inner CV. Reported as within-group Spearman.
- How much of the probe is length: R^2 of PC1..16 on [log n_req, log n_A]; partial Spearman after residualising length.
