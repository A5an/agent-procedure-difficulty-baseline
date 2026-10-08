# Swarm agent `probe`, round 9 (6-7 Oct 2026)

Saved by the main session from the agent's final message.

# Report: `probe`, round 9 (hidden-state probe, Qwen3-8B-Base, L1)

**Verdict in short:** the pre-registered primary hypothesis fails. c_lad plus the probe (p1) reaches 0.244 pooled against c_lad's 0.258 (−0.014, n.s.). The only positive signal is on banking, a single 97-case benchmark with no c_lad.

## Idea and level
- **Level:** L1, local white-box. Qwen3-8B-Base 8-bit MLX reads policy, tools, request and a fixed cue (no generation, no agent run).
- **Prompt:** cached domain prefix, then the SOPBench per-action policy header, then the request, then the cue "Assistant's first step:".
- **What was extracted per case:**
  - the last-token state at layers 12, 16, 20, 24, 28 and 32;
  - the mean state over the request tokens;
  - the state at the end of the policy-only part of the prompt;
  - NLL of the request tokens with the policy and without it;
  - token counts.
- **Scalars computed on the same pass:** shift = ‖last − policy-only state‖ per layer (I2), and request PMI = NLL without policy − NLL with policy (OD1).
- **Leakage:** the extraction script reads no outcome or IRT value. The layer was fixed in PREREG.md before any label was touched (last token, layer 24).
- **Prompts per benchmark:**
  - SOPBench: role, principles and tool descriptions as in `small_model`, plus the action policy.
  - tau2: domain policy, tool JSON and principles.
  - Banking: the 697 knowledge-base documents are not in the prompt. The prefix is the Rho-Bank policy header, shell instructions, the document-group names, 3 tool names and the scenario.

## Variants as pre-registered
All variants use the ladder lad2 built exactly as `ideas8/textcount/run.py` builds t0_clad.
- **t0_clad:** reference, reproduces 0.258.
- **p0_len:** t0_clad + 2 length columns (log request tokens, log policy-header tokens).
- **p1_probe (primary):** t0_clad + 16 PCs of the layer-24 last-token state. The PCA is fitted on all cases of the benchmark, unlabelled, and is **transductive**.
- **p3_scalars:** t0_clad + 8 scalars (6 shifts, request NLL, PMI).
- **p4_centred:** t0_clad + 16 PCs of the state centred by group. The group is the procedure on SOPBench and the domain on tau2, because every tau2 task is its own procedure. This is **transductive** and unlabelled.

Primary hypothesis: p1 > t0_clad on the pooled four-scenario rho, with a paired CI that excludes 0. The adele+probe variant was dropped to keep four.

## Cost
- Inference was one GPU job over 1,205 cases (830 SOPBench, 278 tau2, 97 banking). It finished with no crash.
- The restarted run took 1,072 s for 1,150 cases, about 1 s per case. The first, aborted part was about 5 s per case because the machine was swapping (swap about 15 GB of 16 GB used, another 10 GB process present). I killed only my own process and restarted it with an MLX cache limit. Output is resumable and was reused.
- States are about 190 MB of float16 npz. No `cpu_lock` was held during GPU work. The harness ran under `cpu_lock` for about 3 minutes.

## Main table (pooled within-domain case-level rho, paired CI over procedures)

| variant | pool4 | sop-proc | sop-dom | tau2-proc | tau2-dom | vs adele | vs c_lad | vs p0_len | procedure level, SOPBench new domain |
|---|---|---|---|---|---|---|---|---|---|
| adele | 0.131 | .120 | .125 | .128 | .150 | | −0.127 [−.172, −.071] | | 0.231 |
| t0_clad | 0.258 | .237 | .212 | .298 | .285 | +0.127 [+.071, +.172] | | +0.009 [−.008, +.020] | 0.486 |
| p0_len | 0.249 | .241 | .208 | .293 | .254 | +0.118 [+.065, +.170] | −0.009 [−.020, +.008] | | 0.443 |
| **p1_probe** | **0.244** [.138, .335] | .215 | .209 | .313 | .238 | +0.113 [+.056, +.164] | **−0.014 [−.031, +.025]** | −0.005 [−.027, +.027] | 0.346 (vs c_lad −0.140 [−.295, +.035]) |
| p3_scalars | 0.257 [.153, .352] | .263 | .243 | .282 | .242 | +0.127 [+.063, +.194] | −0.001 [−.025, +.033] | +0.008 [−.021, +.044] | 0.524 [.264, .681] (vs c_lad +0.038 [−.126, +.209]; vs adele +0.293 [+.035, +.521]) |
| p4_centred | 0.238 [.129, .344] | .247 | .221 | .305 | .178 | +0.107 [+.049, +.170] | −0.020 [−.053, +.036] | −0.011 [−.038, +.034] | 0.468 (vs c_lad −0.018 [−.098, +.079]) |

Full lines are in `tc_score_main.txt`.
- The probe adds nothing over c_lad, and nothing over the length control.
- Individual scenarios move by at most ±0.05 to ±0.1 and all CIs include 0. p3 is +0.026 / +0.031 on SOPBench, p4 on tau2-dom is −0.107 [−.174, +.061].
- The p3 procedure-level 0.524 was not the pre-registered main hypothesis. Its CI against c_lad includes 0.

## Banking (fresh benchmark, never used to choose anything)
- No PLAN or CMP features exist for banking, so c_lad cannot be built there.
- **Harness, random_cases, 5 seeds × 5 folds** (Spearman of held-out predicted difficulty with b, higher b = harder):

| variant | seed mean | seed-averaged case-level |
|---|---|---|
| adele | 0.431 | 0.521 |
| q0 ladder(ADeLe) | 0.596 | 0.603 |
| q1 + probe PCs | 0.669 | 0.680 |
| q2 + length | 0.590 | 0.601 |

  - q1 − q0 = +0.077, bootstrap over cases [−0.000, +0.157].
  - q1 − q2 (probe vs length control) = +0.079 [+0.001, +0.161].
  - 97 cases, one domain, and a tiny ladder. It is borderline and not a pre-registered primary result.
- **Single-feature within-banking Spearman with b:**
  - log request tokens: +0.544.
  - PMI: −0.489.
  - Centred mean-request PC1 at layer 24: +0.483.
  - Centred last-token PC1 at layer 24: +0.364.
  - Request NLL: −0.126.
  - Shifts: −0.21 to +0.10.
- **Ridge probe** (PCA64 of the centred layer-24 state, trained on SOPBench+tau2 with b z-scored per benchmark, joint unlabelled PCA basis): +0.278 on banking (p = 0.01). That is below the length variable's +0.544.
- **Within-banking 5-fold ridge:** probe +0.400, length-only +0.513.
- Banking difficulty is largely long, complex scenarios. The probe does not beat length there.

## Diagnostics
All diagnostics are in `diag.txt`.
- **SOPBench within-procedure Spearman with b** (mean over 44 procedures):
  - request NLL −0.231 (the only reasonably clear scalar; easier-to-predict requests are harder);
  - PMI +0.061;
  - log request tokens −0.095;
  - shifts −0.05 to +0.10 (layers 16 and 20 reach +0.10);
  - centred last-token PCs at layer 24: PC1 to PC4 = −0.03, −0.12, +0.08, −0.14;
  - centred mean-request PC1 at layer 24: −0.220.

  Centred PCs of the last-token state are 0 to 0.16 in absolute value at every layer (max +0.164 at layer 32, PC3).
- **Per-layer diagnostic only, never used for selection.** The 24 PC values per benchmark are an exploratory scan, so a lone 0.2 to 0.3 value is not evidence. The layer-12 and 16 pattern on tau2 and banking (PC1 about +0.2 to +0.37) is likely about request length and style.
- **Ridge probe, leave-one-group-out, within-group Spearman with b:**
  - SOPBench: probe +0.145, length-only ridge +0.138, length-residualised probe +0.124. Pooled within-domain rho is +0.146, length +0.105, residualised +0.151.
  - tau2 (leave-one-domain-out, 3 domains): probe −0.138, length-only +0.254. Length beats the probe.
- **Length explained:** variance-weighted R² of the first 16 PCs on [log request tokens, log policy tokens]:
  - raw: 0.11 on SOPBench, 0.47 on tau2 (PC1 R² = 0.70), 0.17 on banking;
  - centred: 0.02 on SOPBench and tau2, 0.17 on banking.

  So centring removes most length on SOPBench and tau2. Banking has a single group, so centring changes nothing there, and its 0.17 is the share left from length and domain.

## Verdict
1. **The pre-registered primary hypothesis fails.** c_lad + probe (0.244) is no better than c_lad (0.258) or c_lad + length (0.249). No scenario has a CI that excludes 0. The SOPBench within-procedure rho of the probe is about the same as that of a length-only ridge (+0.145 vs +0.138). The paper's AUC 0.88 for code correctness does not transfer to cross-procedure difficulty here.
2. **Banking gives the only hint** (+0.08 over the ADeLe ladder, CI touching 0, and the single request-length feature +0.54 explains much of it). A within-benchmark probe there is no better than length (+0.40 vs +0.51).
3. **Next candidate.** If you pursue something, use the request NLL, whose within-procedure SOPBench rho is about −0.23 and which p3 carries. It did not move the harness (+0.026 SOPBench). The procedure-level p3 (0.524 vs c_lad 0.486, n.s.) is a lead only if confirmed on HANDBOOK.

## Limitations
- The PCA and the centring are transductive but unlabelled. The ladder penalty is chosen on training tasks only. The 16-component size and layer 24 were fixed in advance, so other layers or sizes are untested.
- The probe was a single linear head on a base model, 8-bit, with plain mean-centred PCA (no per-dimension scaling). Stronger models, an instruct model, a supervised head with inner-CV layer choice, or per-layer concatenation are not tested. The SOPBench policy header makes its prompt differ from the original agent prompt.
- tau2 has only 3 domains and 278 cases, so its CIs are wide (every tau2 CI above includes 0).
- Banking has no c_lad (no PLAN or CMP), a single domain, and 97 cases.
- The no-policy NLL for PMI is computed after a short conversation header only, and SOPBench has fewer than 30 cases per procedure.
- The README of `ideas9/BRIEF.md` was read. HANDBOOK was not opened.

## Paths
All under `research/screen/ideas9/probe/`:
- `PREREG.md`
- `extract.py` and `states/` (1,205 npz files), `extract.log`, `extract2.log`
- `feats.py`, `probe_*.csv`, `probec_*.csv`, `scal_*.csv`, `len_*.csv`
- `run.py`, `out/`, `out_banking/`, `run.log`, `run_banking.log`
- `score.py`, `tc_score_main.txt`, `tc_score_main.pkl`, `score.log`
- `diag.py`, `diag.txt`, `banking_harness.txt`

I wrote no REPORT.md, as instructed.
