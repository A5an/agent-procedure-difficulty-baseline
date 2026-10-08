Pre-registration small_model (written before features existed or any evaluation), 6 Oct 2026.
Model Qwen3-8B-Base MLX 8 bit, one forward pass per case with KV-cached domain prefix, no outcome information.
Variants: v_a_clad_sm = c_lad families (ADELE, PLAN, CMP on sop) + SM (sop: sm_margin, sm_ent, sm_ptgt, sm_rank_frac; tau2: sm_wr_margin, sm_ent, sm_pmax, sm_p_write), common-penalty Ladder as in ablation/run.py.
v_b = v_a + PRI (sop only: sm_pol_nll, sm_pol_nll_max, sm_msg_nll). v_c = v_a with per-family penalties (FamLadder). d_sm_only = SM(+PRI) alone in the ladder (diagnostic).
Prompt deviation: only the target action's policy is in the prompt (domain prefix = role, principles, tool list with descriptions); other actions' policies omitted for KV sharing. tau2 uses the scenario text as the customer message.
Leave-one-out salience not run unless the analysis shows signal. Single run, no tuning.
