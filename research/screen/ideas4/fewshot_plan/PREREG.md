# Pre-registration, agent fewshot_plan, round 4 (5 Oct 2026). Written before the first evaluation.
tau2 only (SOPBench has no gold action lists to learn from, no pooled number), seed 0 of new_procedures plus new_domain.
Level L1 (cheap LLM plan on paper, no tools, no environment). Gold actions of TRAINING tasks only, enforced by gold_of(case, train_set) assertion.
A: ridge (alpha by grouped CV) from the 9 existing plan3 zero-shot features to gold [agent action count, agent write count, all action count]
   fitted on training tasks of the fold, training rows out-of-fold (GroupKFold by procedure), test rows from the full fit. No LLM calls.
B: gemini-3.8-flash plan (thinking low, temperature 0) with 5 retrieved training tasks (bge cosine, same domain if the training split has it) and their gold
   tool-call lists in the prompt, real tool list. Training tasks get the same kind of plan with examples from the training pool minus their own procedure
   (new_domain: same domain only), so train and test features have the same form. 9 plan features as plan_tools.
Variants (8 columns): A (ridge on A only), A_adele (grouped ridge ADELE+A), B (grouped ridge on B only), B_adele (ADELE+B),
c_lad (reference, copy of combo, rerun on seed 0), c_lad_A (ladder, extra columns ADELE+PLAN+A), c_lad_B (ADELE+PLAN+B), c_lad_Brep (ADELE+B, PLAN replaced).
Metrics: shared harness, pooled over the tau2 scenarios only (two, not four), paired CI vs adele and vs c_lad.
