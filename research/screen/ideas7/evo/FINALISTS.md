# FINALISTS (written before any held-out look)

Selection rule: see finalize.py docstring. Search-split scores are biased upward (selected maxima of ~500 candidates).

Top-12 table (id, island, search score, fresh-shuffle score, combined):

- id 485 msgpol round 11: search +0.0933, fresh +0.0964, combined +0.0948
- id 464 policy round 11: search +0.0882, fresh +0.0874, combined +0.0878
- id 473 policy round 11: search +0.0903, fresh +0.0812, combined +0.0857
- id 434 msgpol round 10: search +0.0860, fresh +0.0848, combined +0.0854
- id 421 policy round 10: search +0.0869, fresh +0.0826, combined +0.0847
- id 427 policy round 10: search +0.0881, fresh +0.0809, combined +0.0845
- id 390 policy round 9: search +0.0861, fresh +0.0825, combined +0.0843
- id 467 policy round 11: search +0.0869, fresh +0.0812, combined +0.0840
- id 459 plantools round 10: search +0.0861, fresh +0.0811, combined +0.0836
- id 432 policy round 10: search +0.0874, fresh +0.0797, combined +0.0835
- id 431 policy round 10: search +0.0869, fresh +0.0770, combined +0.0819
- id 379 policy round 9: search +0.0860, fresh +0.0777, combined +0.0819

Pre-declared held-out and full-protocol evaluation (done once each after this file exists):
- held-out A: same scoring as the search (ridge fitted by CV inside the held-out domains), gain over c_lad alone, per domain and mean over the 4 held-out domains (telecom reported separately, only 3 procedures)
- held-out B (transfer): ridge fitted on all search cases, applied to held-out cases, same gain definition
- full protocol harness: columns evo_fK (plain ladder with c_lad table + finalist features, domain-centred) and evo_fK_fam (per-family penalty ladder), paired CIs vs c_lad and adele; primary column = evo_fK


## Finalist 1: archive id 485 (island msgpol, round 11), search +0.0933, fresh +0.0964
Features kept: ['disagree_p_ref', 'cond_settled_bal', 'plan_ambig_spread', 'msg_vagueness', 'msg_entity_density', 'most_diff_check', 'tool_scope_diff', 'bench_friction']
Idea: Disagreement between LLM descriptions (dryrun, plans, program return) combined with hardest check skip risk, entity density, and cross-benchmark friction.
```python
# idea: Disagreement between LLM descriptions (dryrun, plans, program return) combined with hardest check skip risk, entity density, and cross-benchmark friction.
import re

def features(record) -> dict:
    is_sop = record.get("benchmark") == "sopbench"
    dry = record.get("dryrun") or {}
    parsed = dry.get("parsed") or {}
    probs = parsed.get("outcome_probs") or {}
    p_perf = float(probs.get("perform", dry.get("p_perform") or 0.0) or 0.0)
    p_ref = float(probs.get("refuse", dry.get("p_refuse") or 0.0) or 0.0)

    plans = record.get("plans") or []
    plan_outcomes = [str(p.get("outcome", "")).lower() for p in plans if isinstance(p, dict)]
    n_plans = max(1, len(plan_outcomes))
    plan_ref_rate = sum(1.0 for o in plan_outcomes if any(k in o for k in ("refus", "reject", "fail"))) / n_plans
    
    # Twist: multi-description outcome disagreement (dryrun vs plan vs program refusal)
    prog = record.get("program") or ""
    prog_has_refuse = 1.0 if any(k in prog.lower() for k in ("return false", "raise", "refuse", "return none")) else 0.0
    desc_disagree = max(abs(p_ref - plan_ref_rate), abs(p_ref - prog_has_refuse) if prog else 0.0)

    cond_list = parsed.get("conditions") or []
    n_settled = sum(1.0 for c in cond_list if any(k in str(c.get("status", "")).lower() for k in ("satisfied", "violated")))
    n_db = sum(1.0 for c in cond_list if any(k in str(c.get("status", "")).lower() for k in ("system", "data")))
    cond_settled_bal = (n_settled - n_db) / (float(len(cond_list)) + 1.0)

    ambigs = [float(p.get("ambiguity", 0) or 0) for p in plans if isinstance(p, dict)]
    plan_ambig_spread = (max(ambigs) - min(ambigs)) if len(ambigs) > 1 else (ambigs[0] if ambigs else 0.0)

    msg = record.get("message") or ""
    hedge_words = ("maybe", "perhaps", "might", "could", "around", "approx", "not sure", "either", "or so", "any", "possibly")
    vague_count = sum(1.0 for w in hedge_words if re.search(r"\b" + re.escape(w) + r"\b", msg.lower()))
    dry_ambig = float(dry.get("ambiguous_request") or parsed.get("ambiguous_request") or 0.0)
    msg_vagueness = vague_count + dry_ambig * 2.0

    tokens = re.findall(r"\b[A-Za-z0-9_-]+\b", msg)
    entity_tokens = sum(1.0 for t in tokens if (any(c.isdigit() for c in t) and any(c.isalpha() for c in t)) or (t.isdigit() and len(t) >= 2))
    msg_entity_density = entity_tokens / (float(len(tokens)) + 1.0)

    conds = record.get("conditions") or []
    most_diff_check = 0.0
    for idx, c in enumerate(conds):
        pskip = max(float(c.get("skip_if_must_perform_pred") or 0.0), float(c.get("skip_if_must_refuse_pred") or 0.0))
        st = str(c.get("step_type", "")).lower()
        type_w = 1.4 if st in ("relation", "datetime") else (1.15 if st == "existence" else 1.0)
        pos_w = 1.0 + (idx / float(len(conds))) * 0.35 if conds else 1.0
        score = pskip * type_w * pos_w
        if score > most_diff_check:
            most_diff_check = score

    cond_tools = {t for c in conds for t in (c.get("step_tools") or [])}
    plan_tools = {s.get("tool") for p in plans for s in (p.get("plan") or []) if s.get("tool")}
    tool_scope_diff = float(len(cond_tools ^ plan_tools)) if (cond_tools or plan_tools) else 0.0

    scen = record.get("scenario") or {}
    if not is_sop and scen:
        touched = len(scen.get("policy_clauses_touched") or [])
        avg_steps = sum(len(p.get("plan") or []) for p in plans) / n_plans
        trap_mult = 1.0 + float(scen.get("n_mind_changes") or 0) * 1.5 + float(scen.get("policy_forbidden_request") or 0) * 2.0
        bench_friction = abs(float(touched) - avg_steps) * trap_mult
    else:
        bench_friction = (n_db / (float(len(cond_list)) + 1.0)) * (1.0 + p_perf)

    return {
        "disagree_p_ref": float(desc_disagree),
        "cond_settled_bal": float(cond_settled_bal),
        "plan_ambig_spread": float(plan_ambig_spread),
        "msg_vagueness": float(msg_vagueness),
        "msg_entity_density": float(msg_entity_density),
        "most_diff_check": float(most_diff_check),
        "tool_scope_diff": float(tool_scope_diff),
        "bench_friction": float(bench_friction),
    }
```

## Finalist 2: archive id 459 (island plantools, round 10), search +0.0861, fresh +0.0811
Features kept: ['tool_sym_diff', 'max_write_req', 'max_late_sys_skip', 'sys_vs_msg_ratio', 'sys_entropy_inter', 'vagueness_score', 'entity_load', 'tau_friction']
Idea: Combine tool divergence, write schema load, position-weighted skip risk, message-satisfied vs system-needed condition contrast, entropy interaction, vagueness, entity load, and tau2 friction.
```python
# idea: Combine tool divergence, write schema load, position-weighted skip risk, message-satisfied vs system-needed condition contrast, entropy interaction, vagueness, entity load, and tau2 friction.
import re

def features(record):
    bm = record.get("benchmark") or ""
    plans = record.get("plans") or []
    plan_steps = [s for p in plans for s in (p.get("plan") or []) if isinstance(s, dict)]
    plan_tools = {s.get("tool") for s in plan_steps if s.get("tool")}

    prog = record.get("program") or ""
    prog_tools = set(re.findall(r"tools\.([a-zA-Z0-9_]+)", prog)) if prog else set()

    if bm == "sopbench" and prog:
        tool_diff = float(len(plan_tools ^ prog_tools))
    elif bm == "tau2" and len(plans) > 1:
        p_sets = [{s.get("tool") for s in (p.get("plan") or []) if s.get("tool")} for p in plans]
        tool_diff = float(len(p_sets[0] ^ p_sets[1])) if len(p_sets) >= 2 else 0.0
    else:
        tool_diff = 0.0

    tools = record.get("tools") or []
    tool_defs = {t.get("name"): t for t in tools if isinstance(t, dict)}
    write_tools = [s.get("tool") for s in plan_steps if s.get("writes_db")]
    write_reqs = [len(((tool_defs.get(t) or {}).get("parameters") or {}).get("required") or []) for t in write_tools]
    max_write_req = float(max(write_reqs)) if write_reqs else 0.0

    dry = record.get("dryrun") or {}
    dry_parsed = (dry.get("parsed") or {}).get("conditions") or []
    sys_needed = sum(1 for c in dry_parsed if c.get("status") == "needs_system_data")
    msg_sat = sum(1 for c in dry_parsed if c.get("status") == "satisfied_by_request")
    n_parsed = max(1.0, float(len(dry_parsed)))
    # Twist: ratio of conditions needing db lookup vs conditions already satisfied by the message
    sys_vs_msg_ratio = float((sys_needed - msg_sat) / n_parsed) if bm == "sopbench" else 0.0

    conds = record.get("conditions") or []
    n_conds = max(1.0, float(len(conds)))
    late_sys_skips = [
        float(c.get("skip_if_must_perform_pred") or 0.0) * (1.0 + (i / n_conds))
        for i, c in enumerate(conds)
        if c.get("step_type") in ("relation", "datetime", "existence")
    ]
    max_late_sys_skip = float(max(late_sys_skips)) if late_sys_skips else 0.0

    ent = float(dry.get("decision_entropy") or 0.0)
    sys_ratio = (sys_needed / n_parsed) if bm == "sopbench" else 0.0
    sys_entropy_inter = ent * sys_ratio

    msg = record.get("message") or ""
    vague_pat = r"\b(maybe|perhaps|possibly|if possible|either|around|approx|approximately|roughly|flexible|asap|anytime|whenever|preferably|prefer|might|could|or so|unsure|not sure)\b"
    vague_matches = len(re.findall(vague_pat, msg, re.IGNORECASE))
    plan_ambig = max([p.get("ambiguity") or 0 for p in plans] or [0])
    dry_ambig = 1.0 if (dry.get("ambiguous_request") or dry.get("ambiguous_policy")) else 0.0
    scen = record.get("scenario") or {}
    scen_ambig = float(1.0 if scen.get("ambiguous_reference") else 0.0) if bm == "tau2" else 0.0
    vagueness_score = float(max(plan_ambig, dry_ambig)) + scen_ambig + 0.5 * min(4.0, float(vague_matches))

    id_tokens = set(re.findall(r"\b(?:[A-Z0-9]{4,12}|\d{4}-\d{2}-\d{2}|\d{3}-\d{3}-\d{4}|[\w\.-]+@[\w\.-]+)\b", msg))
    entity_load = float(min(6, len(id_tokens)))

    tau_friction = float((scen.get("n_goals") or 0) + 2 * (scen.get("n_mind_changes") or 0) + (1 if scen.get("withholds_info") else 0) + (2 if scen.get("policy_forbidden_request") else 0)) if bm == "tau2" else 0.0

    return {
        "tool_sym_diff": tool_diff,
        "max_write_req": max_write_req,
        "max_late_sys_skip": max_late_sys_skip,
        "sys_vs_msg_ratio": sys_vs_msg_ratio,
        "sys_entropy_inter": float(sys_entropy_inter),
        "vagueness_score": float(vagueness_score),
        "entity_load": entity_load,
        "tau_friction": tau_friction,
    }
```