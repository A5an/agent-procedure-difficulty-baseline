"""Explainer-style catalogue page. usage: python3 build2.py -> docs/zangir-2026-10-07-catalog/local.html"""
import json, html
from pathlib import Path
from build import load, cls, ROUNDS, BASE, CLAD, OUT, HEAD

HERE = Path(__file__).parent
recs = {}
for f in ["cat_a.json", "cat_b.json", "cat_c.json", "cat_d.json", "cat_e.json", "cat_f.json"]:
    for r in load(f):
        r["_v"] = cls(r); recs[r["id"]] = r

CH = {
 "ch1": "r0_adele r0_adele16 r0_lltm r0_human_time s_models_trees s_models_rank s_models_mirt s_models_agent_assessor s_models_ridge_ht s_models_lltm_ht s_rubric_proc s_rubric_ladder s_rubric_lara s_rubric_rpa r1_fmea r1_anchors r7_adele_plus s_rerun_forecast r1_pairwise r10_fc r10_fcx r10_ablate r10_opus r10_panel r10_fctype r10_fresh",
 "ch2": "r0_length r0_emb_ap r0_emb_ap_adele r0_ap_combined r0_knn_router r0_router_combo r0_amortized_mirt s_rerun_bge s_rerun_amortized_irt s_rerun_finetuned s_rerun_bpm r8_screens r8_textcount",
 "ch3": "r1_dryrun r3_plan_tau r3_plan_tools r3_paper_pilot r4_fewshot_plan r4_fewshot_A5 r4_code_plan r6_planners r4_user_scenario r7_tau_graph r9_gap r10_prior r10_over",
 "ch4": "r2_compile_exec r3_compile_consistency r2_compile_precheck r3_conditions_l0 r3_conditions_l2 r7_cond_ladder r2_steps",
 "ch5": "r1_headroom r6_msg_facts r6_synth_label r6_personas r7_case_signals r1_lupi s_pertype_subout r9_diag r10_ceiling",
 "ch6": "s_rerun_entropy s_rerun_probe r7_small_model r9_probe",
 "ch7": "r1_pilot r1_judge s_router_pilot r2_emulate r2_emulate_run r5_smart_emu r6_emu_feat r2_harness r2_cc_agent r5_smart_agents",
 "ch8": "r3_combo r4_ablation r7_entropy_combo r7_evo r1_modeltricks r10_zs",
 "ch9": "r0_constant s_clean_empty s_targets_shares s_targets_bb s_targets_passk s_transfer_decomp s_transfer_shares_irt s_transfer_overdisp s_pertype_ability s_pertype_shares s_pertype_routing r1_power r2_risk r2_rankstab r3_tiers r5_smart_doc r10_lobo r10_newpop",
 "ch10": "r1_lit r2_fresh r2_lit_exec r2_lit_design r6_lit6 r7_council_cog r7_council_formal r7_council_llm r7_discovery_lit r8_swarm r10_lit",
}
CH = {k: v.split() for k, v in CH.items()}
used = [i for v in CH.values() for i in v]
missing = set(recs) - set(used); dup = {i for i in used if used.count(i) > 1}
assert not missing and not dup and set(used) <= set(recs), (missing, dup, set(used) - set(recs))

E = lambda s: html.escape(str(s if s is not None else ""))
def f3(x, sign=True): return ("+" if sign and x > 0 else "") + f"{x:.3f}".replace("-", "−")
def ci(c): return f" [{f3(c[0])}, {f3(c[1])}]" if c else ""
def rlab(k): r = next(x for x in ROUNDS if x[0] == k); return f"{r[1]} · {r[2]}"
COL = {"sig": "ok", "ns": "gen", "flat": "mut", "worse": "bad", "base": "mut"}
CC = {"ok": "ok", "gen": "gen", "mut": "muted", "bad": "bad"}

def strip(ids):
    P = sorted([recs[i] for i in ids if recs[i]["_v"][2] != "other"], key=lambda r: -r["pooled"])
    if not P: return ""
    x0, x1, lo, hi = 330, 690, -0.05, 0.40
    X = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    rh = 26; H = 52 + rh * len(P) + 10
    o = [f'<figure><div class="fw"><svg viewBox="0 0 760 {H}" role="img" aria-label="Результаты идей главы относительно бейзлайна и c_lad">']
    for v in [0, 0.1, 0.2, 0.3, 0.4]:
        o.append(f'<line x1="{X(v):.1f}" y1="40" x2="{X(v):.1f}" y2="{H-8}" class="gl"/><text x="{X(v):.1f}" y="34" text-anchor="middle" class="tm">{v:.1f}</text>')
    o.append(f'<line x1="{X(BASE):.1f}" y1="40" x2="{X(BASE):.1f}" y2="{H-8}" class="dash"/><text x="{X(BASE):.1f}" y="16" text-anchor="middle" class="tm">бейзлайн</text>')
    o.append(f'<line x1="{X(CLAD):.1f}" y1="40" x2="{X(CLAD):.1f}" y2="{H-8}" class="dash"/><text x="{X(CLAD):.1f}" y="16" text-anchor="middle" class="tm">c_lad</text>')
    for k, r in enumerate(P):
        y = 56 + k * rh; ref = CLAD if r.get("vs_clad") is not None and r["id"] != "r3_combo" else BASE
        if r["id"] == "r0_adele": ref = BASE
        c = COL[r["_v"][2]]; nm = r["name"] if len(r["name"]) <= 40 else r["name"][:38] + "…"
        o.append(f'<text x="{x0-10}" y="{y+4}" text-anchor="end" class="t" style="font-size:13px">{E(nm)}</text>')
        o.append(f'<line x1="{X(ref):.1f}" y1="{y}" x2="{X(r["pooled"]):.1f}" y2="{y}" class="s-{c}" style="stroke-width:3"/>')
        o.append(f'<circle cx="{X(r["pooled"]):.1f}" cy="{y}" r="6" class="c-{CC[c]}"/>')
        o.append(f'<text x="{max(X(r["pooled"]), X(ref))+12:.1f}" y="{y+4}" class="tm" style="font-size:12px">{r["pooled"]:.3f}</text>')
    o.append('</svg></div><figcaption>Все идеи главы, которые мерили на основной метрике. Линия идёт от того, с чем сравнивали (бейзлайн или c_lad, если идею добавляли к нему), до результата. Зелёный значит значимо лучше, синий лучше без значимости, серый на уровне, красный хуже.</figcaption></figure>')
    return "\n".join(o)

def rec_html(r):
    v = r["_v"]; pool = v[2] != "other" and r.get("pooled") is not None
    nums = ""
    if pool:
        if r.get("vs_clad") is not None and r["id"] != "r3_combo":
            nums = f'ρ {r["pooled"]:.3f} · к c_lad {f3(r["vs_clad"])}'
        elif r["id"] == "r0_adele":
            nums = f'ρ {r["pooled"]:.3f}'
        else:
            d = r.get("vs_base"); nums = f'ρ {r["pooled"]:.3f} · к бейзлайну ' + (f3(d) if d is not None else "≈ " + f3(round(r["pooled"] - BASE, 3)))
    elif r.get("proc") is not None:
        nums = f'по процедурам {r["proc"]:.3f}'
    head = f'<summary><span class="nm">{E(r["name"])}</span><span class="mono">{E(nums)}</span><span class="pill {v[1]}">{E(v[0])}</span></summary>'
    body = [f'<p><b>{E(r["short"])}</b></p>', f'<p>{E(r["detail"])}</p>']
    if pool:
        line = f'ρ {r["pooled"]:.3f}'
        if r.get("vs_base") is not None: line += f' · к бейзлайну {f3(r["vs_base"])}{ci(r.get("vs_base_ci"))}'
        elif r.get("vs_base_ci"): line += f' · к бейзлайну ≈ {f3(round(r["pooled"] - BASE, 3))}{ci(r["vs_base_ci"])}'
        if r.get("vs_clad") is not None: line += f' · к c_lad {f3(r["vs_clad"])}{ci(r.get("vs_clad_ci"))}'
        if r.get("proc") is not None: line += f' · по процедурам {r["proc"]:.3f}'
        sc = r.get("scen")
        if sc and any(x is not None for x in sc):
            n = ["SOP проц.", "SOP домен", "τ² проц.", "τ² домен"]
            line += " · " + ", ".join(f"{n[i]} {x:.3f}" for i, x in enumerate(sc) if x is not None)
        body.append(f'<p class="small" style="font-family:var(--mono)">{E(line)}</p>')
    if r.get("other_metric"):
        body.append(f'<p class="small"><span class="muted">{"Ещё замеры" if pool else "Замер"}:</span> {E(r["other_metric"])}</p>')
    if r.get("variants"):
        def vrow(x):
            pp = "" if x.get("pooled") is None else "%.3f" % x["pooled"]
            vb = "" if x.get("vs_base") is None else f3(x["vs_base"]) + ci(x.get("vs_base_ci"))
            vc = "" if x.get("vs_clad") is None else f3(x["vs_clad"])
            return f'<tr><td>{E(x.get("name"))}</td><td class="n">{pp}</td><td class="n">{vb}</td><td class="n">{vc}</td><td>{E(x.get("note"))}</td></tr>'
        rows = "".join(vrow(x) for x in r["variants"])
        body.append(f'<div class="vw"><table class="var"><thead><tr><th>Вариант</th><th>ρ</th><th>к бейзлайну</th><th>к c_lad</th><th>Заметка</th></tr></thead><tbody>{rows}</tbody></table></div>')
    if r.get("note"): body.append(f'<p class="small"><span class="muted">Оговорки:</span> {E(r["note"])}</p>')
    body.append(f'<p class="small muted">{E(rlab(r["round"]))} · уровень {E(r.get("level"))} · папка <code>{E(r.get("folder"))}</code></p>')
    return f'<details class="rec">{head}<div class="in">{"".join(body)}</div></details>'

def lst(ids):
    return f'<p class="small muted" style="margin-top:22px">Все идеи главы ({len(ids)}). Нажмите, чтобы раскрыть подробности и числа.</p><div class="recs">' + "\n".join(rec_html(recs[i]) for i in ids) + "</div>"

body = (HERE / "explained_body.html").read_text()
for k, ids in CH.items():
    body = body.replace(f"%%STRIP:{k}%%", strip(ids)).replace(f"%%LIST:{k}%%", lst(ids))
assert "%%" not in body
head = HEAD.replace("<title>Все идеи и замеры</title>", "<title>Все идеи понятно</title>")
(OUT / "local.html").write_text(head + "\n" + body)
print("written", len(recs), "records,", len(head + body), "bytes")
