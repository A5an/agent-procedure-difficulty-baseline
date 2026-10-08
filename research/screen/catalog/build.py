"""Build docs/zangir-2026-10-07-catalog/local.html from catalog/*.json. usage: python3 build.py"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, re, html
from pathlib import Path
HERE = Path(__file__).parent
OUT = Path(ZR.DOCS + "/zangir-2026-10-07-catalog"); OUT.mkdir(exist_ok=True)
HEAD = Path(ZR.DOCS + "/zangir-2026-10-06-explainer/local.html").read_text().split("\n")[:83]
HEAD = "\n".join(HEAD).replace("<title>Предсказать до запуска</title>", "<title>Все идеи и замеры</title>")
BASE, CLAD = 0.131, 0.258
ROUNDS = [("r0", "Бейзлайн", "28 сен"), ("s", "Методы из статей", "4 окт"), ("r1", "Ночной раунд", "4–5 окт"),
          ("r2", "Раунд 2", "5 окт"), ("r3", "Раунд 3", "5 окт"), ("r4", "Раунд 4", "5 окт"), ("r5", "Умные модели", "6 окт"),
          ("r6", "Раунд 6", "6 окт"), ("r7", "Раунд 7", "6 окт"), ("r8", "Рой", "6 окт"), ("r9", "Раунд 9", "7 окт"), ("r10", "Ночь за 0.6", "8 окт")]
RORD = {k: i for i, (k, _, _) in enumerate(ROUNDS)}

def load(name):
    p = HERE / "edited" / name
    return json.loads((p if p.exists() else HERE / name).read_text())

recs = []
for f in ["cat_a.json", "cat_b.json", "cat_c.json", "cat_d.json", "cat_e.json", "cat_f.json"]:
    recs += load(f)
recs.sort(key=lambda r: (RORD.get(r["round"], 99)))

def cls(r):
    """display verdict: (label, css class, group)"""
    if r["id"] == "r0_adele": return ("это бейзлайн", "p-old", "base")
    if r.get("metric_type") != "pooled4" or r.get("pooled") is None: return ("не на основной метрике", "p-old", "other")
    if r["id"] == "r7_evo": return ("не подтвердилось на свежих данных", "p-no", "worse")
    vc, ci = r.get("vs_clad"), r.get("vs_clad_ci")
    if vc is not None:
        if ci and ci[0] > 0: return ("значимо лучше c_lad", "p-ok", "sig")
        if vc > 0.01: return ("чуть лучше c_lad, не значимо", "p-mid", "ns")
        if vc >= -0.01: return ("на уровне c_lad", "p-old", "flat")
        return ("хуже c_lad", "p-no", "worse")
    d = r.get("vs_base"); d = d if d is not None else r["pooled"] - BASE
    ci = r.get("vs_base_ci")
    if ci and ci[0] > 0: return ("значимо лучше бейзлайна", "p-ok", "sig")
    if d > 0.01: return ("лучше бейзлайна, не значимо", "p-mid", "ns")
    if d >= -0.01: return ("на уровне бейзлайна", "p-old", "flat")
    return ("хуже бейзлайна", "p-no", "worse")

for r in recs:
    r["_v"] = cls(r)
    if r.get("metric_type") == "pooled4" and r.get("pooled") is not None and r.get("vs_base") is None:
        r["_dcalc"] = round(r["pooled"] - BASE, 3)
counts = {}
for r in recs: counts[r["_v"][2]] = counts.get(r["_v"][2], 0) + 1
npool = sum(1 for r in recs if r["_v"][2] not in ("other",))
print(len(recs), "records,", npool, "on the main metric", counts)

# bar chart of every pooled record
P = sorted([r for r in recs if r["_v"][2] != "other"], key=lambda r: -r["pooled"])
x0, x1, lo, hi, W = 330, 700, -0.05, 0.40, 760
X = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
rowh = 19; H = 60 + rowh * len(P) + 20
col = {"sig": "c-ok", "ns": "c-gen", "flat": "c-muted", "worse": "c-bad", "base": "c-ink"}
svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Итог по четырём сценариям для каждой идеи, от лучшей к худшей">']
for v in [0, 0.1, 0.2, 0.3, 0.4]:
    svg.append(f'<line x1="{X(v):.1f}" y1="40" x2="{X(v):.1f}" y2="{H-14}" class="gl"/><text x="{X(v):.1f}" y="32" text-anchor="middle" class="tm">{v:.1f}</text>')
for v, lab in [(BASE, "бейзлайн 0.131"), (CLAD, "c_lad 0.258")]:
    svg.append(f'<line x1="{X(v):.1f}" y1="40" x2="{X(v):.1f}" y2="{H-14}" class="dash"/><text x="{X(v):.1f}" y="16" text-anchor="middle" class="tm">{lab}</text>')
for i, r in enumerate(P):
    y = 50 + i * rowh; nm = html.escape(r["name"] if len(r["name"]) <= 44 else r["name"][:42] + "…")
    a, b = sorted([X(0), X(r["pooled"])])
    rl = next(x[1] for x in ROUNDS if x[0] == r["round"])
    svg.append(f'<text x="{x0-8}" y="{y+12}" text-anchor="end" class="t" style="font-size:12.5px">{nm}</text>'
               f'<rect x="{a:.1f}" y="{y+2}" width="{max(b-a,1.5):.1f}" height="13" rx="2" class="{col[r["_v"][2]]}"/>'
               f'<text x="{max(b, X(0))+6:.1f}" y="{y+12}" class="tm" style="font-size:11.5px">{r["pooled"]:.3f} · {html.escape(rl)}</text>')
svg.append("</svg>")
SVG = "\n".join(svg)

swarm = (HERE / "swarm_ideas.js").read_text()
data = json.dumps([{k: v for k, v in r.items()} for r in recs], ensure_ascii=False)
rounds_js = json.dumps(ROUNDS, ensure_ascii=False)
body = (HERE / "page_body.html").read_text()
body = body.replace("%%SVG%%", SVG).replace("%%DATA%%", data).replace("%%ROUNDS%%", rounds_js).replace("%%SWARM%%", swarm)
body = body.replace("%%N%%", str(len(recs))).replace("%%NPOOL%%", str(npool))
(OUT / "spisok.html").write_text(HEAD + "\n" + body)
print("written", OUT / "spisok.html", len(HEAD + body))
