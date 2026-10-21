"""Text features per condition. Everything comes from the policy text (and the tool list that is part of the prompt)
plus the LLM tags of tag_llm.py (text only). Nothing from the rule tree, step types or outcomes."""
import re, json, numpy as np, pandas as pd
from parse import st, pol
d = pd.read_csv("conds.csv")
UN = json.load(open("units.json"))
tags = json.load(open("tags.json"))
CMP = r"greater|less|more than|fewer|at least|at most|no more|exceed|equal|higher|lower|above|below|minimum|maximum|within|between|>=|<=|<|>"
NEG = r"must not|\bnot\b|\bno\b|\bnever\b|different from|cannot|without|non-|\bnone\b|excluding|unless"
DATE = r"date|time|deadline|day|hour|night|month|year|expire|period|before|after|overlap|schedule|age\b|calendar|interaction"
REF = r"previously|above|the same|that user|their|defined as|as per|according to|such|respective|corresponding|this action|the one"
AUTH = r"logged in|authenticat|password|credentials|identification"
def toolnames(t):
    m = re.search(r"Available tools: (.*)$", t, re.S)
    return [x.strip() for x in m.group(1).split(",")] if m else []
STOP = {"get", "set", "internal", "check", "is", "has", "user", "info", "details"}
rows = []
for _, r in d.iterrows():
    tx = r.text
    tl = tx.lower()
    tools = toolnames(st[UN[r.unit][0]])
    words = set(re.findall(r"[a-z]+", tl))
    best = 0.0
    for tn in tools:
        toks = [x for x in tn.split("_") if x not in STOP]
        if len(toks) >= 1:
            best = max(best, np.mean([x in words or x + "s" in words or x.rstrip("s") in words for x in toks]))
    quoted = re.findall(r'"([A-Za-z_][A-Za-z_0-9]*)"', tx)
    f = dict(unit=r.unit, idx=r.idx)
    f["pos"] = r.idx; f["pos_rel"] = r.idx / max(r.n_cond - 1, 1); f["is_first"] = int(r.idx == 0); f["is_last"] = int(r.idx == r.n_cond - 1)
    f["from_end"] = r.n_cond - 1 - r.idx; f["n_cond"] = r.n_cond
    f["nchar"] = len(tx); f["nword"] = len(tx.split()); f["nsent"] = len(re.findall(r"[.!?](?:\s|$)", tx)); f["lognchar"] = np.log(len(tx))
    f["n_quoted_fields"] = len(quoted); f["n_distinct_fields"] = len(set(quoted)); f["has_field"] = int(len(quoted) > 0)
    f["tool_name_overlap"] = best; f["says_database"] = int("database" in tl); f["says_account"] = int(bool(re.search(r"account|record|listed|exist", tl)))
    f["neg"] = int(bool(re.search(NEG, tl))); f["n_neg"] = len(re.findall(NEG, tl)); f["bold_neg"] = int(bool(re.search(r"\*\*[^*]*(not|no)\b", tl)))
    f["in_or"] = int(r.is_or); f["n_alt"] = r.n_alt; f["nested"] = r.nested; f["seq"] = r.seq
    f["n_numbers"] = len(re.findall(r"\b\d+(?:\.\d+)?\b", tx)); f["has_number"] = int(f["n_numbers"] > 0)
    f["n_compare"] = len(re.findall(CMP, tl)); f["has_compare"] = int(f["n_compare"] > 0)
    f["date_words"] = len(re.findall(DATE, tl)); f["has_date"] = int(f["date_words"] > 0)
    f["date_arith_rx"] = int(bool(re.search(r"\d+\s*(day|night|hour|minute|month|year)|between .* and|overlap|no (earlier|later)", tl)))
    f["ref_words"] = len(re.findall(REF, tl)); f["has_ref"] = int(f["ref_words"] > 0)
    f["n_bold"] = len(re.findall(r"\*\*", tx)) // 2; f["is_auth_text"] = int(bool(re.search(AUTH, tl)))
    f["n_other_cond_shared_fields"] = 0
    f["must_have_style"] = int("must have" in tl or "must be" in tl)
    f["exists_style"] = int(bool(re.search(r"must exist|exist within|listed in|exists", tl)))
    f["n_commas"] = tx.count(",") ; f["n_parens"] = tx.count("(")
    f["n_and"] = len(re.findall(r"\band\b", tl)); f["n_or_word"] = len(re.findall(r"\bor\b", tl))
    tg = {t["idx"]: t for t in (tags.get(r.unit) or [])}.get(int(r.idx), {})
    for k in ["cross_ref", "date_arith", "calc", "n_lookups", "n_comparisons", "conditional", "vague", "precondition_style"]:
        f["llm_" + k] = float(tg.get(k, np.nan))
    rows.append(f)
F = pd.DataFrame(rows)
# shared fields with other conditions of the same policy
for u, g in d.groupby("unit"):
    qs = {int(r.idx): set(re.findall(r'"([A-Za-z_][A-Za-z_0-9]*)"', r.text)) for _, r in g.iterrows()}
    for i in qs:
        others = set().union(*[qs[j] for j in qs if j != i]) if len(qs) > 1 else set()
        F.loc[(F.unit == u) & (F.idx == i), "n_other_cond_shared_fields"] = len(qs[i] & others)
F["n_other_cond_shared_fields"] = F["n_other_cond_shared_fields"].astype(float)
for c in [c for c in F.columns if c.startswith("llm_")]:
    F[c] = F[c].fillna(F[c].median())
F.to_csv("cond_feats.csv", index=False)
print(F.shape, F.isna().sum().sum())
