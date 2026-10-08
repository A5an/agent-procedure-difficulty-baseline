import pandas as pd, json, re, numpy as np
from ceil import load, pooled
s=load("sopbench")
s["pol"]=s.text.map(lambda x: re.search(r"Policy for this action:\n(.*?)\nCustomer message:", x, re.S).group(1))
print("distinct policy texts", s.pol.nunique(), "cases", len(s))
print("cases per policy text", s.groupby("pol").size().value_counts().sort_index().to_dict())
s["polmean"]=s.groupby("pol").b.transform("mean")
print("policy-text mean oracle", pooled(s,"polmean"))
s["pollab"]=s.groupby(["pol","should_succeed"]).b.transform("mean")
print("policy-text x label oracle", pooled(s,"pollab"))
# label composition per policy text
g=s.groupby("pol").should_succeed.agg(["sum","count"])
print(pd.crosstab(g["sum"],g["count"]))
# within policy text, residual variance share
s["bd"]=s.b-s.groupby("domain").b.transform("mean")
tot=(s.bd**2).sum()
print("within pol share",((s.b-s.polmean)**2).sum()/tot, "within pol x label", ((s.b-s.pollab)**2).sum()/tot)
