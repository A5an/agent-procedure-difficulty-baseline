import pandas as pd
so=pd.read_csv("step_outcomes.csv.gz"); co=pd.read_csv("case_outcomes.csv.gz")
so=so.merge(co[["agent","case_id","success","should"]],on=["agent","case_id"])
print(len(so))
print(so.groupby("step_type").agg(n=("step_correct","size"),r=("step_correct","mean"),ver=("verified","mean")))
print(so.groupby(["kind"]).agg(n=("step_correct","size"),r=("step_correct","mean")))
# does product of step_correct = success?
g=so.groupby(["agent","case_id"]).agg(all_ok=("step_correct","min"),success=("success","first"),should=("should","first")).reset_index()
print(pd.crosstab([g.should,g.all_ok],g.success))
print(g.groupby("agent").agg(sr=("success","mean"),ok=("all_ok","mean")).round(2).T.to_string())
