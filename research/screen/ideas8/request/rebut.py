from screen import *
import pandas as pd
pat_reb=re.compile(r"\b[Ii]f (and only if )?(the )?(service )?(agent|assistant|representative|rep)\b[^.]{0,80}\b(says?|tells?|refus\w+|den\w+|declin\w+|cannot|can't|does not|doesn't|will not|won't|not (possible|allowed|able)|disagree\w*|is unable|unable)")
pat_any_if=re.compile(r"\b[Ii]f\b")
pat_hide=re.compile(r"(ONLY (MENTION|PROVIDE|SHARE)|only (mention|provide|share|reveal|give)\b|do not (mention|volunteer|reveal|dump)|don't (mention|volunteer|reveal|dump)|unless (the )?(agent )?ask|if (the )?(agent )?(asks|is asked)|when (the )?(agent )?asks?)")
pat_ask=re.compile(r"\b(insist|persist|do not take no|don't take no|not take no|push back|argue|again)\b",re.I)
for b in ['tau2','tauk_banking']:
    st,tk,it=load(b)
    rows=[]
    for i,t in st.items():
        rows.append(dict(i=i,dom=tk[i]['domain'],b=it.get(i,np.nan),rebut=len(pat_reb.findall(t)),any_if=len(pat_any_if.findall(t)),
           hide=len(pat_hide.findall(t)),insist=len(pat_ask.findall(t)),words=len(t.split())))
    df=pd.DataFrame(rows).dropna()
    print('==',b,len(df))
    for f in ['rebut','any_if','hide','insist','words']:
        per={d:round(spearmanr(g[f],g.b)[0],3) if g[f].std()>0 else None for d,g in df.groupby('dom')}
        print(f,per,'mean',np.nanmean([v for v in per.values() if v is not None]).round(3),'frac>0',round((df[f]>0).mean(),2))
    # partial: remove words via residual
    for f in ['rebut','hide']:
        r=[]
        for d,g in df.groupby('dom'):
            if g[f].std()==0: continue
            lw=np.log(g.words); xr=g[f]-np.polyval(np.polyfit(lw,g[f],1),lw); yr=g.b-np.polyval(np.polyfit(lw,g.b,1),lw)
            r.append(spearmanr(xr,yr)[0])
        print('length-adjusted',f,np.round(r,3),np.mean(r).round(3))
