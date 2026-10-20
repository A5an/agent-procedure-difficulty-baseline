import pandas as pd,itertools
d=pd.read_csv('../pertype/per_agent_subcheck.csv').set_index('agent')
names={'gemini-2.5-flash':'gemini-2.5-flash-mode_fc','gemini-2.5-pro':'gemini-2.5-pro-mode_fc','gpt-5':'gpt-5-mode_fc','gpt-4.1':'gpt-4.1-mode_fc','gpt-5-mini':'gpt-5-mini-mode_fc','claude-3.7':'claude-3-7-sonnet-20250219-mode_fc'}
wc=288/830; wr=542/830
rows=[]
for k,v in names.items():
    r=d.loc[v]; rows.append((k+' alone','-','%.3f'%(wc*r.carry_success+wr*r.refuse_success), '%.3f'%r.carry_success,'%.3f'%r.refuse_success))
for a,b in itertools.permutations(names,2):
    A=d.loc[names[a]];B=d.loc[names[b]]
    carry=A.carry_checks*B.carry_action
    lo=min(A.refuse_success,B.refuse_success); hi=max(A.refuse_success,B.refuse_success)
    ref=A.refuse_success  # A keeps refusals it makes itself
    rows.append((f'A={a} B={b}','mixed','%.3f'%(wc*carry+wr*ref),'%.3f'%carry,'%.3f (A) [%.3f-%.3f]'%(ref,lo,hi)))
for r in rows: print(' | '.join(r))
