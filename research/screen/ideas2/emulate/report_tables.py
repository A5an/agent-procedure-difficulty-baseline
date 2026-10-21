import json, collections, os, sys, random
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
real=json.load(open(HERE+"/real_stats.json"))
import gen, gt
def load(name): return json.load(open(f"{HERE}/validation_rows_{name}.json"))
def is_valid(r): return bool(r.get('loads') and r.get('tool_runs') and r.get('top_keys_ok') and r.get('uk_keys_ok') and r.get('code_label') is not None)
def pct(a,b): return f"{100*a/b:.0f}%" if b else "-"
def real_pats(dom,proc):
    cs,same=gen.modal_variant(dom,proc); ps=[]
    for c in same:
        lab,leaf=gt.leaf_results(dom,c,goal=proc); ps.append((lab,tuple(sorted(k for k,v in leaf.items() if not v))))
    return ps
def rarefied(pats, n, reps=300):
    rnd=random.Random(0); 
    if len(pats)<=n: return len(set(pats))
    return float(np.mean([len(set(rnd.sample(pats,n))) for _ in range(reps)]))
def table(rows, title, outf):
    by=collections.defaultdict(list)
    for r in rows: by[(r['dom'],r['proc'])].append(r)
    lines=[f"### {title}",
    "| procedure | N | deep schema ok (diagnostic) | loads | tool runs | valid | code perform share (real) | LLM perform share | LLM label accuracy | distinct fail patterns, synthetic at n_real draws (all N) vs real same variant | real perform share of the variant |",
    "|---|---|---|---|---|---|---|---|---|---|---|"]
    T=collections.Counter(); accs=[]; cps=[]; lps=[]
    for k,L in by.items():
        n=len(L); name=f"{k[0]}.{k[1]}"
        valid=[r for r in L if is_valid(r)]
        rp=real_pats(*k); nreal=len(rp); rshare=np.mean([a for a,_ in rp])
        sp=[tuple(r['fail_pat']) for r in valid]
        acc=np.mean([r['llm_label']==r['code_label'] for r in valid]) if valid else float('nan')
        cp=np.mean([r['code_label'] for r in valid]) if valid else float('nan'); lp=np.mean([r['llm_label'] for r in valid]) if valid else float('nan')
        lines.append(f"| {name} | {n} | {pct(sum(bool(r.get('schema_ok')) for r in L),n)} | {pct(sum(bool(r.get('loads')) for r in L),n)} | {pct(sum(bool(r.get('tool_runs')) for r in L),n)} | {pct(len(valid),n)} | {cp:.2f} ({rshare:.2f}) | {lp:.2f} | {acc:.2f} | {rarefied(sp,nreal):.1f} ({len(set(sp))}) vs {len(set(p for _,p in rp))} (n={nreal}) | {rshare:.2f} |")
        T['n']+=n; T['valid']+=len(valid); T['schema']+=sum(bool(r.get('schema_ok')) for r in L); T['loads']+=sum(bool(r.get('loads')) for r in L); T['tool']+=sum(bool(r.get('tool_runs')) for r in L)
        T['acc_n']+=len(valid); T['acc_ok']+=sum(r['llm_label']==r['code_label'] for r in valid); T['perf']+=sum(r['code_label'] for r in valid); T['llmperf']+=sum(r['llm_label'] for r in valid)
        T['lab_ok']+=sum(bool(r['code_label']) for r in valid)
        # confusion
        T['tp']+=sum(r['llm_label'] and r['code_label'] for r in valid); T['fp']+=sum(r['llm_label'] and not r['code_label'] for r in valid)
        T['fn']+=sum((not r['llm_label']) and r['code_label'] for r in valid); T['tn']+=sum((not r['llm_label']) and not r['code_label'] for r in valid)
    lines.append(f"| ALL | {T['n']} | {pct(T['schema'],T['n'])} | {pct(T['loads'],T['n'])} | {pct(T['tool'],T['n'])} | {pct(T['valid'],T['n'])} | {T['perf']/max(1,T['acc_n']):.2f} | {T['llmperf']/max(1,T['acc_n']):.2f} | {T['acc_ok']/max(1,T['acc_n']):.2f} | | |")
    lines.append(f"\nConfusion of the LLM label against the code label on valid cases: both perform {T['tp']}, LLM perform but code refuse {T['fp']}, LLM refuse but code perform {T['fn']}, both refuse {T['tn']}.\n")
    open(outf,"a").write("\n".join(lines)+"\n\n"); print("\n".join(lines))
if __name__=="__main__":
    out=HERE+"/tables.md"; open(out,"w").write("")
    for name,title in [("gen_full","v1 prompt (first schema skeleton, types only, variants A and B)"),("gen_v2","v2 prompt (all variants A, C, D)"),("gen_v3","v3 prompt (domain-wide schema, variants E and F)")]:
        if os.path.exists(f"{HERE}/validation_rows_{name}.json"):
            rows=load(name)
            table(rows,title,out)
            if name=="gen_v3":
                for v,desc in [("E","E domain-wide schema, types plus format examples"),("F","F as E plus coverage hint")]:
                    table([r for r in rows if r['variant']==v],"v3 variant "+desc,out)
            if name=="gen_v2":
                for v,desc in [("A","A types only"),("C","C types plus format examples"),("D","D types plus format examples plus coverage hint")]:
                    table([r for r in rows if r['variant']==v],"v2 variant "+desc,out)
