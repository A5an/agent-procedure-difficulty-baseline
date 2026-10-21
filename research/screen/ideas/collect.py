import json, glob, os
rows = {}
for p in sorted(glob.glob('*/out/**/eval.json', recursive=True)):
    grp = p.split('/')[0]
    pooled = json.load(open(p)).get('pooled_new_process', {})
    ex = os.path.join(os.path.dirname(p), 'extra_metrics.json')
    exs = json.load(open(ex)).get('summary', {}) if os.path.exists(ex) else {}
    for c, r in pooled.items():
        e = exs.get(c, {})
        rows[f'{grp}:{c}'] = dict(grp=grp, col=c, rho=r['rho_mean'], ci=r['rho_ci'], worst=r['rho_worst'], by=r['rho_by_scenario'],
            diff=r.get('minus_baseline_ci'), proc=e.get('rho_procedure_level'), proc_diff=e.get('rho_procedure_level_minus_baseline_ci'),
            hard=e.get('hard_recall20_mean'), lift=e.get('selective_lift50_mean'))
json.dump(rows, open('collected.json', 'w'), indent=1)
for k, r in rows.items():
    f = lambda x: 'na' if x is None else f'{x:+.3f}'
    d = r['diff'] or [None, None]; pd_ = r['proc_diff'] or [None, None]
    print(f"{k:42s} rho {f(r['rho'])} by {[round(x,3) for x in r['by']]} diff [{f(d[0])},{f(d[1])}] proc {f(r['proc'])} [{f(pd_[0])},{f(pd_[1])}] hard {f(r['hard'])} lift {f(r['lift'])}")
