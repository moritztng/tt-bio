import csv,sys,collections,statistics as st
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
top={};allm=collections.defaultdict(list)
for r in rows:
    k=(r['mode'],r['pdb'],r['seed']); allm[k].append(float(r['dockq']))
    if k not in top or float(r['ranking_score'])>float(top[k]['ranking_score']): top[k]=r
modes=sorted({k[0] for k in top})
for pdb in ('9DBP','9TH6'):
    seeds=sorted({k[2] for k in top if k[1]==pdb})
    for m in modes:
        v=[float(top[(m,pdb,s)]['dockq']) for s in seeds if (m,pdb,s) in top]
        a=[st.mean(allm[(m,pdb,s)]) for s in seeds if (m,pdb,s) in top]
        print(pdb,m,'top DockQ per seed',[round(x,3) for x in v],'mean',round(st.mean(v),4),'all-sample mean',round(st.mean(a),4),'acceptable(>=0.23)',sum(x>=0.23 for x in v),'/',len(v))
    if len(modes)>1:
        for m in modes[1:]:
            d=[float(top[(m,pdb,s)]['dockq'])-float(top[(modes[0],pdb,s)]['dockq']) for s in seeds if (m,pdb,s) in top and (modes[0],pdb,s) in top]
            print('  ',pdb,m,'-',modes[0],'same-seed dDockQ',[round(x,4) for x in d])
