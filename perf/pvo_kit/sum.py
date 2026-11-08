import json,glob,statistics as st,csv,datetime as dt,sys
root=sys.argv[1]
smi=[]
for r in csv.reader(open(root+'/smi.csv')):
    try: smi.append((dt.datetime.strptime(r[0].strip(),'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=dt.timezone.utc),int(r[1].split()[0]),float(r[3].split()[0]),r[8].strip()))
    except Exception: pass
for f in sorted(glob.glob(root+'/*/*/runs.jsonl')):
    rows=[json.loads(l) for l in open(f)]
    for inp in ('9DBP','9TH6'):
        rr=[r for r in rows if r.get('input')==inp]
        if not rr: continue
        w=[r['wall_s'] for r in rr if r['kind']=='warm']
        clk=[];th=set()
        for r in rr:
            if r['kind']!='warm':continue
            s=dt.datetime.fromisoformat(r['start']); e=s+dt.timedelta(seconds=r['wall_s'])
            clk+= [c for t,c,p,x in smi if s<=t<=e]; th|={x for t,c,p,x in smi if s<=t<=e}
        print(f.split('/')[-3]+'/'+f.split('/')[-2],inp,'cold',[r['wall_s'] for r in rr if r['kind']!='warm'],'warm',w,'med',st.median(w) if w else None,'SM',st.median(clk) if clk else None,'thr',th,'peakres',max(r.get('peak_reserved_gib',0) for r in rr))
