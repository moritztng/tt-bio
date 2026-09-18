"""CPU replay of bare-fold timing, clock/holder coverage and same-seed coordinates."""
from __future__ import annotations
import argparse, gzip, hashlib, itertools, json, math, statistics, sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT/'perf/other512')]
from control import coverage
from audit_accuracy import validate_atoms
from cif_rmsd import read_atoms, kabsch_rmsd, bfactor_plddt

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lines(p):
    p=Path(p)
    if not p.exists(): p=Path(str(p)+'.gz')
    opener=gzip.open if p.suffix=='.gz' else open
    with opener(p,'rt') as f: return [json.loads(x) for x in f if x.strip()]

def holder_coverage(observations,interval,pid):
    start,end=interval['start_monotonic_ns'],interval['end_monotonic_ns']
    rows=[r for r in observations if start<=r['monotonic_ns']<=end]
    points=[start]+[r['monotonic_ns'] for r in rows]+[end]
    bad=[r for r in rows if r.get('error') or r.get('owner_nodes')!=['/dev/tenstorrent/0'] or any(h['pid']!=pid for h in r.get('holders',[]))]
    gap=max(b-a for a,b in zip(points,points[1:]))
    return dict(samples=len(rows),max_gap_ns=gap,bad=bad,passed=bool(rows) and not bad and gap<=1_000_000_000)

def ambient_window(rows,interval,limit):
    if rows is None:return {'recorded':False,'note':'no host CPU witness in this run'}
    start,end=interval['start_monotonic_ns'],interval['end_monotonic_ns']
    sel=[r for r in rows if start<=r['monotonic_ns']<=end]
    foreign=sorted(({**b,'at_ns':r['monotonic_ns']} for r in sel for b in r['busy'] if not b['own']),key=lambda b:-b['cpu_pct'])
    worst=max((b['cpu_pct'] for b in foreign),default=0.0)
    return {'recorded':True,'samples':len(sel),'max_loadavg':max((r['loadavg'] for r in sel),default=None),
            'max_foreign_cpu_pct':worst,'foreign_over_limit':[b for b in foreign if b['cpu_pct']>=limit],
            'foreign_top':foreign[:5],'passed':bool(sel) and worst<limit}

def coordinates(path,size):
    keys,xyz=read_atoms(path);validate_atoms(keys,xyz,size)
    order=sorted(range(len(keys)),key=lambda i:keys[i])
    return [keys[i] for i in order],xyz[order]

def validate_sequence(keys,size):
    import yaml
    from Bio.SeqUtils import seq1
    fixture=yaml.safe_load((ROOT/f'perf/size512/fixtures/cdk2x2_{size}.yaml').read_text())
    expected=fixture['sequences'][0]['protein']['sequence']
    ca=sorted((int(k[1]),k[3]) for k in keys if k[2]=='CA')
    actual=''.join(seq1(name) for _,name in ca)
    if actual!=expected:raise ValueError('CIF residue sequence differs from fixture')

def score(a,b,size):
    ka,xa=coordinates(a,size);kb,xb=coordinates(b,size)
    if ka!=kb:raise ValueError('atom identities differ')
    seq=np.array([int(k[1]) for k in ka]);masks=[seq<=298]
    if size==512:masks.append(seq>298)
    dom=[kabsch_rmsd(xa[m],xb[m]) for m in masks]
    return dict(atoms=len(ka),whole_chain_all_atom_A=kabsch_rmsd(xa,xb),domain_all_atom_A=dom,max_domain_all_atom_A=max(dom),cif_byte_exact=sha(a)==sha(b))

def reduce(root):
    root=Path(root);criterion=json.loads((HERE/'criterion.json').read_text());result={'verdict':'GO','targets':{},'scope':'Current-source bare wall and same-seed A/A only; no upstream accuracy or device-work cycles.'}
    limit=criterion['quiet']['host_cpu_witness']['foreign_cpu_reject_pct']
    ambient=lines(root/'ambient.jsonl') if (root/'ambient.jsonl').exists() or (root/'ambient.jsonl.gz').exists() else None
    result['host_cpu_witness']='recorded' if ambient is not None else 'not recorded in this run'
    for size in criterion['model']['sizes']:
        d=root/str(size)
        if not (d/'result.json').exists():
            result['targets'][str(size)]={'verdict':'STOP','reason':'target not run'};result['verdict']='STOP';continue
        run=json.loads((d/'result.json').read_text());samples=lines(d/'clock.jsonl');holders=lines(d/'holders.jsonl')
        prerequisite_errors=[]
        if run.get('production_diff'):prerequisite_errors.append('production diff')
        if run.get('source_base')!=criterion['source_base']:prerequisite_errors.append('source base')
        if run.get('model_predict_args')!={'recycling_steps':3,'sampling_steps':200,'diffusion_samples':1,'max_parallel_samples':None}:prerequisite_errors.append('model config')
        if run.get('release_response',[None])[0]!=0 or run.get('sampler_returncode')!=0:prerequisite_errors.append('release/sampler')
        for snap in [run.get('before',{}),run.get('after',{})]+[r[k] for r in run['rows'] for k in ['before','after']]:
            if snap.get('boot_id')!=run['before']['boot_id'] or snap.get('containment')!='active' or snap.get('module_srcversion')!='A10759A24565BC5BBE903C5':prerequisite_errors.append('boot/containment')
            if any(h['pid']!=run['pid'] for h in snap.get('holders',[])):prerequisite_errors.append('foreign holder snapshot')
        if run.get('after',{}).get('own_nodes'):prerequisite_errors.append('device remains open')
        records=[];accepted=[];paths={}
        for r in run['rows']:
            clock=coverage(samples,r);hc=holder_coverage(holders,r,run['pid']);amb=ambient_window(ambient,r,limit);rr=dict(r,clock=clock,holder_coverage=hc,ambient=amb)
            failures=list(prerequisite_errors)
            if abs(r['elapsed_s']-(r['end_monotonic_ns']-r['start_monotonic_ns'])/1e9)>1e-12:failures.append('timer mismatch')
            if r.get('above_cap_sdpa_counts')!=[0,0]:failures.append('above-cap route')
            if not math.isfinite(float(r.get('plddt',float('nan')))):failures.append('nonfinite confidence')
            for name,val in [('clock',clock['pass']),('holders',hc['passed']),('run_valid',r.get('valid',False)),
                             ('foreign host CPU',amb['passed'] if amb['recorded'] else True)]:
                if not val:failures.append(name)
            p=d/r['cif'];paths[r['label']]=p
            try:
                keys,xyz=coordinates(p,size);validate_sequence(keys,size)
                if sha(p)!=r['cif_sha256']:raise ValueError('CIF hash mismatch')
                bf=bfactor_plddt(p)
                if not bf or not math.isfinite(bf['mean_ca']):raise ValueError('invalid pLDDT column')
                rr.update(atoms=len(keys),ca_count=size,cif_plddt=bf)
            except (ValueError,OSError) as e:failures.append(str(e))
            rr['accepted']=not failures and r['label']!='cold';rr['rejections']=failures
            records.append(rr)
            if rr['accepted']:accepted.append(rr)
        pairs=[]
        for i in range(criterion['pairs']):
            a,b=f'A{i}',f'A_repeat{i}'
            if a not in paths or b not in paths:continue
            pair=score(paths[a],paths[b],size);pair['labels']=[a,b]
            selected=[r for r in records if r['label'] in [a,b]]
            pair['timing_accepted']=len(selected)==2 and all(r['accepted'] for r in selected)
            pair['abs_delta_s']=abs(selected[1]['elapsed_s']-selected[0]['elapsed_s'])
            pair['signed_repeat_minus_A_s']=selected[1]['elapsed_s']-selected[0]['elapsed_s']
            pair['passes_structure_bar']=pair['max_domain_all_atom_A']<=criterion['accuracy']['bars_A'][str(size)]
            pairs.append(pair)
        allpairs=[dict(labels=[a['label'],b['label']],**score(d/a['cif'],d/b['cif'],size)) for a,b in itertools.combinations(accepted,2)]
        times=[r['elapsed_s'] for r in accepted]
        ok=(run.get('completed',False) and len([p for p in pairs if p['timing_accepted'] and p['passes_structure_bar']])>=criterion['minimum_pairs'] and all(p['max_domain_all_atom_A']<=criterion['accuracy']['bars_A'][str(size)] for p in allpairs))
        out={'verdict':'GO' if ok else 'STOP','accepted_folds':len(times),'rows':records,'adjacent_pairs':pairs,'all_warm_structure_pairs':allpairs,'run_errors':run.get('errors',[]),'accuracy_context':criterion['accuracy']}
        if times:
            out['summary']={'median_s':statistics.median(times),'min_s':min(times),'max_s':max(times),'spread_s':max(times)-min(times),'sample_stdev_s':statistics.stdev(times) if len(times)>1 else None,'during_min_MHz':min(r['clock']['min_MHz'] for r in accepted),'during_max_MHz':max(r['clock']['max_MHz'] for r in accepted),'median_elapsed_device_clock_equivalent_Mcycles':statistics.median(times)*1350,'cycle_units_warning':'Elapsed device-clock-equivalent only, includes host gaps; not measured device work.','max_all_warm_pair_domain_A':max((p['max_domain_all_atom_A'] for p in allpairs),default=None),'adjacent_abs_delta_median_s':statistics.median([p['abs_delta_s'] for p in pairs if p['timing_accepted']]) if any(p['timing_accepted'] for p in pairs) else None,'all_warm_cif_byte_exact':len({r['cif_sha256'] for r in accepted})==1,'max_foreign_cpu_pct':max((r['ambient']['max_foreign_cpu_pct'] for r in accepted if r['ambient']['recorded']),default=None),'max_loadavg':max((r['ambient']['max_loadavg'] for r in accepted if r['ambient']['recorded']),default=None)}
        result['targets'][str(size)]=out
        if not ok:result['verdict']='STOP'
    if (root/'excluded.json').exists():
        result['exclusion']=json.loads((root/'excluded.json').read_text());result['verdict']='STOP'
        for v in result['targets'].values():
            v['verdict']='STOP';v['accepted_folds']=0;v.pop('summary',None)
            for r in v.get('rows',[]):r['accepted']=False;r['rejections'].append('whole capture excluded')
            for p in v.get('adjacent_pairs',[]):p['timing_accepted']=False
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    result=reduce(a.run);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{x:y for x,y in v.items() if x in ['verdict','accepted_folds','summary','run_errors']} for k,v in result['targets'].items()},indent=2))
    sys.exit(0 if result['verdict']=='GO' else 2)
