"""CPU replay of the qchunk A/B: acceptance, the paired statistic, its in-session A/A floor,
cross-arm structure and the chunk census. Exits non-zero unless every criterion holds."""
from __future__ import annotations
import argparse, itertools, json, math, statistics, sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT/'perf/other512')]
from control import coverage
from fold_reduce import holder_coverage, ambient_window, coordinates, score, lines, sha, validate_sequence
from cif_rmsd import bfactor_plddt

PREDICT_ARGS={'recycling_steps':3,'sampling_steps':200,'diffusion_samples':1,'max_parallel_samples':None}

def boot_median(values,n=20000,seed=0):
    if len(values)<2:return None
    a=np.asarray(values,dtype=float);rng=np.random.default_rng(seed)
    draws=np.median(rng.choice(a,size=(n,a.size),replace=True),axis=1)
    return {'median':float(np.median(a)),'mean':float(a.mean()),'n':int(a.size),
            'ci95':[float(np.percentile(draws,2.5)),float(np.percentile(draws,97.5))],
            'excludes_zero':bool(np.percentile(draws,2.5)>0 or np.percentile(draws,97.5)<0),
            'positive':int((a>0).sum()),'negative':int((a<0).sum())}

def reduce(root):
    root=Path(root);criterion=json.loads((HERE/'criterion.json').read_text())
    result={'verdict':'GO','flag':criterion['flag'],'targets':{},
            'scope':'One flag, two sizes, one card, pinned 1350 MHz. Wall seconds and same-seed structure only; no device-cycle census and no per-op attribution.'}
    limit=criterion['quiet']['host_cpu_witness']['foreign_cpu_reject_pct']
    ambient=lines(root/'ambient.jsonl') if (root/'ambient.jsonl').exists() or (root/'ambient.jsonl.gz').exists() else None
    result['host_cpu_witness']='recorded' if ambient is not None else 'not recorded in this run'
    for size in criterion['model']['sizes']:
        d=root/str(size)
        if not (d/'result.json').exists():
            result['targets'][str(size)]={'verdict':'STOP','reason':'target not run'};result['verdict']='STOP';continue
        run=json.loads((d/'result.json').read_text());samples=lines(d/'clock.jsonl');holders=lines(d/'holders.jsonl')
        pre=[]
        if run.get('production_diff'):pre.append('production diff')
        if run.get('source_base')!=criterion['source_base']:pre.append('source base')
        if run.get('model_predict_args')!=PREDICT_ARGS:pre.append('model config')
        if run.get('release_response',[None])[0]!=0 or run.get('sampler_returncode')!=0:pre.append('release/sampler')
        if not run.get('flag',{}).get('default_on') or run.get('flag',{}).get('in_environ'):pre.append('flag default')
        for snap in [run.get('before',{}),run.get('after',{})]+[r[k] for r in run['rows'] for k in ['before','after']]:
            if snap.get('boot_id')!=run['before']['boot_id'] or snap.get('containment')!='active' or snap.get('module_srcversion')!='A10759A24565BC5BBE903C5':pre.append('boot/containment')
            if any(h['pid']!=run['pid'] for h in snap.get('holders',[])):pre.append('foreign holder snapshot')
        if run.get('after',{}).get('own_nodes'):pre.append('device remains open')
        records=[];accepted=[];paths={}
        for r in run['rows']:
            clock=coverage(samples,r);hc=holder_coverage(holders,r,run['pid']);amb=ambient_window(ambient,r,limit)
            rr=dict(r,clock=clock,holder_coverage=hc,ambient=amb)
            failures=list(pre)
            if abs(r['elapsed_s']-(r['end_monotonic_ns']-r['start_monotonic_ns'])/1e9)>1e-12:failures.append('timer mismatch')
            if r.get('above_cap_sdpa_counts')!=[0,0]:failures.append('above-cap route')
            if r.get('flag_during_fold')!=(r['arm']=='on'):failures.append('flag/arm mismatch')
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
            # the two census folds warm each arm's program config and are never timed
            rr['accepted']=not failures and not r['census'];rr['rejections']=failures
            records.append(rr)
            if rr['accepted']:accepted.append(rr)
        timed=[r for r in records if not r['census']]
        usable=all(r['accepted'] for r in timed) and len(timed)==4*criterion['reps']
        # ---- the paired statistic, from adjacent folds only
        cross=[];same=[]
        for x,y in zip(timed,timed[1:]):
            if not (x['accepted'] and y['accepted']):continue
            if x['arm']!=y['arm']:
                off,on=(y,x) if y['arm']=='off' else (x,y)
                cross.append({'labels':[x['label'],y['label']],'off_minus_on_s':off['elapsed_s']-on['elapsed_s'],'ratio_off_over_on':off['elapsed_s']/on['elapsed_s']})
            else:
                same.append({'labels':[x['label'],y['label']],'arm':x['arm'],'delta_s':y['elapsed_s']-x['elapsed_s'],'abs_delta_s':abs(y['elapsed_s']-x['elapsed_s'])})
        # ---- drift-immune ABBA rep contrast: mean(off) - mean(on) inside each `on off off on` rep
        reps=[]
        for i in range(criterion['reps']):
            block=[r for r in timed if r['label'].startswith(f'R{i}_')]
            if len(block)!=4 or not all(r['accepted'] for r in block):continue
            on=[r['elapsed_s'] for r in block if r['arm']=='on'];off=[r['elapsed_s'] for r in block if r['arm']=='off']
            reps.append({'rep':i,'on_mean_s':statistics.mean(on),'off_mean_s':statistics.mean(off),'off_minus_on_s':statistics.mean(off)-statistics.mean(on)})
        stats={'adjacent_cross_pairs':boot_median([c['off_minus_on_s'] for c in cross]),
               'adjacent_cross_ratio':boot_median([c['ratio_off_over_on']-1.0 for c in cross]),
               'abba_rep_contrast':boot_median([r['off_minus_on_s'] for r in reps]),
               'AA_same_arm_adjacent':boot_median([s['delta_s'] for s in same]),
               'AA_same_arm_adjacent_abs_median_s':statistics.median([s['abs_delta_s'] for s in same]) if same else None,
               'baseline_single_pair_floor_s':criterion['AA_floor_s'][str(size)]}
        for key in ['adjacent_cross_pairs','abba_rep_contrast','AA_same_arm_adjacent']:
            if stats[key]:stats[key]['Mcycles_at_1350']= {k:(v*1350 if isinstance(v,(int,float)) else [x*1350 for x in v]) for k,v in stats[key].items() if k in ('median','mean','ci95')}
        arms={a:[r['elapsed_s'] for r in accepted if r['arm']==a] for a in ('on','off')}
        stats['arm_medians_s']={a:statistics.median(v) for a,v in arms.items() if v}
        stats['arm_n']={a:len(v) for a,v in arms.items()}
        # ---- parity
        by_arm={a:sorted({r['cif_sha256'] for r in accepted if r['arm']==a}) for a in ('on','off')}
        crosspairs=[dict(labels=[x['label'],y['label']],arms=[x['arm'],y['arm']],**score(d/x['cif'],d/y['cif'],size))
                    for x,y in itertools.combinations(accepted,2) if x['arm']!=y['arm']]
        bar=criterion['accuracy']['bars_A'][str(size)]
        parity={'cif_sha256_by_arm':by_arm,'bit_exact_within_on':len(by_arm['on'])==1,'bit_exact_within_off':len(by_arm['off'])==1,
                'bit_exact_across_arms':len(set(by_arm['on'])|set(by_arm['off']))==1,
                'cross_arm_pairs':len(crosspairs),
                'max_cross_arm_domain_A':max((p['max_domain_all_atom_A'] for p in crosspairs),default=None),
                'max_cross_arm_whole_chain_A':max((p['whole_chain_all_atom_A'] for p in crosspairs),default=None),
                'bar_A':bar,'seed_floor_context_A':criterion['accuracy'],
                'plddt_by_arm':{a:sorted({r['plddt'] for r in accepted if r['arm']==a}) for a in ('on','off')}}
        parity['passes_bar']=parity['max_cross_arm_domain_A'] is not None and parity['max_cross_arm_domain_A']<=bar
        census=run.get('census_summary',{})
        census_ok=(bool(census) and not census.get('off',{}).get('sites_moved') and bool(census.get('on',{}).get('sites_moved'))
                   and census.get('on',{}).get('calls')==census.get('off',{}).get('calls'))
        ok=(run.get('completed',False) and usable and len(cross)>=criterion['minimum_cross_pairs']
            and len(same)>=criterion['minimum_AA_pairs'] and parity['passes_bar'] and census_ok)
        out={'verdict':'GO' if ok else 'STOP','accepted_folds':len(accepted),'timed_folds':len(timed),
             'clock':{'during_min_MHz':min((r['clock']['min_MHz'] for r in accepted),default=None),
                      'during_max_MHz':max((r['clock']['max_MHz'] for r in accepted),default=None),
                      'total_during_samples':sum(r['clock']['samples'] for r in accepted)},
             'statistics':stats,'parity':parity,'census':run.get('census',{}),'census_summary':census,'census_ok':census_ok,
             'adjacent_cross_pairs':cross,'same_arm_adjacent_pairs':same,'abba_reps':reps,
             'above_cap_route_counters':sorted({tuple(r['above_cap_sdpa_counts']) for r in records}),
             'ambient':{'max_foreign_cpu_pct':max((r['ambient']['max_foreign_cpu_pct'] for r in accepted if r['ambient']['recorded']),default=None),
                        'max_loadavg':max((r['ambient']['max_loadavg'] for r in accepted if r['ambient']['recorded']),default=None)},
             'rows':records,'run_errors':run.get('errors',[])}
        out['above_cap_route_counters']=[list(t) for t in out['above_cap_route_counters']]
        result['targets'][str(size)]=out
        if not ok:result['verdict']='STOP'
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    r=reduce(a.run);a.out.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:{x:y for x,y in v.items() if x in ['verdict','accepted_folds','statistics','parity','census_summary','clock','ambient','run_errors']} for k,v in r['targets'].items()},indent=2,default=str))
    sys.exit(0 if r['verdict']=='GO' else 2)
