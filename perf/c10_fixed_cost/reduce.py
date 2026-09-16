"""CPU replay of the interleaved clock arms: per-cell timing, the inverse-clock fit, and what 10.0 s demands.

Row validation is c10-bare-baseline's, unchanged, except that each fold's clock coverage is scored
against its OWN requested clock instead of a hardcoded 1350 MHz.
"""
from __future__ import annotations
import argparse, gzip, hashlib, itertools, json, math, statistics, sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT/'perf/other512')]
from clockarm import coverage, fit_inverse_clock, propagate_two_clock
from audit_accuracy import validate_atoms
from cif_rmsd import read_atoms, kabsch_rmsd, bfactor_plddt

TARGET_S=10.0
CUT_F_S=1.0

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

def demand(F,C,clock):
    """What TARGET_S at `clock` costs, with F untouched and with F cut to CUT_F_S."""
    out={'target_s':TARGET_S,'at_MHz':clock,'F_s':F,'C_Mcycles':C,
         'seconds_now':F+C/clock,
         'F_alone_at_current_cycles_s':CUT_F_S+C/clock,
         'reachable_with_F_untouched':F<TARGET_S}
    if F<TARGET_S:
        need=clock*(TARGET_S-F)
        out['F_untouched']={'cycles_allowed_Mcycles':need,'cycle_cut_Mcycles':C-need,
                            'cycle_cut_pct':100*(C-need)/C,
                            'device_work_seconds_allowed_at_MHz':need/clock}
    else:
        out['F_untouched']={'impossible':f'the measured fixed term alone is {F:.3f} s, above {TARGET_S} s: '
                                        'no cycle cut whatsoever reaches the target'}
    need2=clock*(TARGET_S-CUT_F_S)
    out['F_cut_to_1s']={'assumed_F_s':CUT_F_S,'cycles_allowed_Mcycles':need2,'cycle_cut_Mcycles':C-need2,
                        'cycle_cut_pct':100*(C-need2)/C,'already_met':C<=need2}
    return out

def reduce(root):
    root=Path(root);criterion=json.loads((HERE/'criterion.json').read_text())
    result={'verdict':'GO','targets':{},'scope':'Clock-immune fixed term of the current-tree bare fold, from interleaved pinned-clock arms. No device-cycle census, no per-op budget, no upstream accuracy.'}
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
        records=[];accepted=[]
        for r in run['rows']:
            clk=r['clock_MHz']
            clock=coverage(samples,r,clk);hc=holder_coverage(holders,r,run['pid']);amb=ambient_window(ambient,r,limit)
            rr=dict(r,clock=clock,holder_coverage=hc,ambient=amb)
            failures=list(prerequisite_errors)
            if clk not in criterion['clock']['arms_MHz']:failures.append('clock not an authorised arm')
            if not r.get('settle',{}).get('settled'):failures.append('clock did not settle before the fold')
            if r.get('force_response',[None])[0]!=0:failures.append('FORCE_AICLK not accepted')
            if abs(r['elapsed_s']-(r['end_monotonic_ns']-r['start_monotonic_ns'])/1e9)>1e-12:failures.append('timer mismatch')
            if r.get('above_cap_sdpa_counts')!=[0,0]:failures.append('above-cap route')
            if not math.isfinite(float(r.get('plddt',float('nan')))):failures.append('nonfinite confidence')
            for name,val in [('clock',clock['pass']),('holders',hc['passed']),('run_valid',r.get('valid',False)),
                             ('foreign host CPU',amb['passed'] if amb['recorded'] else True)]:
                if not val:failures.append(name)
            p=d/r['cif']
            try:
                keys,xyz=coordinates(p,size);validate_sequence(keys,size)
                if sha(p)!=r['cif_sha256']:raise ValueError('CIF hash mismatch')
                bf=bfactor_plddt(p)
                if not bf or not math.isfinite(bf['mean_ca']):raise ValueError('invalid pLDDT column')
                rr.update(atoms=len(keys),ca_count=size,cif_plddt=bf)
            except (ValueError,OSError) as e:failures.append(str(e))
            rr['accepted']=not failures and r['label']!='cold';rr['rejections']=failures
            rr['cif_path']=str(p)
            records.append(rr)
            if rr['accepted']:accepted.append(rr)
        cells={}
        for clk in criterion['clock']['arms_MHz']:
            sel=[r for r in accepted if r['clock_MHz']==clk]
            t=[r['elapsed_s'] for r in sel]
            if not t:
                cells[str(clk)]={'accepted_folds':0,'enough':False};continue
            sd=statistics.stdev(t) if len(t)>1 else None
            cells[str(clk)]={'accepted_folds':len(t),'enough':len(t)>=criterion['minimum_folds_per_cell'],
                'median_s':statistics.median(t),'mean_s':statistics.fmean(t),'min_s':min(t),'max_s':max(t),
                'spread_s':max(t)-min(t),'sample_stdev_s':sd,'se_median_s':(sd/math.sqrt(len(t))) if sd else None,
                'during_min_MHz':min(r['clock']['min_MHz'] for r in sel),'during_max_MHz':max(r['clock']['max_MHz'] for r in sel),
                'mean_W':statistics.fmean([r['clock']['mean_W'] for r in sel if r['clock']['mean_W'] is not None]) if any(r['clock']['mean_W'] is not None for r in sel) else None,
                'max_C':max((r['clock']['max_C'] for r in sel if r['clock']['max_C'] is not None),default=None),
                'host_cpu_s_median':statistics.median([r['host_cpu_s'] for r in sel]),
                'host_cpu_s_min':min(r['host_cpu_s'] for r in sel),'host_cpu_s_max':max(r['host_cpu_s'] for r in sel),
                'adjacent_abs_delta_s':[abs(b-a) for a,b in zip(t,t[1:])],
                'adjacent_abs_delta_median_s':statistics.median([abs(b-a) for a,b in zip(t,t[1:])]) if len(t)>1 else None,
                'elapsed_device_clock_equivalent_Mcycles':statistics.median(t)*clk}
        out={'accepted_folds':len(accepted),'rows':records,'cells':cells}
        # structure: identical bytes are exactly zero by construction, so only distinct CIFs are scored
        by_hash={}
        for r in accepted:by_hash.setdefault(r['cif_sha256'],[]).append(r)
        reps=[v[0] for v in by_hash.values()]
        pairs=[dict(labels=[a['label'],b['label']],clocks=[a['clock_MHz'],b['clock_MHz']],**score(a['cif_path'],b['cif_path'],size))
               for a,b in itertools.combinations(reps,2)]
        out['structure']={'distinct_cif_sha256':len(by_hash),'all_accepted_byte_exact':len(by_hash)==1,
            'folds_per_hash':{h[:16]:[r['label'] for r in v] for h,v in by_hash.items()},
            'distinct_pairs_scored':pairs,
            'max_pair_domain_A':max((p['max_domain_all_atom_A'] for p in pairs),default=0.0),
            'bar_A':criterion['accuracy']['bars_A'][str(size)],
            'context':criterion['accuracy']}
        out['structure']['passes_bar']=out['structure']['max_pair_domain_A']<=out['structure']['bar_A']
        ok=(run.get('completed',False)
            and all(cells[str(c)]['enough'] for c in criterion['clock']['arms_MHz'])
            and out['structure']['passes_bar'])
        if len(accepted)>=2*criterion['minimum_folds_per_cell'] and len({r['clock_MHz'] for r in accepted})>=2:
            pts=[(r['clock_MHz'],r['elapsed_s']) for r in accepted]
            out['fit_all_folds']=fit_inverse_clock(pts)
            meds=[(c,cells[str(c)]['median_s']) for c in criterion['clock']['arms_MHz'] if cells[str(c)].get('median_s') is not None]
            out['fit_cell_medians']=fit_inverse_clock(meds)
            hi,lo=max(c for c,_ in meds),min(c for c,_ in meds)
            out['two_clock_endpoints']=propagate_two_clock(
                hi,cells[str(hi)]['median_s'],cells[str(hi)]['se_median_s'] or 0.0,
                lo,cells[str(lo)]['median_s'],cells[str(lo)]['se_median_s'] or 0.0)
            f=out['fit_all_folds']
            floor=max((cells[str(c)].get('adjacent_abs_delta_median_s') or 0.0) for c in criterion['clock']['arms_MHz'])
            fastest=min(cells[str(c)]['median_s'] for c in criterion['clock']['arms_MHz'])
            # Three independent estimators of the same F. Their spread IS the model-misfit term, so
            # F is bounded by it rather than quoted to more digits than the model supports.
            est=[('all_folds',f['F_s'],f.get('se_F_s') or 0.0),
                 ('cell_medians',out['fit_cell_medians']['F_s'],out['fit_cell_medians'].get('se_F_s') or 0.0),
                 ('two_clock_endpoints',out['two_clock_endpoints']['F_s'],out['two_clock_endpoints']['se_F_s'])]
            lo_b=min(v-e for _,v,e in est);hi_b=max(v+e for _,v,e in est)
            exact=out['fit_cell_medians']['max_abs_residual_s']<=max(floor,0.05)
            out['model_check']={'aa_timing_floor_s':floor,'max_abs_residual_s':f['max_abs_residual_s'],
                'rms_residual_s':f['rms_residual_s'],
                'cell_median_max_abs_residual_s':out['fit_cell_medians']['max_abs_residual_s'],
                'cell_median_residual_s':dict(zip([str(c) for c in out['fit_cell_medians']['clocks_MHz']],out['fit_cell_medians']['residual_s'])),
                'exact_inverse_clock_form_holds':exact,
                'estimators_F_s':{k:{'F_s':v,'se_s':e} for k,v,e in est},
                'F_bound_s':[lo_b,hi_b],'F_bound_halfwidth_s':(hi_b-lo_b)/2,
                'fixed_term_identification':'point-identified' if exact else 'bounded by model misfit: the exact two-parameter T = F + C/f form is refuted, F is an interval',
                'F_is_meaningful':0.0<f['F_s']<fastest,
                'refutation_criteria':criterion['refutation'],
                'note':'refuting the exact 1/f form is a FINDING about the fold, not a failed capture: the arms held, the host was quiet and the structure never moved. Only an F at or below 0 s, or at or above the fastest fold, would make F meaningless.'}
            out['demand']=demand(f['F_s'],f['C_Mcycles'],criterion['clock']['warm_MHz'])
            out['demand_at_F_bounds']={'F_low':demand(lo_b,f['C_Mcycles'],criterion['clock']['warm_MHz'])['F_untouched'],
                                       'F_high':demand(hi_b,f['C_Mcycles'],criterion['clock']['warm_MHz'])['F_untouched']}
            if not out['model_check']['F_is_meaningful']:ok=False
        out['verdict']='GO' if ok else 'STOP'
        out['run_errors']=run.get('errors',[])
        result['targets'][str(size)]=out
        if not ok:result['verdict']='STOP'
    sizes=[s for s in result['targets'] if 'fit_all_folds' in result['targets'][s]]
    if len(sizes)==2:
        a,b=sorted(sizes,key=lambda s:-int(s))
        fa,fb=result['targets'][a]['fit_all_folds'],result['targets'][b]['fit_all_folds']
        ea,eb=fa.get('se_F_s') or 0.0,fb.get('se_F_s') or 0.0
        result['cross_size']={'sizes':[a,b],'F_s':{a:fa['F_s'],b:fb['F_s']},'C_Mcycles':{a:fa['C_Mcycles'],b:fb['C_Mcycles']},
            'F_difference_s':fa['F_s']-fb['F_s'],'F_difference_se_s':math.sqrt(ea**2+eb**2),
            'F_size_independent_within_2se':abs(fa['F_s']-fb['F_s'])<=2*math.sqrt(ea**2+eb**2),
            'work_cycle_ratio':fa['C_Mcycles']/fb['C_Mcycles'],
            'note':'featurization and CIF writing scale with the target, so a size-dependent F is expected rather than a defect'}
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    result=reduce(a.run);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{x:y for x,y in v.items() if x in ['verdict','accepted_folds','cells','fit_all_folds','fit_cell_medians','two_clock_endpoints','model_check','demand','run_errors']} for k,v in result['targets'].items()}|{'cross_size':result.get('cross_size')},indent=2))
    sys.exit(0 if result['verdict']=='GO' else 2)
