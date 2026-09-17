"""CPU replay of the ladder: per-cell timing, the two-clock work separation, and the per-LEG exponent.

Row validation is c10-bare-baseline's by way of c10-fixed-cost, unchanged: each fold's clock
coverage is scored against its OWN requested clock, and a fold is accepted only if the source,
config, inputs, device holders, host CPU, structure and pLDDT all hold.

What is new is the ladder arithmetic, and it is deliberately NOT a global fit. A single fit over
384/512/640/768 would average a rising exponent into a flat one, and whether the exponent rises IS
the question: a rising one is the grid-under-fill artifact's signature, a flat sublinear one is the
size-independent term's. So each leg is computed on its own two neighbouring sizes and reported
with its own standard error.
"""
from __future__ import annotations
import argparse, gzip, hashlib, itertools, json, math, statistics, sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT/'perf/other512')]
from clockarm import coverage, fit_inverse_clock, propagate_two_clock
from sizefit import work_two_clock, leg_exponent, size_independent_floor
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
    # The fixtures are one chain of the CDK2 sequence repeated and truncated, so the 512 aa
    # convention (split after residue 298) generalises to whole 298-residue blocks plus the tail.
    seq=np.array([int(k[1]) for k in ka])
    masks=[(seq>lo)&(seq<=min(lo+298,size)) for lo in range(0,size,298)]
    masks=[m for m in masks if m.any()]
    dom=[kabsch_rmsd(xa[m],xb[m]) for m in masks]
    return dict(atoms=len(ka),whole_chain_all_atom_A=kabsch_rmsd(xa,xb),domain_all_atom_A=dom,max_domain_all_atom_A=max(dom),cif_byte_exact=sha(a)==sha(b))

def criterion_anchor(criterion):
    return json.loads((HERE/'prediction.json').read_text())['anchor']


def ladder(result,criterion):
    """Per-LEG exponents over the measured ladder. Never a global fit -- see the module docstring."""
    rungs=sorted(int(s) for s,v in result['targets'].items() if 'work' in v and v.get('verdict')=='GO')
    out={'sizes_aa':rungs,'W_Mcycles':{str(n):result['targets'][str(n)]['work']['W_Mcycles'] for n in rungs},
         'F_s':{str(n):result['targets'][str(n)]['work']['F_s'] for n in rungs},
         'se_W_Mcycles':{str(n):result['targets'][str(n)]['work']['se_W_Mcycles'] for n in rungs},
         'se_p_quotability_bar':0.15,'legs':[]}
    missing=[n for n in criterion['model']['ladder_aa'] if n not in rungs]
    out['unmeasured_sizes_aa']=missing
    for n1,n2 in zip(rungs,rungs[1:]):
        t1,t2=result['targets'][str(n1)],result['targets'][str(n2)]
        leg=leg_exponent(n1,t1['work']['W_Mcycles'],t1['work']['se_W_Mcycles'],
                         n2,t2['work']['W_Mcycles'],t2['work']['se_W_Mcycles'])
        floor_u=math.hypot(t1['aa_floor']['work_uncertainty_from_floor_Mcycles'],
                           t2['aa_floor']['work_uncertainty_from_floor_Mcycles'])
        signal=abs(t2['work']['W_Mcycles']-t1['work']['W_Mcycles'])
        leg['aa_floor_check']={'work_signal_Mcycles':signal,'floor_implied_uncertainty_Mcycles':floor_u,
            'signal_exceeds_floor':signal>floor_u,
            'note':'the leg is only quotable if the work it moved is larger than what repeating an identical arm at either end already costs'}
        leg['quotable']=leg['se_p_work']<=out['se_p_quotability_bar'] and leg['aa_floor_check']['signal_exceeds_floor']
        leg.update(size_independent_floor(n1,t1['work']['W_Mcycles'],n2,t2['work']['W_Mcycles']))
        leg['reads']=('ARTIFACT: the work term grows at or faster than 1.7 here' if leg['p_work']>=1.7
                      else 'sublinear' if leg['p_work']<1.0 else 'between linear and 1.7')
        out['legs'].append(leg)
    if len(rungs)>=2 and str(512) in out['W_Mcycles'] and str(768) in out['W_Mcycles']:
        out['leg_512_to_768_direct']=leg_exponent(512,out['W_Mcycles']['512'],out['se_W_Mcycles']['512'],
                                                  768,out['W_Mcycles']['768'],out['se_W_Mcycles']['768'])
    out['quotable']=bool(out['legs']) and all(l['quotable'] for l in out['legs'])
    ps=[l['p_work'] for l in out['legs']]
    out['exponent_rises']=len(ps)>=2 and all(b>a for a,b in zip(ps,ps[1:]))
    out['exponent_shape']=[{'leg':f"{l['sizes_aa'][0]}->{l['sizes_aa'][1]}",'p_work':l['p_work'],
                            'se_p_work':l['se_p_work']} for l in out['legs']]
    top=out.get('leg_512_to_768_direct') or (out['legs'][-1] if out['legs'] else None)
    if top:
        out['decisive_leg']={'sizes_aa':top['sizes_aa'],'p_work':top['p_work'],'se_p_work':top['se_p_work'],
            'artifact_threshold':1.7,
            'sigma_below_artifact_threshold':(1.7-top['p_work'])/top['se_p_work'] if top['se_p_work'] else None,
            'verdict':'ARTIFACT' if top['p_work']>=1.7 else 'FINDING SURVIVES'}
    return out


def reduce(root):
    root=Path(root);criterion=json.loads((HERE/'criterion.json').read_text())
    result={'verdict':'GO','targets':{},'scope':'How the clock-scaled work term of the current-tree bare fold grows with the target, over a 384/512/640/768 aa ladder of one fixture family, from interleaved pinned-clock arms at 1350 and 800 MHz. No device-cycle census, no per-op budget, no upstream accuracy, no optimisation.'}
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
                'spread_s':max(t)-min(t),'sample_stdev_s':sd,
                'se_mean_s':(sd/math.sqrt(len(t))) if sd else None,
                'se_median_s':(1.2533*sd/math.sqrt(len(t))) if sd else None,
                'se_median_note':'1.2533*sd/sqrt(n), the asymptotic standard error OF A MEDIAN. The mean\'s sd/sqrt(n) understates it by 25 %, and the medians are what the two-clock separation consumes.',
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
        arms=criterion['clock']['arms_MHz'];hi,lo=max(arms),min(arms)
        if all(cells[str(c)].get('median_s') is not None for c in arms) and len(set(arms))>=2:
            pts=[(r['clock_MHz'],r['elapsed_s']) for r in accepted]
            out['fit_all_folds']=fit_inverse_clock(pts)
            out['work']=work_two_clock(hi,cells[str(hi)]['median_s'],cells[str(hi)]['se_median_s'] or 0.0,
                                       lo,cells[str(lo)]['median_s'],cells[str(lo)]['se_median_s'] or 0.0)
            # A/A timing floor: consecutive same-clock folds inside a cell are identical arms.
            floor=max((cells[str(c)].get('adjacent_abs_delta_median_s') or 0.0) for c in arms)
            out['aa_floor']={'timing_floor_s':floor,
                'per_cell_adjacent_abs_delta_median_s':{str(c):cells[str(c)].get('adjacent_abs_delta_median_s') for c in arms},
                'structural_floor_A':out['structure']['max_pair_domain_A'],
                'work_uncertainty_from_floor_Mcycles':abs(out['work']['dW_dt_gain'][0])*math.hypot(floor,floor),
                'note':'the A/A floor is what an identical arm repeated costs. A leg whose work signal is inside the floor-implied uncertainty of its two sizes cannot support an exponent claim.'}
            out['work_check']={'W_positive':out['work']['W_Mcycles']>0,
                'F_between_zero_and_fastest_fold':0.0<out['work']['F_s']<cells[str(hi)]['median_s'],
                'two_clocks_only':'with exactly two arms the inverse-clock fit is the closed form: the residual is zero BY CONSTRUCTION and carries no information, so no goodness-of-fit is claimed here. c10-fixed-cost tested the T = F + W/f form against four clocks at 298 and 512 aa and it held; this row assumes it rather than re-testing it.',
                'se_W_Mcycles':out['work']['se_W_Mcycles'],'se_F_s':out['work']['se_F_s']}
            if not (out['work_check']['W_positive'] and out['work_check']['F_between_zero_and_fastest_fold']):ok=False
            if size==512:
                a=criterion_anchor(criterion)
                dt=cells[str(hi)]['median_s'];dW=out['work']['W_Mcycles']
                tol_W=2*math.hypot(out['work']['se_W_Mcycles'],a['W_stated_error_Mcycles'])
                out['anchor']={'t_1350_s':dt,'t_1350_allowed_s':a['t_1350_allowed_s'],
                    't_1350_reproduces':a['t_1350_allowed_s'][0]<=dt<=a['t_1350_allowed_s'][1],
                    't_800_s':cells[str(lo)]['median_s'],'t_800_allowed_s':a['t_800_allowed_s'],
                    't_800_reproduces':a['t_800_allowed_s'][0]<=cells[str(lo)]['median_s']<=a['t_800_allowed_s'][1],
                    'W_Mcycles':dW,'W_of_record_Mcycles':a['W_Mcycles'],'W_delta_Mcycles':dW-a['W_Mcycles'],
                    'W_tolerance_Mcycles':tol_W,'W_reproduces':abs(dW-a['W_Mcycles'])<=tol_W,
                    'F_s':out['work']['F_s'],'F_of_record_s':a['F_s'],
                    'F_delta_s':out['work']['F_s']-a['F_s'],
                    'F_tolerance_s':2*math.hypot(out['work']['se_F_s'],a['F_stated_error_s']),
                    'consequence':'if the anchor does not reproduce, nothing else in this session may be quoted'}
                out['anchor']['reproduces']=(out['anchor']['t_1350_reproduces'] and out['anchor']['t_800_reproduces']
                                             and out['anchor']['W_reproduces'])
                if not abs(out['work']['F_s']-a['F_s'])<=out['anchor']['F_tolerance_s']:
                    out['anchor']['F_note']='F is outside the combined error of the number of record. F is not the anchor criterion (W and the two cell medians are) but the disagreement is recorded rather than hidden.'
                if not out['anchor']['reproduces']:ok=False
        out['verdict']='GO' if ok else 'STOP'
        out['run_errors']=run.get('errors',[])
        result['targets'][str(size)]=out
        if not ok:result['verdict']='STOP'
    result['ladder']=ladder(result,json.loads((HERE/'criterion.json').read_text()))
    if result['ladder'].get('quotable') is False and result['verdict']=='GO':
        result['verdict']='STOP'
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    result=reduce(a.run);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{x:y for x,y in v.items() if x in ['verdict','accepted_folds','cells','work','work_check','aa_floor','anchor','run_errors']} for k,v in result['targets'].items()}|{'ladder':result.get('ladder'),'verdict':result['verdict']},indent=2))
    sys.exit(0 if result['verdict']=='GO' else 2)
