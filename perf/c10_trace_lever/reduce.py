"""CPU replay of the interleaved diffusion_trace arms: per-cell timing, the paired delta, the
clock-immune test and the digest comparison.

Row validation is c10-fixed-cost's, unchanged, plus this row's own step-counter and arm checks.
Nothing here touches a device; it reads the committed capture and re-derives every number.
"""
from __future__ import annotations
import argparse, gzip, hashlib, json, math, statistics, sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT/'perf/other512')]
from clockarm import coverage
from audit_accuracy import validate_atoms
from cif_rmsd import read_atoms, kabsch_rmsd

AA_FLOOR_S={'512':0.055,'298':0.043}   # c10-bare-baseline's adjacent A/A pair deltas

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

def score(a,b,size):
    ka,xa=coordinates(a,size);kb,xb=coordinates(b,size)
    if ka!=kb:raise ValueError('atom identities differ')
    seq=np.array([int(k[1]) for k in ka]);masks=[seq<=298]
    if size==512:masks.append(seq>298)
    dom=[kabsch_rmsd(xa[m],xb[m]) for m in masks]
    return dict(atoms=len(ka),whole_chain_all_atom_A=kabsch_rmsd(xa,xb),domain_all_atom_A=dom,
                max_domain_all_atom_A=max(dom),cif_byte_exact=sha(a)==sha(b))

def adjacent(vals):
    return max((abs(b-a) for a,b in zip(vals,vals[1:])),default=None)

def cell_stats(vals):
    return dict(n=len(vals),median=statistics.median(vals),mean=statistics.fmean(vals),
                stdev=statistics.stdev(vals) if len(vals)>1 else 0.0,
                min=min(vals),max=max(vals),adjacent_max_abs_delta=adjacent(vals))

def reduce(root):
    root=Path(root);criterion=json.loads((HERE/'criterion.json').read_text())
    prediction=json.loads((HERE/'prediction.json').read_text())
    result={'verdict':'GO','targets':{},
            'scope':'What the ttnn diffusion trace removes from the bare Boltz-2 fold, from arms interleaved in one process at two pinned clocks. No device-cycle census, no per-op attribution.',
            'prediction_digest':hashlib.sha256((HERE/'prediction.json').read_bytes()).hexdigest(),
            'predicted_delta_s':prediction['predicted_delta_s']}
    limit=criterion['quiet']['host_cpu_witness']['foreign_cpu_reject_pct']
    ambient=lines(root/'ambient.jsonl') if (root/'ambient.jsonl').exists() or (root/'ambient.jsonl.gz').exists() else None
    result['host_cpu_witness']='recorded' if ambient is not None else 'not recorded in this run'
    for size in criterion['model']['sizes']:
        d=root/str(size);T={}
        result['targets'][str(size)]=T
        if not (d/'result.json').exists():
            T.update(verdict='STOP',reason='target not run');result['verdict']='STOP';continue
        run=json.loads((d/'result.json').read_text())
        if run.get('smoke'):
            T.update(verdict='STOP',reason='smoke capture is not a measurement');result['verdict']='STOP';continue
        samples=lines(d/'clock.jsonl');holders=lines(d/'holders.jsonl')
        pre=[]
        if run.get('production_diff'):pre.append('production diff')
        if run.get('source_base')!=criterion['source_base']:pre.append('source base')
        if run.get('sampling_steps')!=criterion['model']['sampling_steps']:pre.append('sampling steps')
        if run.get('trace_region_size_reported')!=criterion['lever']['region_bytes']:pre.append('trace region')
        if run.get('model_predict_args')!={'recycling_steps':3,'sampling_steps':200,'diffusion_samples':1,'max_parallel_samples':None}:pre.append('model config')
        if run.get('release_response',[None])[0]!=0 or run.get('sampler_returncode')!=0:pre.append('release/sampler')
        if not run.get('completed'):pre.append('capture did not complete')
        for snap in [run.get('before',{}),run.get('after',{})]+[r[k] for r in run['rows'] for k in ['before','after']]:
            if snap.get('boot_id')!=run['before']['boot_id'] or snap.get('containment')!='active' or snap.get('module_srcversion')!='A10759A24565BC5BBE903C5':pre.append('boot/containment')
            if any(h['pid']!=run['pid'] for h in snap.get('holders',[])):pre.append('foreign holder snapshot')
        if run.get('after',{}).get('own_nodes'):pre.append('device remains open')
        steps=criterion['model']['sampling_steps']
        records=[];accepted=[]
        for r in run['rows']:
            clk=r['clock_MHz'];arm=r['arm'];other='untraced' if arm=='traced' else 'traced'
            clock=coverage(samples,r,clk);hc=holder_coverage(holders,r,run['pid']);amb=ambient_window(ambient,r,limit)
            rr=dict(r,clock=clock,holder_coverage=hc,ambient=amb);rr.pop('before',None);rr.pop('after',None);rr.pop('metrics',None)
            f=list(pre)
            if clk not in criterion['clock']['arms_MHz']:f.append('clock not an authorised arm')
            if not r.get('settle',{}).get('settled'):f.append('clock did not settle before the fold')
            if r.get('force_response',[None])[0]!=0:f.append('FORCE_AICLK not accepted')
            if abs(r['elapsed_s']-(r['end_monotonic_ns']-r['start_monotonic_ns'])/1e9)>1e-12:f.append('timer mismatch')
            if r.get('above_cap_sdpa_counts')!=[0,0]:f.append('above-cap route')
            if not math.isfinite(float(r.get('plddt',float('nan')))):f.append('nonfinite confidence')
            if r.get('diffusion_trace')!=(arm=='traced'):f.append('arm flag disagrees with label')
            if r.get('score_calls',{}).get(arm)!=steps:f.append(f'not {steps} calls down {arm}')
            if r.get('score_calls',{}).get(other)!=0:f.append(f'leaked calls down {other}')
            if (arm=='traced')!=bool(r.get('trace_present_after')):f.append('trace presence disagrees with arm')
            for name,val in [('clock',clock['pass']),('holders',hc['passed']),('run_valid',r.get('valid',False)),
                             ('foreign host CPU',amb['passed'] if amb['recorded'] else True)]:
                if not val:f.append(name)
            rr['failures']=sorted(set(f));rr['accepted']=(not f) and r['label']!='cold'
            records.append(rr)
            if rr['accepted']:accepted.append(rr)
        T['rows']=records;T['n_accepted']=len(accepted)
        T['warmup_discarded']=[r['label'] for r in records if r['label']=='cold']
        if not accepted:
            T.update(verdict='STOP',reason='no accepted fold');result['verdict']='STOP';continue
        cells={}
        for clk in criterion['clock']['arms_MHz']:
            for arm in ['untraced','traced']:
                vals=[r['elapsed_s'] for r in accepted if r['clock_MHz']==clk and r['arm']==arm]
                if vals:cells[f'{clk}_{arm}']=cell_stats(vals)
        T['cells']=cells
        T['session_AA_floor_s']=max((c['adjacent_max_abs_delta'] for c in cells.values() if c['adjacent_max_abs_delta'] is not None),default=None)
        T['committed_AA_floor_s']=AA_FLOOR_S[str(size)]
        # paired deltas: consecutive accepted (untraced, traced) at the same clock, either order
        pairs=[]
        for i in range(len(accepted)-1):
            x,y=accepted[i],accepted[i+1]
            if x['clock_MHz']!=y['clock_MHz'] or x['arm']==y['arm']:continue
            u,t=(x,y) if x['arm']=='untraced' else (y,x)
            pairs.append({'clock_MHz':x['clock_MHz'],'untraced':u['label'],'traced':t['label'],
                          'delta_s':u['elapsed_s']-t['elapsed_s'],
                          'host_cpu_delta_s':u['host_cpu_s']-t['host_cpu_s']})
        T['pairs']=pairs
        deltas={}
        for clk in criterion['clock']['arms_MHz']:
            cu,ct=cells.get(f'{clk}_untraced'),cells.get(f'{clk}_traced')
            if not (cu and ct):continue
            pd=[p['delta_s'] for p in pairs if p['clock_MHz']==clk]
            dm=cu['median']-ct['median']
            deltas[str(clk)]={'n_untraced':cu['n'],'n_traced':ct['n'],
                'untraced_median_s':cu['median'],'traced_median_s':ct['median'],
                'delta_s':dm,'delta_Mcycles':dm*clk,'speedup':cu['median']/ct['median'],
                'paired_n':len(pd),'paired_median_delta_s':statistics.median(pd) if pd else None,
                'paired_min_s':min(pd) if pd else None,'paired_max_s':max(pd) if pd else None,
                'host_cpu_delta_s':cu.get('n') and (statistics.median([p['host_cpu_delta_s'] for p in pairs if p['clock_MHz']==clk]) if pd else None),
                'exceeds_committed_AA_floor':abs(dm)>AA_FLOOR_S[str(size)],
                'exceeds_session_AA_floor':T['session_AA_floor_s'] is None or abs(dm)>T['session_AA_floor_s']}
        T['deltas']=deltas
        # clock-immune vs clock-scaled: a host-dispatch gain is constant in SECONDS across clocks
        hi,lo=str(criterion['primary_clock_MHz']),None
        for c in criterion['clock']['arms_MHz']:
            if str(c)!=hi and str(c) in deltas:lo=str(c)
        if lo and hi in deltas:
            dh,dl=deltas[hi]['delta_s'],deltas[lo]['delta_s']
            scaled=dh*int(hi)/int(lo)
            T['mechanism']={'high_MHz':int(hi),'low_MHz':int(lo),'delta_high_s':dh,'delta_low_s':dl,
                'if_clock_immune_delta_low_would_be_s':dh,
                'if_clock_scaled_delta_low_would_be_s':scaled,
                'observed_minus_immune_s':dl-dh,'observed_minus_scaled_s':dl-scaled,
                'clock_immune_fraction':(scaled-dl)/(scaled-dh) if abs(scaled-dh)>1e-12 else None,
                'verdict':('clock-immune (host dispatch)' if abs(dl-dh)<abs(dl-scaled) else 'clock-scaled (device work)')}
        # parity: digests and geometry
        du=sorted({r['cif_sha256'] for r in accepted if r['arm']=='untraced'})
        dt=sorted({r['cif_sha256'] for r in accepted if r['arm']=='traced'})
        par={'distinct_untraced_digests':len(du),'distinct_traced_digests':len(dt),
             'untraced_digest':du[0] if len(du)==1 else du,'traced_digest':dt[0] if len(dt)==1 else dt,
             'byte_identical_across_arms':du==dt and len(du)==1}
        ru=next(r for r in accepted if r['arm']=='untraced');rt=next(r for r in accepted if r['arm']=='traced')
        par['geometry']=score(d/ru['cif'],d/rt['cif'],size)
        par['bar_A']=criterion['accuracy']['bars_A'][str(size)]
        par['seed_floor_A']=(criterion['accuracy']['user_512_seed_floor_A'] if size==512
                             else criterion['accuracy']['archived_upstream_seed_spread_A']['298'])
        par['within_bar']=par['geometry']['max_domain_all_atom_A']<=par['bar_A']
        par['plddt']={'untraced':sorted({r['plddt'] for r in accepted if r['arm']=='untraced'}),
                      'traced':sorted({r['plddt'] for r in accepted if r['arm']=='traced'})}
        T['parity']=par
        T['per_arm_consistency']={'untraced_one_digest':len(du)==1,'traced_one_digest':len(dt)==1}
        # verdict for this size
        v='GO'
        if hi not in deltas:v='STOP'
        elif deltas[hi]['n_untraced']<criterion['minimum_folds_per_arm_at_primary_clock'] or deltas[hi]['n_traced']<criterion['minimum_folds_per_arm_at_primary_clock']:v='STOP'
        elif not deltas[hi]['exceeds_committed_AA_floor']:v='NO-GO'
        elif not (par['byte_identical_across_arms'] or par['within_bar']):v='NO-GO'
        T['verdict']=v
        T['refutation_fired']=[] if v=='GO' else [('gain inside the A/A floor' if v=='NO-GO' else 'insufficient accepted folds or missing cell')]
        if v!='GO':result['verdict']=v if result['verdict']=='GO' else result['verdict']
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('--out',type=Path)
    a=ap.parse_args()
    out=reduce(a.run)
    txt=json.dumps(out,indent=2,default=float)+'\n'
    (a.out or a.run/'analysis.json').write_text(txt)
    print(txt)
    return 0 if out['verdict'] in ('GO','NO-GO') else 3

if __name__=='__main__':sys.exit(main())
