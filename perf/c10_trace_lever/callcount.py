"""Call census, NOT a timing measurement: does the traced arm actually issue fewer ttnn calls?

The A/B measured no gain. That has two very different explanations -- trace replay engaged and
the removed host dispatch was never on the critical path, or trace replay silently did nothing.
`ttnn.deallocate` alone is 122,112 of the fold's top-level ttnn calls and is issued inside the
per-step DiT stream, so counting it separates the two. The counter is a python wrapper on the
module attribute, installed for both arms, and it perturbs timing, which is why this runs as its
own script and its elapsed seconds are NOT reported as a measurement anywhere.

Also folds the largest committed fixture with the 1 GiB trace region reserved, which is the
OOM hard-stop check: the region must not push a size the product serves out of memory.
"""
from __future__ import annotations
import argparse, json, os, socket, sys, time, traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT),str(HERE),str(ROOT/'scripts/gpu_vs_tt'),str(ROOT/'perf/other512')]
from control import own_nodes, snapshot, validate_snapshot, write_json

TRACE_REGION_BYTES=1<<30

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--size',type=int,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--arms',default='untraced,traced')
    ap.add_argument('--region',type=int,default=TRACE_REGION_BYTES)
    a=ap.parse_args()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    res=dict(kind='call census, not a timing measurement',pid=os.getpid(),host=socket.gethostname(),
             size=a.size,region_bytes=a.region,rows=[],errors=[],completed=False)
    save=lambda:write_json(out/'callcount.json',res)
    T=None
    try:
        res['before']=snapshot();validate_snapshot(res['before'])
        import torch,ttnn
        import tt_bio.tenstorrent as T
        import tt_baseline as B
        from fold_ab_multi import patch_boltz2_cfg
        dev=T.get_device(trace_region_size=a.region)
        res['trace_region_size_reported']=T.trace_region_size()
        if own_nodes()!=['/dev/tenstorrent/0']:raise RuntimeError('wrong opened device')
        B.RECYCLING_STEPS=3;B.SAMPLING_STEPS=200;B.DIFFUSION_SAMPLES=1;B.SEED=0
        patch_boltz2_cfg();B._card_info=lambda:{}
        fixture=ROOT/f'perf/size512/fixtures/cdk2x2_{a.size}'
        _f,meta,state=B.build_fold('boltz2',out/'msa',fixture.with_suffix('.yaml'),fixture.with_suffix('.a3m'),
                                   instrument=False,hoist=False,fast=False,trace=False,recycling_steps=3)
        ad=state.model.structure_module
        n={'deallocate':0}
        real=ttnn.deallocate
        def counted(*args,**kw):
            n['deallocate']+=1
            return real(*args,**kw)
        ttnn.deallocate=counted
        res['peak_dram_probe']=str(dev)
        for arm in a.arms.split(','):
            ad._diffusion_trace=(arm=='traced')
            n['deallocate']=0
            ttnn.synchronize_device(dev)
            t0=time.monotonic_ns()
            metrics,best,feats=state.predict_one(fixture.with_suffix('.yaml'),meta['job_cfg'])
            ttnn.synchronize_device(dev)
            t1=time.monotonic_ns()
            row=dict(arm=arm,deallocate_calls=n['deallocate'],
                     instrumented_elapsed_s=(t1-t0)/1e9,
                     note='elapsed is perturbed by the counter and is NOT a measurement',
                     plddt=metrics.get('plddt'),
                     trace_present_after=getattr(ad.score_model,'_diff_trace',None) is not None)
            res['rows'].append(row);save()
            print(json.dumps(row),flush=True)
            del metrics,best,feats
        ttnn.deallocate=real
        res['completed']=True
    except BaseException as e:
        res['errors'].append(repr(e));traceback.print_exc()
    finally:
        if T is not None:
            try:T.cleanup()
            except BaseException as e:res['errors'].append('cleanup: '+repr(e))
        save()
    return 0 if res['completed'] else 2

if __name__=='__main__':sys.exit(main())
