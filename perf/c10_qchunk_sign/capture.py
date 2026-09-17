"""One warmed device context per size; TT_BIO_SDPA_GRID_Q_CHUNK interleaved ABBA inside it.

Reuses c10-bare-baseline's timer boundary, clock sampler, holder census, quiet gate and float64
scorer unchanged. The only thing added is the arm switch and the per-call-site chunk census that
proves the switch reached the device.
"""
from __future__ import annotations
import argparse, gzip, hashlib, importlib.metadata, json, math, os, shutil, signal, socket, subprocess, sys, time, traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT),str(HERE),str(ROOT/'scripts/gpu_vs_tt'),str(ROOT/'perf/other512')]
from control import coverage, digest, holders, own_nodes, snapshot, validate_snapshot, write_json
from force_aiclk import smc, FORCE_AICLK
from fold_reduce import holder_coverage, coordinates, score, lines

def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True)
def quiet(s,opened=False):
    validate_snapshot(s,opened)
    if any(h['pid']!=os.getpid() for h in s['holders']):raise RuntimeError('foreign holder on shared host')

def binaries():
    paths=set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        p=line.split()[-1]
        if p.startswith('/') and '.so' in p and any(x in p.lower() for x in ['ttnn','tt_metal','tt-metal','libumd','tracy']):paths.add(p)
    return [digest(p) for p in sorted(paths)]

def census_rows(T):
    out=[]
    for (site,q_len,k_len,work,d),(calls,shipped,chunk,units) in sorted(T.SDPA_GRID_Q_CHUNK_PICKS.items()):
        out.append(dict(site=site,q_len=q_len,k_len=k_len,work=work,d=d,calls=calls,
                        shipped_chunk=shipped,chunk=chunk,units=units,moved=chunk!=shipped))
    return out

def schedule(reps):
    """ABBA: `on off off on` per rep. The lever arm sits at positions 1 and 4 of every rep, so a
    linear drift across the rep cancels exactly in the rep contrast; the middle `off off` and the
    `on|on` rep boundary are the A/A control, drawn from this same sequence rather than a second
    session."""
    plan=[('census_on','on'),('census_off','off')]
    for i in range(reps):
        for j,arm in enumerate(['on','off','off','on']):plan.append((f'R{i}_{j}_{arm}',arm))
    return plan

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--size',type=int,choices=[298,512],required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    criterion=json.loads((HERE/'criterion.json').read_text())
    result=dict(pid=os.getpid(),host=socket.gethostname(),size=a.size,rows=[],errors=[],criterion=digest(HERE/'criterion.json'),source_commit=git('rev-parse','HEAD').strip(),source_base=criterion['source_base'],dirty_status=git('status','--short'),production_diff=git('diff',criterion['source_base'],'--','tt_bio','scripts/gpu_vs_tt/tt_baseline.py','perf/other512/fold_ab_multi.py'),started_utc_ns=time.time_ns(),completed=False)
    save=lambda:write_json(out/'result.json',result)
    sampler=None;fd=None;T=None;monitor_log=None
    def interrupted(sig,frame):raise RuntimeError(f'signal {sig}; release clock and device')
    for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]:signal.signal(sig,interrupted)
    try:
        result['before']=snapshot();quiet(result['before']);save()
        if socket.gethostname()!='tt-quietbox2':raise RuntimeError('wrong host')
        if os.getloadavg()[0]>2.0:raise RuntimeError('load prerequisite failed after benchlock wait')
        if result['production_diff']:raise RuntimeError('production source differs from requested base')
        result['dirty_diff']=git('diff','HEAD');(out/'dirty.patch').write_text(result['dirty_diff'])
        for key,expected in [('TT_VISIBLE_DEVICES','0'),('TT_BIO_LEASE_CARDS','0'),('TT_BIO_LEASE_HOLDER','worker:c10-qchunk-sign')]:
            if os.environ.get(key)!=expected:raise RuntimeError(f'wrong {key}')
        forbidden={k:v for k,v in os.environ.items() if (k.startswith('TT_BIO_') and k not in {'TT_BIO_LEASE_CARDS','TT_BIO_LEASE_HOLDER'}) or k.startswith('TT_METAL_PROFILER') or k.startswith('TRACY')}
        if forbidden:raise RuntimeError(f'non-default environment: {forbidden}')
        result['environment']={k:v for k,v in os.environ.items() if k.startswith(('TT_','OMP_','MKL_','PYTHONPATH','LD_LIBRARY_PATH','BENCHLOCK_'))}
        import torch,ttnn
        import tt_bio.tenstorrent as T
        import tt_baseline as B
        from fold_ab_multi import patch_boltz2_cfg
        result['ttnn_version']=importlib.metadata.version('ttnn');result['ttnn_path']=ttnn.__file__;result['tt_bio_path']=T.__file__;result['ttnn_config']=str(ttnn.CONFIG)
        if Path(T.__file__).resolve()!=ROOT/'tt_bio/tenstorrent.py':raise RuntimeError('wrong production source loaded')
        if '/site-packages/ttnn/' not in ttnn.__file__:raise RuntimeError('expected production wheel')
        for attr in ['enable_logging','enable_graph_report','enable_detailed_buffer_report','enable_detailed_tensor_report']:
            if getattr(ttnn.CONFIG,attr):raise RuntimeError(f'profiling enabled: {attr}')
        result['profiling']={'tracy_launched':False,'graph':False,'generic_observer':False,'module_timers':False,'sum_profiling':False,'runtime_env':{k:v for k,v in os.environ.items() if 'PROFIL' in k or 'TRACY' in k}}
        # The shipped default of the flag under test, read before anything touches it.
        result['flag']={'name':'TT_BIO_SDPA_GRID_Q_CHUNK','default_on':bool(T._SDPA_GRID_Q_CHUNK),'in_environ':'TT_BIO_SDPA_GRID_Q_CHUNK' in os.environ}
        if not result['flag']['default_on'] or result['flag']['in_environ']:raise RuntimeError('flag default is not the shipped ON, or it is set in the environment')
        root=Path('/sys/class/tenstorrent/tenstorrent!0')
        result['assigned_node_sysfs']={'resolved':str(root.resolve()),'device':str((root/'device').resolve()),'subsystem_device':(root/'device/subsystem_device').read_text().strip()}
        for name in ['tt_asic_id','tt_card_type','tt_fw_bundle_ver','tt_m3app_fw_ver','tt_serial']:
            if (root/name).exists():result['assigned_node_sysfs'][name]=(root/name).read_text().strip()
        B.RECYCLING_STEPS=3;B.SAMPLING_STEPS=200;B.DIFFUSION_SAMPLES=1;B.SEED=0
        patch_boltz2_cfg();B._card_info=lambda:{'assigned_node_sysfs':result['assigned_node_sysfs']}
        fixture=ROOT/f'perf/size512/fixtures/cdk2x2_{a.size}'
        target=fixture.with_suffix('.yaml');msa=fixture.with_suffix('.a3m')
        if 'templates:' in target.read_text():raise RuntimeError('templates in input')
        result['inputs']=[digest(target),digest(msa)]
        for p in [target,msa]:
            committed=subprocess.check_output(['git','show',criterion['source_base']+':'+str(p.relative_to(ROOT))],cwd=ROOT)
            if committed!=p.read_bytes():raise RuntimeError('fixture differs from committed input')
        unused_fold,meta,state=B.build_fold('boltz2',out/'msa',target,msa,instrument=False,hoist=False,fast=False,trace=False,recycling_steps=3)
        dev=T.get_device();result['opened']=snapshot();quiet(result['opened'],True)
        if own_nodes()!=['/dev/tenstorrent/0']:raise RuntimeError('wrong actual opened device')
        result['model_meta']=meta;result['model_predict_args']=dict(state.model.predict_args)
        result['acquisition_sources']=[digest(p) for p in sorted(HERE.glob('*.py'))]
        expected={'recycling_steps':3,'sampling_steps':200,'diffusion_samples':1,'max_parallel_samples':None}
        if dict(state.model.predict_args)!=expected:raise RuntimeError('actual model config mismatch')
        if meta['n_msa']!=35:raise RuntimeError('MSA rows not 35')
        result['msa_cache']=[digest(p) for p in (out/'msa').glob('*.a3m')]
        if len(result['msa_cache'])!=1 or result['msa_cache'][0]['sha256']!=result['inputs'][1]['sha256']:raise RuntimeError('MSA cache bytes mismatch')
        result['loaded_binaries']=binaries();result['sdpa_cap']=T._triatt_sdpa._Q_SPLIT_MAX_S
        if result['sdpa_cap']!=1024 or not T._SDPA_FUSED_LARGE_S:raise RuntimeError('SDPA defaults differ')
        result['compute_grid_main']=list(T.COMPUTE_GRID_MAIN);result['cores']=T.COMPUTE_GRID_MAIN[0]*T.COMPUTE_GRID_MAIN[1]
        result['timer']='synchronize; monotonic_ns; complete state.predict_one incl. output writing; synchronize; monotonic_ns'

        def arm(name):
            T._SDPA_GRID_Q_CHUNK=(name=='on')
            T._grid_q_chunk.cache_clear();T._sdpa_q_chunk.cache_clear();T._sdpa_program_config_for_shape.cache_clear()
            if T._SDPA_GRID_Q_CHUNK!=(name=='on'):raise RuntimeError('arm switch did not take')

        fd=os.open('/dev/tenstorrent/0',os.O_RDWR|os.O_APPEND)
        result['force_response']=list(smc(fd,FORCE_AICLK,1350))
        if result['force_response'][0]!=0:raise RuntimeError('FORCE_AICLK failed')
        monitor_log=(out/'sampler.log').open('w')
        sampler=subprocess.Popen([sys.executable,str(HERE/'control.py'),str(out/'clock.jsonl'),str(os.getpid())],stdin=subprocess.PIPE,stdout=monitor_log,stderr=subprocess.STDOUT)
        time.sleep(.2);save()
        plan=schedule(criterion['reps']);result['schedule']=[list(x) for x in plan];result['census']={}
        reference=None
        for label,armname in plan:
            census=label.startswith('census')
            arm(armname)
            T.SDPA_GRID_Q_CHUNK_CENSUS=census
            if census:T.SDPA_GRID_Q_CHUNK_PICKS.clear()
            before=snapshot();quiet(before,True)
            if dict(state.model.predict_args)!=expected:raise RuntimeError('model config changed between labels')
            if meta['job_cfg']['seed']!=0:raise RuntimeError('seed changed between labels')
            if before['boot_id']!=result['before']['boot_id']:raise RuntimeError('boot changed')
            struct_dir=Path(meta['struct_dir'])
            for p in struct_dir.glob('*'):p.unlink()
            T.SDPA_FUSED_LARGE_S_STATS[:]=[0,0]
            ttnn.synchronize_device(dev)
            start=time.monotonic_ns()
            metrics,best,feats=state.predict_one(target,meta['job_cfg'])
            ttnn.synchronize_device(dev)
            end=time.monotonic_ns()
            row=dict(label=label,arm=armname,census=census,start_monotonic_ns=start,end_monotonic_ns=end,elapsed_s=(end-start)/1e9,metrics=metrics,plddt=metrics.get('plddt'),before=before,after=snapshot(),above_cap_sdpa_counts=list(T.SDPA_FUSED_LARGE_S_STATS),flag_during_fold=bool(T._SDPA_GRID_Q_CHUNK),valid=False)
            result['rows'].append(row)
            if census:
                result['census'][armname]=census_rows(T);T.SDPA_GRID_Q_CHUNK_CENSUS=False
            keep=out/'cifs'/label;keep.mkdir(parents=True)
            for p in struct_dir.glob('*'):
                if p.is_file():shutil.copyfile(p,keep/p.name)
            cifs=list(keep.glob('*.cif'))
            if len(cifs)!=1:raise RuntimeError('expected one CIF')
            row['cif']=str(cifs[0].relative_to(out));row['cif_sha256']=digest(cifs[0])['sha256'];coordinates(cifs[0],a.size)
            if row['plddt'] is None or not math.isfinite(float(row['plddt'])):raise RuntimeError('nonfinite confidence')
            if row['flag_during_fold']!=(armname=='on'):raise RuntimeError('flag moved during the fold')
            quiet(row['after'],True)
            if row['after']['boot_id']!=result['before']['boot_id']:raise RuntimeError('boot changed')
            if row['above_cap_sdpa_counts']!=[0,0]:raise RuntimeError('above-cap route unexpectedly reached')
            time.sleep(.15)
            row['clock']=coverage(lines(out/'clock.jsonl'),row);row['holders']=holder_coverage(lines(out/'holders.jsonl'),row,os.getpid())
            if reference is None:reference=cifs[0]
            row['structure_vs_reference']=score(reference,cifs[0],a.size)
            if not row['clock']['pass']:raise RuntimeError('clock artifact or incomplete clock coverage')
            if not row['holders']['passed']:raise RuntimeError('co-tenancy or incomplete holder coverage')
            row['valid']=True;save()
            print(json.dumps({'label':label,'arm':armname,'elapsed_s':round(row['elapsed_s'],3),'MHz':row['clock']['min_MHz'],'cif':row['cif_sha256'][:12],'plddt':row['plddt'],'dA':round(row['structure_vs_reference']['max_domain_all_atom_A'],4)}),flush=True)
            del metrics,best,feats
        # both arms must have been seen, and the census must show the switch reached the device
        moved={k:sorted({r['site'] for r in v if r['moved']}) for k,v in result['census'].items()}
        result['census_summary']={k:{'calls':sum(r['calls'] for r in v),'calls_moved':sum(r['calls'] for r in v if r['moved']),'sites_moved':moved[k],'sites_declined':sorted({r['site'] for r in v if not r['moved']})} for k,v in result['census'].items()}
        if moved.get('off'):raise RuntimeError('the rule moved a site with the flag OFF')
        if not moved.get('on'):raise RuntimeError('the rule moved nothing with the flag ON: the arm switch never reached the device')
        result['completed']=True
    except BaseException as e:
        result['errors'].append(repr(e));traceback.print_exc()
    finally:
        if T is not None:
            try:T.SDPA_GRID_Q_CHUNK_CENSUS=False;T._SDPA_GRID_Q_CHUNK=True;T._grid_q_chunk.cache_clear();T._sdpa_q_chunk.cache_clear();T._sdpa_program_config_for_shape.cache_clear()
            except BaseException as e:result['errors'].append('flag restore: '+repr(e))
        if fd is not None:
            try:result['release_response']=list(smc(fd,FORCE_AICLK,0))
            except BaseException as e:result['errors'].append('release: '+repr(e));result['completed']=False
            finally:os.close(fd)
        if sampler is not None:
            try:sampler.communicate(b'stop\n',timeout=15)
            except subprocess.TimeoutExpired:sampler.terminate();sampler.wait(timeout=5)
            result['sampler_returncode']=sampler.returncode
            if sampler.returncode:result['completed']=False
        if monitor_log:monitor_log.close()
        if T is not None:
            try:T.cleanup()
            except BaseException as e:result['errors'].append('cleanup: '+repr(e));result['completed']=False
        try:
            result['after']=snapshot()
            validate_snapshot(result['after'])
            if result['after']['own_nodes']:raise RuntimeError('device still open after cleanup')
            if result.get('release_response',[None])[0]!=0:raise RuntimeError('clock release not confirmed')
            if result['after']['boot_id']!=result['before']['boot_id']:raise RuntimeError('boot changed')
        except BaseException as e:result['errors'].append('snapshot: '+repr(e));result['completed']=False
        for name in ['clock.jsonl','holders.jsonl']:
            p=out/name
            if p.exists():
                data=p.read_bytes();(out/(name+'.gz')).write_bytes(gzip.compress(data,mtime=0));p.unlink()
        result['finished_utc_ns']=time.time_ns();save()
    return 0 if result['completed'] else 2

if __name__=='__main__':sys.exit(main())
