"""Interleaved core_grid A/B on one warmed device context, node 1, pinned 1350 MHz.

Arms are defined by a spec file: an ordered list of {name, lines}. `A` must be first and carry no
lines, so it is the untouched default path and doubles as this node`s anchor against the 14.881 s
number of record. Labels cycle A, <arm>, A, <arm>, ... so every arm is adjacent to a fresh A and
no arm can win on drift. The wrapper from arm.py is installed for every arm including A.
"""
from __future__ import annotations
import argparse, gzip, importlib.metadata, json, math, os, shutil, signal, socket, subprocess, sys, time, traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT),str(HERE),str(ROOT/"scripts/gpu_vs_tt"),str(ROOT/"perf/other512")]
import arm as ARMS
from control import coverage, digest, holders, own_nodes, snapshot, validate_snapshot, write_json, DEV, NODE
from force_aiclk import smc, FORCE_AICLK
from reduce import coordinates, score, lines


def holder_cov(observations, interval, pid):
    """holder_coverage, but scoped to the assigned node.

    The shared helper rejects any holder of /dev/tenstorrent/0 anywhere on the host. This row is
    granted node 1 and a sibling worker legitimately holds another card, so node scoping is the
    correction, not a loosening: a foreign holder of OUR node still rejects, and every other
    holder is recorded so the session says what it ran beside."""
    start, end = interval["start_monotonic_ns"], interval["end_monotonic_ns"]
    rows = [r for r in observations if start <= r["monotonic_ns"] <= end]
    points = [start] + [r["monotonic_ns"] for r in rows] + [end]
    bad = [r for r in rows if r.get("error") or r.get("owner_nodes") != [DEV]
           or any(DEV in h["nodes"] and h["pid"] != pid for h in r.get("holders", []))]
    others = sorted({h["pid"] for r in rows for h in r.get("holders", []) if h["pid"] != pid})
    gap = max(b - a for a, b in zip(points, points[1:]))
    return dict(samples=len(rows), max_gap_ns=gap, bad=bad, other_node_holder_pids=others,
                passed=bool(rows) and not bad and gap <= 1_000_000_000)

def git(*a):return subprocess.check_output(["git",*a],cwd=ROOT,text=True)
CO_TENANTS=[]
def quiet(s,opened=False):
    # validate_snapshot already refuses a foreign holder on OUR node. A sibling worker holding a
    # DIFFERENT card is not a reason to refuse: this row runs interleaved arms inside one session,
    # so any co-tenant load is carried equally by both arms. Record it; do not pretend it is absent.
    validate_snapshot(s,opened)
    foreign=[h for h in s["holders"] if h["pid"]!=os.getpid()]
    if foreign:CO_TENANTS.append({"monotonic_ns":s["monotonic_ns"],"holders":foreign})

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--size",type=int,choices=[298,512],required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--arms",type=Path,required=True)
    ap.add_argument("--reps",type=int,default=3)
    a=ap.parse_args()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    criterion=json.loads((HERE/"criterion.json").read_text())
    spec=json.loads(a.arms.read_text())
    if spec["arms"][0]["name"]!="A" or spec["arms"][0]["lines"]:raise RuntimeError("arm A must be the untouched default")
    result=dict(pid=os.getpid(),host=socket.gethostname(),node=NODE,size=a.size,reps=a.reps,rows=[],errors=[],
        arms=spec,arms_digest=digest(a.arms),criterion=digest(HERE/"criterion.json"),
        source_commit=git("rev-parse","HEAD").strip(),source_base=criterion["source_base"],
        dirty_status=git("status","--short"),
        production_diff=git("diff",criterion["source_base"],"--","tt_bio","scripts/gpu_vs_tt/tt_baseline.py","perf/other512/fold_ab_multi.py"),
        started_utc_ns=time.time_ns(),completed=False)
    save=lambda:write_json(out/"result.json",result)
    sampler=None;fd=None;T=None;monitor_log=None
    def interrupted(sig,frame):raise RuntimeError(f"signal {sig}; release clock and device")
    for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]:signal.signal(sig,interrupted)
    try:
        result["before"]=snapshot();quiet(result["before"]);save()
        if socket.gethostname()!="tt-quietbox2":raise RuntimeError("wrong host")
        if os.getloadavg()[0]>2.0:raise RuntimeError("load prerequisite failed after benchlock wait")
        if result["production_diff"]:raise RuntimeError("production source differs from requested base")
        result["dirty_diff"]=git("diff","HEAD");(out/"dirty.patch").write_text(result["dirty_diff"])
        for key,expected in [("TT_VISIBLE_DEVICES",NODE),("TT_BIO_LEASE_CARDS",NODE),("TT_BIO_LEASE_HOLDER","worker:c10-core-grid")]:
            if os.environ.get(key)!=expected:raise RuntimeError(f"wrong {key}")
        forbidden={k:v for k,v in os.environ.items() if (k.startswith("TT_BIO_") and k not in {"TT_BIO_LEASE_CARDS","TT_BIO_LEASE_HOLDER"}) or k.startswith("TT_METAL_PROFILER") or k.startswith("TRACY")}
        if forbidden:raise RuntimeError(f"non-default environment: {forbidden}")
        result["environment"]={k:v for k,v in os.environ.items() if k.startswith(("TT_","OMP_","MKL_","PYTHONPATH","LD_LIBRARY_PATH","BENCHLOCK_","C10_"))}
        import torch,ttnn
        import tt_bio.tenstorrent as T
        import tt_baseline as B
        from fold_ab_multi import patch_boltz2_cfg
        result["ttnn_version"]=importlib.metadata.version("ttnn");result["ttnn_path"]=ttnn.__file__;result["tt_bio_path"]=T.__file__
        if Path(T.__file__).resolve()!=ROOT/"tt_bio/tenstorrent.py":raise RuntimeError("wrong production source loaded")
        if "/site-packages/ttnn/" not in ttnn.__file__:raise RuntimeError("expected production wheel")
        for attr in ["enable_logging","enable_graph_report","enable_detailed_buffer_report","enable_detailed_tensor_report"]:
            if getattr(ttnn.CONFIG,attr):raise RuntimeError(f"profiling enabled: {attr}")
        ARMS.install(ttnn,T)
        result["core_grid_main"]={"x":T.CORE_GRID_MAIN.x,"y":T.CORE_GRID_MAIN.y}
        root=Path(f"/sys/class/tenstorrent/tenstorrent!{NODE}")
        result["assigned_node_sysfs"]={"resolved":str(root.resolve()),"device":str((root/"device").resolve())}
        for name in ["tt_asic_id","tt_card_type","tt_fw_bundle_ver","tt_m3app_fw_ver","tt_serial"]:
            if (root/name).exists():result["assigned_node_sysfs"][name]=(root/name).read_text().strip()
        B.RECYCLING_STEPS=3;B.SAMPLING_STEPS=200;B.DIFFUSION_SAMPLES=1;B.SEED=0
        patch_boltz2_cfg();B._card_info=lambda:{"assigned_node_sysfs":result["assigned_node_sysfs"]}
        fixture=ROOT/f"perf/size512/fixtures/cdk2x2_{a.size}"
        target=fixture.with_suffix(".yaml");msa=fixture.with_suffix(".a3m")
        if "templates:" in target.read_text():raise RuntimeError("templates in input")
        result["inputs"]=[digest(target),digest(msa)]
        unused,meta,state=B.build_fold("boltz2",out/"msa",target,msa,instrument=False,hoist=False,fast=False,trace=False,recycling_steps=3)
        dev=T.get_device();result["opened"]=snapshot();quiet(result["opened"],True)
        if own_nodes()!=[DEV]:raise RuntimeError(f"wrong actual opened device: {own_nodes()}")
        result["model_meta"]=meta;result["model_predict_args"]=dict(state.model.predict_args)
        result["acquisition_sources"]=[digest(p) for p in sorted(HERE.glob("*.py"))]
        expected={"recycling_steps":3,"sampling_steps":200,"diffusion_samples":1,"max_parallel_samples":None}
        if dict(state.model.predict_args)!=expected:raise RuntimeError("actual model config mismatch")
        if meta["n_msa"]!=35:raise RuntimeError("MSA rows not 35")
        result["sdpa_cap"]=T._triatt_sdpa._Q_SPLIT_MAX_S
        if result["sdpa_cap"]!=1024 or not T._SDPA_FUSED_LARGE_S:raise RuntimeError("SDPA defaults differ")
        fd=os.open(DEV,os.O_RDWR|os.O_APPEND)
        result["force_response"]=list(smc(fd,FORCE_AICLK,1350))
        if result["force_response"][0]!=0:raise RuntimeError("FORCE_AICLK failed")
        monitor_log=(out/"sampler.log").open("w")
        sampler=subprocess.Popen([sys.executable,str(HERE/"control.py"),str(out/"clock.jsonl"),str(os.getpid())],stdin=subprocess.PIPE,stdout=monitor_log,stderr=subprocess.STDOUT)
        time.sleep(.2);save()
        by_name={x["name"]:x["lines"] for x in spec["arms"]}
        labels=[("cold","A")]+[(f"{n}{i}",n) for i in range(a.reps) for n in by_name]
        reference=None
        for label,armname in labels:
            before=snapshot();quiet(before,True)
            if dict(state.model.predict_args)!=expected:raise RuntimeError("model config changed between labels")
            if meta["job_cfg"]["seed"]!=0:raise RuntimeError("seed changed between labels")
            if before["boot_id"]!=result["before"]["boot_id"]:raise RuntimeError("boot changed")
            ARMS.set_arm(by_name[armname]);ARMS.record(label=="cold")
            struct_dir=Path(meta["struct_dir"])
            for p in struct_dir.glob("*"):p.unlink()
            T.SDPA_FUSED_LARGE_S_STATS[:]=[0,0]
            ttnn.synchronize_device(dev)
            start=time.monotonic_ns()
            metrics,best,feats=state.predict_one(target,meta["job_cfg"])
            ttnn.synchronize_device(dev)
            end=time.monotonic_ns()
            row=dict(label=label,arm=armname,start_monotonic_ns=start,end_monotonic_ns=end,
                elapsed_s=(end-start)/1e9,plddt=metrics.get("plddt"),before=before,after=snapshot(),
                injected=dict(ARMS.INJECTED),refused=dict(ARMS.REFUSED),
                above_cap_sdpa_counts=list(T.SDPA_FUSED_LARGE_S_STATS),valid=False)
            result["rows"].append(row)
            if label=="cold":
                ARMS.record(False)
                write_json(out/"site_census.json",dict(rows=ARMS.census_rows(),seen_lines=dict(ARMS.SEEN_LINES)))
                result["site_census"]=str(out/"site_census.json")
            keep=out/"cifs"/label;keep.mkdir(parents=True)
            for p in struct_dir.glob("*"):
                if p.is_file():shutil.copyfile(p,keep/p.name)
            cifs=list(keep.glob("*.cif"))
            if len(cifs)!=1:raise RuntimeError("expected one CIF")
            row["cif_sha256"]=digest(cifs[0])["sha256"];coordinates(cifs[0],a.size)
            if row["plddt"] is None or not math.isfinite(float(row["plddt"])):raise RuntimeError("nonfinite confidence")
            quiet(row["after"],True)
            if row["after"]["boot_id"]!=result["before"]["boot_id"]:raise RuntimeError("boot changed")
            if row["above_cap_sdpa_counts"]!=[0,0]:raise RuntimeError("above-cap route unexpectedly reached")
            time.sleep(.15)
            row["clock"]=coverage(lines(out/"clock.jsonl"),row);row["holders"]=holder_cov(lines(out/"holders.jsonl"),row,os.getpid())
            if reference is None and armname=="A" and label!="cold":reference=cifs[0]
            if reference is not None:row["structure_vs_A0"]=score(reference,cifs[0],a.size)
            if not row["clock"]["pass"]:raise RuntimeError("clock artifact or incomplete clock coverage")
            if not row["holders"]["passed"]:raise RuntimeError("co-tenancy or incomplete holder coverage")
            row["valid"]=True;save()
            print(json.dumps({"label":label,"arm":armname,"elapsed_s":round(row["elapsed_s"],4),
                "MHz":row["clock"]["min_MHz"],"injected":sum(row["injected"].values()),
                "refused":sum(row["refused"].values()),"cif":row["cif_sha256"][:12],
                "rmsd_A":(row.get("structure_vs_A0") or {}).get("max_domain_all_atom_A")}),flush=True)
            del metrics,best,feats
        result["completed"]=True
    except BaseException as e:
        result["errors"].append(repr(e));traceback.print_exc()
    finally:
        if fd is not None:
            try:result["release_response"]=list(smc(fd,FORCE_AICLK,0))
            except BaseException as e:result["errors"].append("release: "+repr(e));result["completed"]=False
            finally:os.close(fd)
        if sampler is not None:
            try:sampler.communicate(b"stop\n",timeout=15)
            except subprocess.TimeoutExpired:sampler.terminate();sampler.wait(timeout=5)
            result["sampler_returncode"]=sampler.returncode
            if sampler.returncode:result["completed"]=False
        if monitor_log:monitor_log.close()
        if T is not None:
            try:T.cleanup()
            except BaseException as e:result["errors"].append("cleanup: "+repr(e));result["completed"]=False
        try:
            result["after"]=snapshot();validate_snapshot(result["after"])
            if result["after"]["own_nodes"]:raise RuntimeError("device still open after cleanup")
            if result.get("release_response",[None])[0]!=0:raise RuntimeError("clock release not confirmed")
            if result["after"]["boot_id"]!=result["before"]["boot_id"]:raise RuntimeError("boot changed")
        except BaseException as e:result["errors"].append("snapshot: "+repr(e));result["completed"]=False
        for name in ["clock.jsonl","holders.jsonl"]:
            p=out/name
            if p.exists():
                data=p.read_bytes();(out/(name+".gz")).write_bytes(gzip.compress(data,mtime=0));p.unlink()
        result["co_tenants"]=CO_TENANTS
        result["finished_utc_ns"]=time.time_ns();save()
    return 0 if result["completed"] else 2

if __name__=="__main__":sys.exit(main())
