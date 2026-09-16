"""One device context: exact dense graph followed by alternating DM controls."""
import argparse, hashlib, importlib, json, os, socket, subprocess, sys, time, types
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"c10_dm_control"))
from control import (ROOT, METAL, CLOCK_REF, write_json, digest, snapshot,
                     validate_snapshot, own_nodes, coverage)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    criterion=json.loads(Path(__file__).with_name("census_criterion.json").read_text())
    result={"scope":criterion["scope"],"requested_node":0,"requested_clock_MHz":1350,
            "host":socket.gethostname(),"pid":os.getpid(),"intervals":[],
            "criterion":digest(Path(__file__).with_name("census_criterion.json"))}
    sampler=None; clock=None; T=None
    def save(): write_json(a.out/"capture.json",result)
    try:
        result["before"]=snapshot(); save(); validate_snapshot(result["before"])
        for name,want in (("TT_VISIBLE_DEVICES","0"),("TT_BIO_LEASE_CARDS","0"),
                          ("TT_BIO_LEASE_HOLDER","worker:c10-burst-census"),("TT_BIO_AICLK","1350")):
            if os.environ.get(name)!=want: raise RuntimeError(f"Wrong {name}")
        import torch
        from tt_bio.main import ensure_p300_mesh_descriptor
        ensure_p300_mesh_descriptor()
        import ttnn, tracy
        import tt_bio.tenstorrent as T
        torch.set_num_threads(2); torch.set_grad_enabled(False)
        modules={m.__name__:digest(m.__file__) for m in (ttnn,tracy)}
        shared=sorted({l.split()[-1] for l in Path("/proc/self/maps").read_text().splitlines()
                       if "/" in l and ("tt-metal" in l or "_ttnn" in l or "libtt_" in l or "tracy" in l.lower())})
        result["runtime"]={"modules":modules,"shared_libraries":[digest(p) for p in shared],
            "python":str(Path(sys.executable).resolve()),
            "flags":{k:v for k,v in os.environ.items() if k.startswith(("TT_","TTNN_","PYTHONPATH","LD_LIBRARY_PATH","OMP_"))}}
        for row in list(modules.values())+result["runtime"]["shared_libraries"]:
            if not row["path"].startswith(str(METAL)+"/"):
                raise RuntimeError(f"Wrong loaded path: {row['path']}")
        if not shared: raise RuntimeError("No loaded shared-library evidence")
        result["source_head"]=subprocess.check_output(["git","rev-parse","HEAD"],cwd=METAL,text=True).strip()
        if not result["source_head"].startswith("1452925b"): raise RuntimeError("Wrong metal source")
        paths=["tt_metal/hw/inc/api/dataflow/dataflow_api.h","tt_metal/hostdevcommon/api/hostdevcommon/profiler_common.h",
          "tt_metal/tools/profiler/kernel_profiler.hpp","tools/tracy/device_post_proc_config.py",
          "tools/tracy/process_device_log.py","tools/tracy/process_ops_logs.py","build_Release/CMakeCache.txt"]
        result["source_files"]=[digest(METAL/p) for p in paths]
        result["profiler_tools"]=[digest(METAL/"build/tools/profiler/bin"/p) for p in ("capture-release","csvexport-release")]
        if "ENABLE_TRACY:BOOL=ON" not in (METAL/"build_Release/CMakeCache.txt").read_text():
            raise RuntimeError("Tracy build flag absent")
        result["bio_head"]=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
        for label,where in (("metal",METAL),("bio",ROOT)):
            diff=subprocess.check_output(["git","diff","HEAD","--binary"],cwd=where)
            (a.out/(label+"_dirty.patch")).write_bytes(diff)
            result[label+"_dirty_diff_sha256"]=hashlib.sha256(diff).hexdigest()
        source=subprocess.check_output(["git","show",CLOCK_REF],cwd=ROOT)
        result["clock_source"]={"ref":CLOCK_REF,"sha256":hashlib.sha256(source).hexdigest()}
        clock=types.ModuleType("c10_clock"); exec(compile(source,CLOCK_REF,"exec"),clock.__dict__)
        dev=T.get_device()
        result["opened"]=snapshot(); save(); validate_snapshot(result["opened"],opened=True)
        # Check the actual node BEFORE FORCE_AICLK can target it.
        if clock._open_nodes()!=[0]: raise RuntimeError("Clock physical node mismatch")
        clock.engage("blackhole")
        result["clock_hold"]=clock.status()
        if result["clock_hold"].get("nodes")!=[0]: raise RuntimeError("Clock target mismatch")
        sampler=subprocess.Popen([sys.executable,str(ROOT/"perf/c10_dm_control/control.py"),
            str(a.out/"clock.jsonl"),str(os.getpid())],stdin=subprocess.PIPE,text=True)
        from census_work import capture
        capture(ttnn,torch,T,dev,a.out,result,save)
        result["verdict"]="CAPTURED_ANALYSIS_REQUIRED"
    except BaseException as e:
        result["error"]=repr(e); result["verdict"]="STOP"; raise
    finally:
        try:
            if sampler is not None and sampler.poll() is None: sampler.communicate("stop\n",timeout=15)
            if sampler is not None: result["sampler_exit"]=sampler.returncode
            result["clock_hold_final"]=clock.status() if clock else {}
        finally:
            try:
                if clock: clock.release()
                result["clock_released"]=not clock or clock.status()=={}
            finally:
                if T: T.cleanup()
                result["after"]=snapshot()
                if result.get("before",{}).get("boot_id")!=result["after"]["boot_id"]:
                    result["verdict"]="STOP"; result["error"]="Host reset"
                save()
    print(json.dumps({"verdict":result["verdict"],"intervals":len(result["intervals"])}),flush=True)

if __name__=="__main__": main()
