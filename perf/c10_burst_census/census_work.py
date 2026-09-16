"""Bounded profiler windows while the production model performs its full default work."""
from collections import Counter
from contextlib import nullcontext
import gzip, hashlib, json, os, shutil, sys, time
from pathlib import Path
from control import ROOT, coverage, digest, snapshot, validate_snapshot, write_json
from perf.c10_generic_identity.observer import Capture

def capture(ttnn, torch, T, dev, out, result, save):
    criterion=json.loads((ROOT/"perf/c10_burst_census/census_criterion.json").read_text())
    if json.loads((ROOT/"perf/c10_burst_census/runs/smoke2/analysis.json").read_text())["verdict"]!="GO": raise RuntimeError("Smoke not GO")
    result["census_criterion"]=digest(ROOT/"perf/c10_burst_census/census_criterion.json")
    result["model_protocol"]=criterion["model"]; result["windows"]=[];result["drains"]=[]
    logs=out.parent/"tracy/.logs"; rawpath=logs/"profile_log_device.csv"
    (out/"windows").mkdir()
    result["instrument"]={"mid_run_dump":os.environ.get("TT_METAL_PROFILER_MID_RUN_DUMP"),"program_support_count":os.environ.get("TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT"),"device_data_push_disabled":os.environ.get("TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY"),"python_partial":True}
    selected_bytes=0
    def drain(label,keep=False):
        nonlocal selected_bytes
        ttnn.synchronize_device(dev); ttnn.ReadDeviceProfiler(dev)
        if not rawpath.exists(): raise RuntimeError("Mid-run raw dump missing")
        data=rawpath.read_bytes()
        if len(data)>criterion["budget"]["max_drain_bytes"]: raise RuntimeError("Raw chunk resource cap")
        row={"label":label,"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest(),"retained":keep}
        if keep:
            dest=out/"windows"/(label+".csv.gz")
            dest.write_bytes(gzip.compress(data,mtime=0)); row["archive"]=digest(dest)
            selected_bytes+=dest.stat().st_size
            if selected_bytes>criterion["budget"]["max_retained_bytes"]: raise RuntimeError("Retained resource cap")
        result["drains"].append(row)
        # dumpDeviceResults waits for its writer and closes the append stream before return.
        # Keep the exact header for the next append. No other process writes this run's log.
        rawpath.write_bytes(b"".join(data.splitlines(keepends=True)[:2]))
    small=ttnn.from_torch(torch.ones(32,32,dtype=torch.bfloat16),layout=ttnn.TILE_LAYOUT,device=dev,memory_config=ttnn.DRAM_MEMORY_CONFIG)
    def fence():
        for _ in range(3):
            z=ttnn.exp(small);ttnn.deallocate(z)
        ttnn.synchronize_device(dev)
    def window(label, fn):
        live=snapshot();validate_snapshot(live,opened=True)
        result.setdefault("holder_snapshots",[]).append(live)
        drain(label+"_before");fence()
        rec={"case":label,"start_monotonic_ns":time.monotonic_ns(),"device_counter_before":ttnn._ttnn.get_device_operation_id()}
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        try:
            with Capture(ttnn,out/"windows"/(label+".jsonl"),config={"scope":label}): value=fn()
            ttnn.synchronize_device(dev)
        finally:
            graph=ttnn.graph.end_graph_capture()
        rec["end_monotonic_ns"]=time.monotonic_ns();rec["device_counter_after"]=ttnn._ttnn.get_device_operation_id()
        if isinstance(graph,str):graph=json.loads(graph)
        if len(graph)>criterion["budget"]["max_window_graph_nodes"]: raise RuntimeError("Graph node resource cap")
        (out/"windows"/(label+"_graph.json.gz")).write_bytes(gzip.compress(json.dumps(graph).encode(),mtime=0))
        rec["graph_nodes"]=len(graph)
        fence();drain(label,True)
        result["windows"].append(rec);save();print("WINDOW",label,rec["graph_nodes"],flush=True)
        return value
    counts=Counter(); originals=[]; active=False
    for name,want in criterion["window_selection"].items():
        cls=getattr(T,name);attr="forward" if issubclass(cls,T.TorchWrapper) else "__call__"
        original=getattr(cls,attr);originals.append((cls,attr,original))
        def bind(name,want,fn):
            def call(obj,*args,**kwargs):
                nonlocal active
                dims=[list(x.shape) for x in args if hasattr(x,"shape")][:2]
                sig=name+"|"+str(dims);counts[sig]+=1
                if active:return fn(obj,*args,**kwargs)
                active=True
                try:
                    if counts[sig]==want:
                        label=name+"_"+str(sum(w["case"].startswith(name+"_") for w in result["windows"]))
                        return window(label,lambda:fn(obj,*args,**kwargs))
                    value=fn(obj,*args,**kwargs);drain("uncaptured_"+name);return value
                finally:active=False
            return call
        setattr(cls,attr,bind(name,want,original))
    sys.path[:0]=[str(ROOT/"scripts/gpu_vs_tt"),str(ROOT/"perf/other512")]
    import tt_baseline as B
    from fold_ab_multi import patch_boltz2_cfg
    B.RECYCLING_STEPS=3;B.SAMPLING_STEPS=200;B.DIFFUSION_SAMPLES=1;B.SEED=0
    patch_boltz2_cfg()
    # Avoid tt_baseline's optional all-board tt-smi telemetry subprocess.
    B._card_info=lambda:{"card_type":"recorded assigned node0", "sysfs_subsystem":Path("/sys/class/tenstorrent/tenstorrent!0/device/subsystem_device").read_text().strip()}

    fix=ROOT/"perf/size512/fixtures"
    result["fixtures"]=[digest(fix/("cdk2x2_512."+ext)) for ext in ("yaml","a3m")]
    try:
        fold,meta,state=B.build_fold("boltz2",out/"msa",fix/"cdk2x2_512.yaml",fix/"cdk2x2_512.a3m",recycling_steps=3)
        if meta["n_msa"]!=35:raise RuntimeError("MSA row count mismatch")
        result["model_meta"]=meta;save();drain("model_loaded")
        result["fold_start_monotonic_ns"]=time.monotonic_ns();save()
        wall,metrics=fold()
        result["fold_end_monotonic_ns"]=time.monotonic_ns()
        result["fold_metrics"]=metrics;result["diagnostic_instrumented_fold_wall_s"]=wall
        result["model_completed"]=True
        result["module_occurrences"]=dict(counts)
        (out/"cifs").mkdir()
        result["cifs"]=[]
        for path in Path(meta["struct_dir"]).glob("*.cif"):
            dest=out/"cifs"/path.name;shutil.copyfile(path,dest);result["cifs"].append(digest(dest))
        drain("fold_tail");save()
    finally:
        for cls,attr,fn in originals:setattr(cls,attr,fn)
        result["module_occurrences"]=dict(counts);save()
    x=ttnn.from_torch(torch.ones(8192,8192,dtype=torch.bfloat16),layout=ttnn.TILE_LAYOUT,device=dev,memory_config=ttnn.DRAM_MEMORY_CONFIG)
    y=ttnn.from_torch(torch.ones(8192,8192,dtype=torch.bfloat16),layout=ttnn.TILE_LAYOUT,device=dev,memory_config=ttnn.DRAM_MEMORY_CONFIG)
    cfg=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,math_approx_mode=False,fp32_dest_acc_en=True,packer_l1_acc=True)
    functions={"stream_copy":lambda:ttnn.clone(x),"stream_add":lambda:ttnn.add(x,y,memory_config=ttnn.DRAM_MEMORY_CONFIG),"dense_hifi4":lambda:ttnn.matmul(x,y,compute_kernel_config=cfg,memory_config=ttnn.DRAM_MEMORY_CONFIG)}
    for key,fn in functions.items():
        z=fn();ttnn.synchronize_device(dev);ttnn.deallocate(z);drain("warm_"+key)
    result["control_accuracy"]=[]
    for round_id in range(3):
        for key,fn in functions.items():
            def run(fn=fn):
                outputs=[]
                for _ in range(4):outputs.append(fn())
                return outputs
            outputs=window(key+"_"+str(round_id),run)
            for z in outputs:
                a=ttnn.to_torch(z)
                expected=8192 if key=="dense_hifi4" else 2 if key=="stream_add" else 1
                ok=bool(torch.all(a==expected))
                result["control_accuracy"].append({"case":key,"round":round_id,"float64_constant_reference":expected,"exact":ok})
                if not ok:raise RuntimeError("Roof control math failure")
                ttnn.deallocate(z)
            save()
    samples=[json.loads(l) for l in (out/"clock.jsonl").read_text().splitlines() if l.endswith("}")]
    for rec in result["windows"]:rec["clock"]=coverage(samples,rec)
    result["fold_clock"]=coverage(samples,{"start_monotonic_ns":result["fold_start_monotonic_ns"],"end_monotonic_ns":result["fold_end_monotonic_ns"]})
    for z in (x,y,small):ttnn.deallocate(z)
    drain("controls_tail");save()
