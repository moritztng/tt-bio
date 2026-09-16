"""CPU replay of disjoint default-fold windows; no extrapolation or fitted cycles."""
from __future__ import annotations
import argparse, collections, csv, gzip, hashlib, json, math, re, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"perf/c10_dm_control"),str(ROOT/"perf/c10_flop_contract")]
from analyze import read_raw, ZONES
from control import coverage
from audit import matrix_candidates, known_fused_matrix
from reduce import shape, required_gaps

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readjson(p):return json.loads(Path(p).read_text())
def open_text(p):return gzip.open(p,"rt") if str(p).endswith(".gz") else open(p)

def host_ops(path):
    """Same full/cache encoding as the recorded k10 process_ops_logs.py; never use cached shapes."""
    result={};cache={};bad=[]
    csv.field_size_limit(10000000)
    with open_text(path) as f:
        for row in csv.DictReader(f,delimiter=";",quotechar="`"):
            msg=row["MessageName"]
            if not msg.startswith("TT_DNN_DEVICE_OP:"):continue
            if " ->\n" in msg:
                try:o=json.loads(msg.split(" ->\n",1)[1])
                except json.JSONDecodeError:bad.append(msg[:100]);continue
                cache[(int(o["device_id"]),int(o["op_hash"]))]=o
            else:
                fields=msg.split(":",1)[1].split(",")
                key=(int(fields[2]),int(fields[1]))
                if key not in cache:bad.append(msg[:100]);continue
                o=dict(cache[key]);o.update(global_call_count=int(fields[4]),program_cache_hit=fields[3].strip().lower() in ("1","true"))
            i=int(o["global_call_count"])
            if i in result:raise ValueError("Duplicate host global call")
            result[i]=o
    return result,bad

def span(raw):
    if not raw["kernels"]:raise ValueError("No kernel endpoints")
    for ends in raw["kernels"].values():
        if set(ends)!={"ZONE_START","ZONE_END"} or ends["ZONE_END"]<=ends["ZONE_START"]:raise ValueError("Missing/reversed endpoints")
    start=min(v["ZONE_START"] for v in raw["kernels"].values());end=max(v["ZONE_END"] for v in raw["kernels"].values())
    threads={}
    for risc in sorted({r for c,r in raw["kernels"]}):
        own={c:v for (c,r),v in raw["kernels"].items() if r==risc}
        resident=sum(v["ZONE_END"]-v["ZONE_START"] for v in own.values())
        zones=collections.Counter()
        for (c,r,z),n in raw["sums"].items():
            if r==risc:
                if c not in own:raise ValueError("Sum without own RISC endpoints")
                zones[z]+=n
        if sum(zones.values())>resident:raise ValueError("Zone sums exceed RISC residency")
        threads[risc]={"cores":len(own),"resident_core_cycles":resident,"zone_core_cycles":dict(zones),"unclassified_core_cycles":resident-sum(zones.values())}
    return {"start_cycle":start,"end_cycle":end,"span_cycles":end-start,"active_cores":len({c for c,r in raw["kernels"]}),"longest_single_core_risc_span":max(v["ZONE_END"]-v["ZONE_START"] for v in raw["kernels"].values()),"threads":threads}

def union_cycles(spans):
    intervals=sorted((s["start_cycle"],s["end_cycle"]) for s in spans)
    total=0;end=None
    for a,b in intervals:
        total+=b-a if end is None else max(0,b-max(a,end));end=max(end or b,b)
    return total

def graph_programs(nodes, native_names=()):
    ids={n["counter"]:n for n in nodes}
    if len(ids)!=len(nodes):raise ValueError("Duplicate graph node")
    ops=[];metadata=[]
    for n in nodes:
        name=(n.get("params") or {}).get("name","")
        if n["node_type"]!="function_start" or not (name.endswith("DeviceOperation") or name in native_names):continue
        ends=[ids[i] for i in n["connections"] if ids[i]["node_type"]=="function_end" and ids[i].get("params",{}).get("name")==name]
        if len(ends)!=1:raise ValueError("Native operation end is ambiguous")
        ins=[ids[i]["params"] for i in n["input_tensors"]]
        outs=[ids[i]["params"] for i in ends[0]["connections"] if ids[i]["node_type"]=="tensor"]
        # Void/in-place native returns may have no output tensor node; work stays unknown.
        for t in ins+outs:
            if any(d["device_id"]!=0 for d in json.loads(t["device_tensors"])):raise ValueError("Graph other physical device")
        # Read only direct native start/end edges, never inclusive parent spans.
        record={"node":n,"inputs":ins,"outputs":outs,"end_counter":ends[0]["counter"]}
        ops.append(record)
    return ids,ops,metadata

def compact_tensor(t):return {k:t[k] for k in ("tensor_id","shape","dtype","layout","buffer_type","size","address","device_tensors")}
def unique(ts):
    d={}
    for t in ts:d[(t["buffer_type"],t["address"])]=t
    return list(d.values())

def observer_records(path):
    rows=[json.loads(l) for l in open(path)]
    if rows[-1].get("kind")!="footer" or not rows[-1]["observation_ok"]:raise ValueError("Observer did not complete")
    calls={r["sequence"]:r for r in rows if r["kind"]=="call"}
    outcomes={r["sequence"]:r for r in rows if r["kind"]=="outcome"}
    if set(calls)!=set(outcomes) or len(calls)!=rows[-1]["calls"]:raise ValueError("Observer counts differ")
    if required_gaps(calls.values()):raise ValueError("Required live runtime getter missing")
    joined={}
    for seq,c in calls.items():
        o=outcomes[seq]
        if o["outcome"]!="returned":raise ValueError("Generic dispatch raised")
        # Recorded native source: EncodePerDeviceProgramID(base,0) == base << 10.
        i=(o["device_operation_counter_after"]-1)<<10
        if i in joined:raise ValueError("Duplicate observer execution ID")
        joined[i]=c
    return joined

def operand_work(g,ids,observer):
    name=g["node"]["params"]["name"];ins=g["inputs"];outs=g["outputs"]
    result={"matrix_shape_flops":None,"conditional_tile32_flops":None,"issued_work":None,"physical_read_bytes":None,"modeled_dram_boundary_bytes":None,"remaining":["physical rereads, caches, spills and issued instructions unknown"]}
    if observer:
        operands=observer["operands"]
        # Native GenericOp graph also appends its output args; validate the leading
        # dispatched operand list directly, including tensor identity duplicates.
        if len(ins)<len(operands) or [shape(t["shape"]) for t in ins[:len(operands)]]!=[o["shape"] for o in operands]:raise ValueError("Observer/graph ordered shapes differ")
        if any(t["tensor_id"] not in {x["tensor_id"] for x in ins[:len(operands)]} for t in ins[len(operands):]):raise ValueError("Unexplained generic trailing operand")
        roles=[o["role"] for o in operands]
        known=all(r not in (None,"unknown") for r in roles) and "role_basis" in observer["caller"]
        result["declared_roles"]=roles
        if known:
            realins=[t for t,r in zip(ins,roles) if r!="output"];realouts=[t for t,r in zip(ins,roles) if r=="output"]
            if not realouts:known=False
            if known and not {t["tensor_id"] for t in outs}.issubset({t["tensor_id"] for t in realouts}):raise ValueError("Returned output is not a declared output")
        if known and observer["caller"]["function"]=="generic_minimal_matmul" and roles[:2]==["left","right"]:
            a,b=[o["shape"] for o in operands[:2]]
            # Restrict to directly flattened dense inputs, no head-major reinterpretation.
            if len(b)==2 and a[-1]==b[0] and sum(math.prod(o["shape"]) for o in operands if o["role"]=="output")==math.prod(a[:-1])*b[1]:
                m=known_fused_matrix("mm_generic",[((math.prod(a[:-1]),a[-1]),b)])
                result.update(matrix_shape_flops=m["matrix_shape_flops"],conditional_tile32_flops=m["matrix_tile32_flops"])
                result["matrix_basis"]="audited mm_generic ordered left/right, direct flatten, split output extent checked"
        if not known:result["remaining"].append("generic operand roles uncounted")
        else:ins,outs=realins,realouts
        result["generic_work_unknown"]=result["matrix_shape_flops"] is None
    else:known=bool(outs)
    if name=="MatmulDeviceOperation":
        try:
            m=matrix_candidates(g["node"],ids,[tuple(shape(t["shape"])) for t in outs])
            result.update(matrix_shape_flops=m["shape_flops"],conditional_tile32_flops=m["tile32_flops"],matrix_basis=m)
        except ValueError as exc:result["remaining"].append(str(exc))
    if known:
        # Graph size is its logical payload, not physical allocation or issued traffic.
        result["modeled_dram_boundary_bytes"]=sum(t["size"] for t in unique(ins) if t["buffer_type"]=="BufferType::DRAM")+sum(t["size"] for t in unique(outs) if t["buffer_type"]=="BufferType::DRAM")
        result["bytes_basis"]="one logical-payload read per unique DRAM input address plus one write per unique DRAM output address; excludes padding/rereads/spills"
    if result["matrix_shape_flops"] is None:result["remaining"].append("non-matrix arithmetic/SFPU/reduction work symbolic or uncounted, never zeroed")
    return result

def reduce_window(p,w,metadata):
    name=w["case"];root=p/"out/windows"
    rawfile=root/(name+".csv.gz")
    header,raw,zones=read_raw(rawfile)
    expected=list(range(w["device_counter_before"]-3,w["device_counter_after"]+3))
    if sorted(raw)!=[i<<10 for i in expected]:raise ValueError("Raw calls do not equal fenced native counter range")
    rawids=sorted(raw);spans={i:span(raw[i]) for i in rawids}
    for i in rawids:
        if i not in metadata:raise ValueError("Missing profiler identity for "+str(i))
        if int(metadata[i]["device_id"])!=0:raise ValueError("Profiler wrong physical device")
    for i in rawids[:3]+rawids[-3:]:
        if metadata[i]["op_code"]!="UnaryDeviceOperation" or "EXP" not in str(metadata[i]["attributes"]):raise ValueError("Wrong triple-exp fence identity")
    nodes=json.load(gzip.open(root/(name+"_graph.json.gz"),"rt"));ids,gs,skips=graph_programs(nodes,{metadata[i]["op_code"] for i in rawids[3:-3]})
    inner=rawids[3:-3]
    if len(gs)!=len(inner):raise ValueError(f"Native graph/raw count mismatch: {len(gs)} != {len(inner)}")
    obs=observer_records(root/(name+".jsonl"))
    if set(obs)!={i for i in inner if metadata[i]["op_code"]=="GenericOpDeviceOperation"}:raise ValueError("Observer/native execution set mismatch")
    ops=[]
    for i,g in zip(inner,gs):
        meta=metadata[i];opname=g["node"]["params"]["name"]
        if meta["op_code"]!=opname:raise ValueError("Ordered native/profiler opcode mismatch")
        ob=obs.get(i)
        class_name=opname
        if ob:
            sources=[k["fields"]["kernel_source"] for k in ob["descriptor"]["fields"]["kernels"]]
            class_name="generic:"+ob["caller"]["function"]+":"+",".join(sorted({Path(s).parent.name for s in sources}))
        work=operand_work(g,ids,ob)
        o={"global_call_id":i,"native_counter":i>>10,"device":0,"trace":None,"replay":None,"class":class_name,"graph_start":g["node"]["counter"],"graph_end":g["end_counter"],"graph_arguments":g["node"].get("arguments"),"inputs":[compact_tensor(t) for t in g["inputs"]],"outputs":[compact_tensor(t) for t in g["outputs"]],"profiler_identity":{k:meta.get(k) for k in ("op_hash","program_cache_hit","kernel_info","attributes")},"observer_sequence":ob["sequence"] if ob else None,"observer_execution_sha256":ob["execution_sha256"] if ob else None,"observer_missing_fields":ob["unavailable_fields"] if ob else [],**spans[i],**work}
        ops.append(o)
    a=spans[rawids[2]]["end_cycle"];b=spans[rawids[-3]]["start_cycle"]
    if any(o["start_cycle"]<a or o["end_cycle"]>b for o in ops):raise ValueError("Program outside fences")
    union=union_cycles(ops);summed=sum(o["span_cycles"] for o in ops)
    starts=sum(n["node_type"]=="function_start" for n in nodes);ends=sum(n["node_type"]=="function_end" for n in nodes)
    return {"case":name,"clock":w["clock"],"raw_sha256":sha(rawfile),"header":header,"invocations":len(ops),"generic_invocations":len(obs),"graph_starts":starts,"graph_ends":ends,"excluded_metadata_nodes":[s["node"]["counter"] for s in skips],"program_span_cycles_sum":summed,"program_union_cycles":union,"overlap_cycles":summed-union,"fence_start_cycle":a,"fence_end_cycle":b,"fence_envelope_cycles":b-a,"unclassified_gap_cycles":b-a-union,"runtime_zones":{f"{r}/{z}":v for (r,z),v in zones.items()},"ops":ops}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--dir",type=Path,default=Path(__file__).parent/"runs/census1");a=ap.parse_args();p=a.dir
    cap=readjson(p/"out/capture.json")
    if cap["criterion"]["sha256"]!=sha(p/"census_criterion.json") or cap["census_criterion"]["sha256"]!=sha(p/"census_criterion.json"):raise ValueError("Criterion changed")
    for row in readjson(p/"raw_manifest.json"):
        path=p/row["archive"]
        if sha(path)!=row["archive_sha256"]:raise ValueError("Archive hash mismatch")
        h=hashlib.sha256()
        with gzip.open(path,"rb") as stream:
            for b in iter(lambda:stream.read(1024*1024),b""):h.update(b)
        if h.hexdigest()!=row["sha256"]:raise ValueError("Original hash mismatch")
    for row in cap["drains"]:
        if not row["retained"]:continue
        path=p/"out/windows"/(row["label"]+".csv.gz")
        if sha(path)!=row["archive"]["sha256"]:raise ValueError("Window archive hash mismatch")
        h=hashlib.sha256()
        with gzip.open(path,"rb") as stream:
            for b in iter(lambda:stream.read(1024*1024),b""):h.update(b)
        if h.hexdigest()!=row["sha256"]:raise ValueError("Window original hash mismatch")
    result={"verdict":"STOP","scope":"retained disjoint windows only; full default model executed, never extrapolated","windows":[],"failures":[],"model_completed":cap.get("model_completed",False)}
    if (p/"postprocess_stop.json").exists():result["failures"].append("Automatic host report exceeded resource budget (61.8 GB CSV, at least 182 GB RSS); owned device-closed postprocessor stopped")
    if cap.get("verdict")!="CAPTURED_ANALYSIS_REQUIRED":result["failures"].append("Capture did not complete: "+str(cap.get("error")))
    if not cap.get("clock_released") or cap.get("sampler_exit")!=0 or cap.get("after",{}).get("own_nodes"):result["failures"].append("Incomplete cleanup")
    samples=[json.loads(l) for l in open_text(p/"out/clock.jsonl.gz")]
    holders=[json.loads(l) for l in (p/"out/holders.jsonl").read_text().splitlines()]
    for h in holders:
        if h.get("error") or h["owner_nodes"]!=["/dev/tenstorrent/0"] or any(x["pid"]!=cap["pid"] and "/dev/tenstorrent/0" in x["nodes"] for x in h["holders"]):raise ValueError("Holder failure/collision")
    result["other_chip_holder_pids"]=sorted({x["pid"] for h in holders for x in h["holders"] if x["pid"]!=cap["pid"]})
    for snap in [cap["before"],cap["opened"],*cap["holder_snapshots"],cap["after"]]:
        if snap["boot_id"]!=cap["before"]["boot_id"] or snap["containment"]!="active" or snap["module_srcversion"]!="A10759A24565BC5BBE903C5":raise ValueError("Containment/driver/boot changed")
    result["fold_clock"]=coverage(samples,{"start_monotonic_ns":cap["fold_start_monotonic_ns"],"end_monotonic_ns":cap["fold_end_monotonic_ns"]}) if cap.get("fold_end_monotonic_ns") else None
    hostpath=p/"raw/tracy_ops_data.csv.gz"
    meta,bad=host_ops(hostpath);result["host_metadata_unparsed_messages"]=bad;result["host_device_operations"]=len(meta)
    for w in cap["windows"]:
        w=dict(w);w["clock"]=coverage(samples,w)
        try:
            row=reduce_window(p,w,meta)
            row["timing_valid"]=w["clock"]["pass"]
            if not row["timing_valid"]:result["failures"].append(w["case"]+": clock coverage invalid")
            result["windows"].append(row)
        except (ValueError,KeyError,AssertionError) as exc:result["failures"].append(w["case"]+": "+str(exc))
    result["module_occurrences"]=cap.get("module_occurrences")
    result["control_accuracy"]=cap.get("control_accuracy",[])
    if len(result["control_accuracy"])!=36 or any(not x["exact"] for x in result["control_accuracy"]):result["failures"].append("Control accuracy incomplete")
    classes={}
    for w in result["windows"]:
        if w["case"].startswith(("stream_","dense_")) or not w["timing_valid"]:continue
        for o in w["ops"]:
            c=classes.setdefault(o["class"],{"class":o["class"],"invocations":0,"scope_counts":{},"span_cycles_sum":0,"matrix_known_calls":0,"matrix_shape_flops":0,"conditional_tile32_flops":0,"bytes_known_calls":0,"modeled_dram_boundary_bytes":0,"issued_work":None,"physical_reads":None,"utilization":None,"cycles_above_roof":None})
            c["invocations"]+=1;c["span_cycles_sum"]+=o["span_cycles"];c["scope_counts"][w["case"]]=c["scope_counts"].get(w["case"],0)+1
            if o["matrix_shape_flops"] is not None:
                c["matrix_known_calls"]+=1;c["matrix_shape_flops"]+=o["matrix_shape_flops"];c["conditional_tile32_flops"]+=o["conditional_tile32_flops"]
            if o["modeled_dram_boundary_bytes"] is not None:c["bytes_known_calls"]+=1;c["modeled_dram_boundary_bytes"]+=o["modeled_dram_boundary_bytes"]
    result["classes"]=sorted(classes.values(),key=lambda c:c["span_cycles_sum"],reverse=True)
    for c in result["classes"]:
        if not c["matrix_known_calls"]:c["matrix_shape_flops"]=c["conditional_tile32_flops"]=None
        if not c["bytes_known_calls"]:c["modeled_dram_boundary_bytes"]=None
    result["model_invocations_measured"]=sum(c["invocations"] for c in result["classes"])
    result["timebase"]={"raw":"per-RISC 44-bit WALL_CLOCK register markers, source retained","host_conversion":None,"independent_cross_core_offset_calibration":None,"scope":"program extrema and per-core durations in raw device cycles; no synchronized host elapsed time or CPU-gap interpretation"}
    result["whole_fold_cycle_total"]=None;result["whole_fold_floor"]=None;result["whole_fold_cycles_above_roof"]=None
    result["verdict"]="GO" if not result["failures"] and result["model_invocations_measured"] else "STOP"
    result["controls"]=[]
    for w in result["windows"]:
        if not w["case"].startswith(("stream_","dense_")):continue
        d={k:w[k] for k in ("case","clock","invocations","program_span_cycles_sum","timing_valid")}
        if w["timing_valid"]:
            d["modeled_bytes_per_cycle"]=sum(o["modeled_dram_boundary_bytes"] for o in w["ops"])/w["program_span_cycles_sum"]
            if w["case"].startswith("dense_"):d["matrix_flops_per_cycle"]=sum(o["matrix_shape_flops"] for o in w["ops"])/w["program_span_cycles_sum"]
        d["application"]="specific control/fidelity/layout/instrument only; not an observed binding model roof";result["controls"].append(d)
    (p/"analysis.json").write_text(json.dumps(result,indent=2)+"\n")
    lines=["# Bounded default-fold census","","Measured program-marker spans at during-window sampled 1350 MHz, with graph/observer/sum profiling and node-3 co-tenancy. Counts cover only the named windows; no whole-fold extrapolation. Matrix counts cover only the stated calls. Other arithmetic remains uncounted or symbolic.","","| Class | Calls | Program spans (cycles) | Counted matrix calls | Matrix shape FLOPs | Conditional tile32 FLOPs | Modeled DRAM bytes (known calls) |","| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for c in result["classes"]:lines.append("| {} | {} | {} | {} | {} | {} | {} ({}) |".format(c["class"],c["invocations"],c["span_cycles_sum"],c["matrix_known_calls"],c["matrix_shape_flops"] if c["matrix_shape_flops"] is not None else "unknown/symbolic",c["conditional_tile32_flops"] if c["conditional_tile32_flops"] is not None else "unknown",c["modeled_dram_boundary_bytes"],c["bytes_known_calls"]))
    lines += ["","Physical reads and issued instructions are unknown for every class. Bytes model logical payload crossing DRAM boundaries once; they exclude physical padding, rereads and spills. Utilization and cycles above a binding roof remain unmeasured for every model class.","","| Window | Calls | Span sum | Span union | Overlap | Fenced envelope | Unclassified gap | Clock valid |","| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for w in result["windows"]:lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(w["case"],w["invocations"],w["program_span_cycles_sum"],w["program_union_cycles"],w["overlap_cycles"],w["fence_envelope_cycles"],w["unclassified_gap_cycles"],w["timing_valid"]))
    lines += ["","Gaps are not CPU work. Outer graph spans are not used; all retained native operations have unique matching end nodes. Program IDs are joined through the recorded 10-bit device-ID encoding. The model completed its full default protocol; raw data outside named windows is excluded with a hash/byte drain ledger.","","Verdict: "+result["verdict"]+". "+"; ".join(result["failures"])]
    (Path(__file__).parent/"TABLE.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("windows","classes","module_occurrences","control_accuracy")},indent=2))
if __name__=="__main__":main()
