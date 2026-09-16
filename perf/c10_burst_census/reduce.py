"""Replay fenced smoke evidence. A valid capture may still have a STOP verdict."""
from __future__ import annotations
import argparse, csv, gzip, hashlib, json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "perf/c10_dm_control"))
from analyze import read_raw, reduce_op
from control import coverage

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def archive_check(p):
    rows = json.loads((p / "raw_manifest.json").read_text())
    for row in rows:
        path = p / "raw" / Path(row["archive"]["path"]).name
        if sha(path) != row["archive"]["sha256"]: raise ValueError("Archive hash mismatch")
        opener = gzip.open if path.suffix == ".gz" else open
        h = hashlib.sha256()
        with opener(path, "rb") as f:
            for b in iter(lambda: f.read(1024*1024), b""): h.update(b)
        if h.hexdigest() != row["sha256"]: raise ValueError("Raw hash mismatch")
    return len(rows)

def fenced(rows, count):
    marks = [i for i,r in enumerate(rows) if r["OP CODE"] == "UnaryDeviceOperation"]
    if len(marks) != count*6: raise ValueError("Fence count mismatch")
    groups = [marks[i:i+3] for i in range(0,len(marks),3)]
    if any(g != list(range(g[0],g[0]+3)) for g in groups): raise ValueError("Fence ambiguity")
    return [(a[-1],b[0]) for a,b in zip(groups[::2],groups[1::2])]

def shape(s):
    m = re.fullmatch(r"Shape\(\[([0-9, ]+)\]\)",s)
    if not m: raise ValueError("Unknown graph shape")
    return [int(x) for x in m.group(1).split(",")]

def required_gaps(calls):
    return sorted({v for call in calls for v in call["unavailable_fields"]
                   if "/core_ranges/" in v or v.endswith("/runtime_args")})

def reduce(p):
    archive_check(p)
    cap = json.loads((p/"out/capture.json").read_text())
    if cap["criterion"]["sha256"] != sha(p/"criterion.json"): raise ValueError("Criterion changed")
    if cap["verdict"] != "CAPTURED_ANALYSIS_REQUIRED": raise ValueError("Capture incomplete")
    if not cap["clock_released"] or cap["sampler_exit"] != 0 or cap["after"]["own_nodes"]:
        raise ValueError("Cleanup incomplete")
    for snap in [cap["before"],cap["opened"],*cap["holder_snapshots"],cap["after"]]:
        if snap["boot_id"] != cap["before"]["boot_id"] or snap["containment"] != "active" or snap["module_srcversion"] != "A10759A24565BC5BBE903C5":
            raise ValueError("Boot/containment/driver mismatch")
        if any(h["pid"] != cap["pid"] and "/dev/tenstorrent/0" in h["nodes"] for h in snap["holders"]):
            raise ValueError("Assigned-node collision")
    samples = [json.loads(l) for l in (p/"out/clock.jsonl").read_text().splitlines()]
    holders = [json.loads(l) for l in (p/"out/holders.jsonl").read_text().splitlines()]
    for h in holders:
        if h.get("error") or h["owner_nodes"] != ["/dev/tenstorrent/0"]: raise ValueError("Holder observation failed")
        if any(x["pid"] != cap["pid"] and "/dev/tenstorrent/0" in x["nodes"] for x in h["holders"]): raise ValueError("Collision")
    other = sorted({x["pid"] for h in holders for x in h["holders"] if x["pid"] != cap["pid"]})
    with (p/"raw/ops.csv").open() as f: rows = list(csv.DictReader(f))
    header,raw,zones = read_raw(p/"raw/profile_log_device.csv.gz")
    ids = [int(r["GLOBAL CALL COUNT"]) for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(raw): raise ValueError("Raw/profiler identity mismatch")
    if any(r["DEVICE ID"] != "0" for r in rows): raise ValueError("Wrong device")
    ops = [reduce_op(r,raw[i]) for r,i in zip(rows,ids)]
    if any(b["start_cycle"] < a["end_cycle"] for a,b in zip(ops,ops[1:])): raise ValueError("Overlapping/reset serial timebase")
    intervals = cap["intervals"]
    if [i["case"] for i in intervals] != ["matmul_off","matmul_on","reblock_off","reblock_on"]: raise ValueError("Wrong arm schedule")
    table=[]; gaps=[]; joins=[]
    for interval,(left,right) in zip(intervals,fenced(rows,4)):
        case=interval["case"]; is_mm=case.startswith("matmul")
        clock=coverage(samples,interval)
        if not clock["pass"]: raise ValueError("Clock coverage failure")
        selected=ops[left+1:right]
        if len(selected)!=4 or any(o["op"]!="GenericOpDeviceOperation" for o in selected): raise ValueError("Dispatch multiplicity")
        graph=json.loads((p/"out"/(case+"_graph.json")).read_text())
        byid={n["counter"]:n for n in graph}
        graphops=[n for n in graph if n["node_type"]=="function_start" and (n.get("params") or {}).get("name")=="GenericOpDeviceOperation"]
        if len(graphops)!=4: raise ValueError("Graph multiplicity")
        expected=[[64,64]]*3 if is_mm else [[1,64,64,32],[1,32,64,64]]
        calls=[]
        if case.endswith("_on"):
            obs=[json.loads(l) for l in (p/"out"/(case+".jsonl")).read_text().splitlines()]
            if obs[-1]["kind"]!="footer" or not obs[-1]["observation_ok"] or obs[-1]["calls"]!=4: raise ValueError("Observer footer")
            calls=[r for r in obs if r["kind"]=="call"]
            if [c["sequence"] for c in calls]!=list(range(4)): raise ValueError("Observer multiplicity")
            gaps+=required_gaps(calls)
            if len({c["binding_cache_hash"] for c in calls})!=1: raise ValueError("Descriptor cache identity changed")
        for index,(g,o) in enumerate(zip(graphops,selected)):
            tensors=[byid[i]["params"] for i in g["input_tensors"]]
            # Native generic graph includes the destination twice, with the same tensor ID.
            if len(tensors)!=len(expected)+1 or tensors[-1]["tensor_id"]!=tensors[-2]["tensor_id"]: raise ValueError("Unexplained native graph operands")
            if [shape(t["shape"]) for t in tensors[:-1]]!=expected: raise ValueError("Actual graph shape mismatch")
            for t in tensors:
                if any(x["device_id"]!=0 for x in json.loads(t["device_tensors"])): raise ValueError("Graph physical device mismatch")
            if calls:
                call=calls[index]
                roles=["left","right","output"] if is_mm else ["input","output"]
                if [x["role"] for x in call["operands"]]!=roles or [x["shape"] for x in call["operands"]]!=expected or "role_basis" not in call["caller"]: raise ValueError("Observer operand roles")
            joins.append({"case":case,"observer_sequence":index if calls else None,"graph_counter":g["counter"],"device":0,"global_call":o["call_id"],"trace":None,"replay":None,"basis":"single-thread four-call bounded control between triple-exp fences; multiplicity, actual operand order/shapes, source-audited roles and output sequence checked","operands":tensors[:-1]})
        start=ops[left]["end_cycle"]; end=ops[right]["start_cycle"]
        span_sum=sum(o["span_cycles"] for o in selected)
        if any(o["start_cycle"]<start or o["end_cycle"]>end for o in selected): raise ValueError("Fence bounds")
        table.append({"class":case,"scope":"bounded identity control, not model","invocations":4,"observed_clock_MHz":1350,"clock":clock,"cores_each":sorted({o["cores"] for o in selected}),"raw_span_cycles_sum":span_sum,"fence_start_cycle":start,"fence_end_cycle":end,"fenced_envelope_cycles":end-start,"unclassified_gap_cycles":end-start-span_sum,"overlap_cycles":0,"matrix_shape_flops":4*2*64**3 if is_mm else None,"matrix_basis":"ordered 64x64 @ 64x64, 2/FMA" if is_mm else "permutation, not a contraction","conditional_tile_padding":"already tile aligned; issued work unverified","issued_work":None,"modeled_compulsory_bytes":4*(3*64*64*2 if is_mm else 2*64*64*32*2),"physical_reads":"unknown, includes no proof about cache/spills/rereads","roof_control_basis":None,"utilization":None,"cycles_above_roof":None,"graph_tensor_cpu_calls":sum(n["node_type"]=="function_start" and (n.get("params") or {}).get("name")=="Tensor::cpu" for n in graph),"ops":selected})
    if len(cap["controls"])!=16 or any(not c["float64_exact"] or c["max_abs"]!=0 or c["off_on_exact"] is False for c in cap["controls"]): raise ValueError("Float64/exact output control")
    return {"verdict":"STOP" if gaps else "SMOKE_REQUIRES_REVIEW","reason":"Installed CoreRange getters unavailable; observer per-core runtime snapshots fail. No model census authorized past this identity gate.","capture_valid":True,"identity_smoke_pass":False,"raw_header":header,"boot_id":cap["before"]["boot_id"],"other_chip_holder_pids":other,"holder_observations":len(holders),"float64_exact_checks":16,"on_off_exact_checks":8,"binding_smoke":cap["binding_smoke"],"required_identity_gaps":sorted(set(gaps)),"joins":joins,"table":table,"model_invocations_measured":0,"model_coverage_fraction":0,"whole_fold_closure":None,"model_roof":None,"model_floor":None,"model_cycles_above_roof":None,"rank_scope":"raw instrumented spans in these 16 small controls only; not a model class ranking or observer overhead estimate","limits":["No synchronized host/device timebase conversion","No profiler perturbation calibration","No machine-code provenance proof","All program-cache-hit labels false; descriptor hash equality is not a binary cache-hit proof","Missing runtime arguments prevent matmul rebinding validation","Co-tenant node3 recorded; host walls diagnostic only"]}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--dir",type=Path,default=Path(__file__).parent/"runs/smoke1");ap.add_argument("--require-go",action="store_true");a=ap.parse_args()
    result=reduce(a.dir)
    (a.dir/"analysis.json").write_text(json.dumps(result,indent=2)+"\n")
    lines=["# Bounded identity-control cycle table","","These are 16 small smoke invocations at sampled 1350 MHz, with node-3 co-tenancy and graph/profiler instrumentation. They are not a model census or a roof. All outputs equal the independent float64 references exactly.","","| Control | Calls | Raw program spans, cycles | Fenced envelope, cycles | Unclassified gaps, cycles | Matrix shape FLOPs | Modeled compulsory bytes | Roof / utilization |","| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in sorted(result["table"],key=lambda x:x["raw_span_cycles_sum"],reverse=True):
        lines.append(f"| {r['class']} | {r['invocations']} | {r['raw_span_cycles_sum']} | {r['fenced_envelope_cycles']} | {r['unclassified_gap_cycles']} | {r['matrix_shape_flops'] if r['matrix_shape_flops'] is not None else 'not a contraction'} | {r['modeled_compulsory_bytes']} | unmeasured |")
    lines += ["","Physical reads and issued arithmetic remain unknown. Gaps are not CPU work. The four control envelopes are disjoint; no parent/child sum or invocation extrapolation is used. Zero model invocations were captured. Matmul uses HiFi4 with FP32 accumulation; reblock's source descriptor uses HiFi2. Neither is a throughput roof.","","STOP: CoreRange getter mismatch prevents per-core runtime snapshots. Repair and revalidate the observer before any current-default model windows or matched roofs. No whole-fold floor, binding constraint, headroom or campaign ceiling follows from this table."]
    (Path(__file__).parent/"TABLE.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({k:result[k] for k in ("verdict","reason","capture_valid","float64_exact_checks","on_off_exact_checks","model_invocations_measured")},indent=2))
    return 2 if a.require_go and result["verdict"]!="GO" else 0
if __name__=="__main__": raise SystemExit(main())
