"""Reduce raw integer DM cycles; never recover cycles from derived nanoseconds."""
from __future__ import annotations
import argparse, ast, collections, csv, gzip, hashlib, json, re, sys
from fractions import Fraction
from pathlib import Path
import dm_report
from control import ROOT, coverage, digest, write_json

RISCS=("BRISC","NCRISC","TRISC_0","TRISC_1","TRISC_2")
ZONES=("DM-NOC-READ-BARRIER","DM-CB-RESERVE-BACK","DM-CB-WAIT-FRONT","DM-NOC-WRITE-WAIT","DM-SEM-WAIT")
def readtext(path):
    return gzip.open(path,"rt") if str(path).endswith(".gz") else Path(path).open()

def integer(x):
    if not re.fullmatch(r"[0-9]+",x): raise ValueError(f"Not an integer tick: {x!r}")
    return int(x)

def read_raw(path):
    ops={}; zone_counts=collections.Counter()
    with readtext(path) as f:
        header=next(f).strip()
        if not header.startswith("ARCH: blackhole, CHIP_FREQ[MHz]: 1350,"):
            raise ValueError(f"Unexpected profiler timebase: {header}")
        for r in csv.DictReader(f,skipinitialspace=True):
            if r["PCIe slot"]!="0" or r["trace id"] or r["trace id counter"]:
                raise ValueError("Mixed device or replay timebase")
            op=integer(r["run host ID"]); core=(integer(r["core_x"]),integer(r["core_y"]))
            risc=r["RISC processor type"]; zone=r["zone name"]
            t=integer(r["time[cycles since reset]"])
            o=ops.setdefault(op,{"kernels":{},"sums":{},"zones":set()})
            if zone.endswith("-KERNEL") and risc in RISCS:
                key=(core,risc)
                k=o["kernels"].setdefault(key,{})
                kind=r["type"]
                if kind not in ("ZONE_START","ZONE_END") or kind in k:
                    raise ValueError(f"Duplicate/invalid kernel marker: {op} {key}")
                k[kind]=t
            if r["type"]=="ZONE_TOTAL":
                key=(core,risc,zone)
                if key in o["sums"]: raise ValueError(f"Duplicate sum marker: {op} {key}")
                o["sums"][key]=integer(r["data"])
                o["zones"].add((risc,zone))
                zone_counts[(risc,zone)]+=1
    return header,ops,zone_counts

def reduce_op(row, raw):
    n=integer(row["CORE COUNT"])
    kernels=raw["kernels"]
    for risc in RISCS:
        own=[v for (c,r),v in kernels.items() if r==risc]
        if len(own)!=n: raise ValueError(f"Core count mismatch for {risc}: {len(own)} != {n}")
        if any(set(v)!={"ZONE_START","ZONE_END"} or v["ZONE_END"]<=v["ZONE_START"] for v in own):
            raise ValueError("Missing/reversed kernel span")
    start=min(v["ZONE_START"] for v in kernels.values())
    end=max(v["ZONE_END"] for v in kernels.values())
    span=end-start
    sums={}
    for (c,r,z),v in raw["sums"].items():
        sums[(r,z)]=sums.get((r,z),0)+v
    threads={}
    for risc in RISCS:
        own=[v for (c,r),v in kernels.items() if r==risc]
        resident=sum(v["ZONE_END"]-v["ZONE_START"] for v in own)
        expected=list(ZONES) if risc in ("BRISC","NCRISC") else (
            ["CB-COMPUTE-WAIT-FRONT"] if risc=="TRISC_0" else
            ["CB-COMPUTE-RESERVE-BACK"] if risc=="TRISC_2" else [])
        vals={z:sums.get((risc,z),0) for z in expected}
        blocked=sum(vals.values())
        if blocked>resident: raise ValueError(f"Zones exceed own RISC residency: {risc}")
        threads[risc]={"resident_core_cycles_sum":resident,"zone_core_cycles_sum":vals,
                      "unclassified_core_cycles_sum":resident-blocked,
                      "zone_fraction_of_op_span":{z:float(Fraction(v,n*span)) for z,v in vals.items()},
                      "zone_fraction_of_own_residency":{z:float(Fraction(v,resident)) for z,v in vals.items()}}
    legacy=dm_report.per_op(row)
    # The legacy helper is provenance only; verify it agrees with raw-cycle normalization.
    for name,risc,zone in (("nc_noc_read","NCRISC","DM-NOC-READ-BARRIER"),
                          ("nc_reserve_back","NCRISC","DM-CB-RESERVE-BACK"),
                          ("trisc0_wait_front","TRISC_0","CB-COMPUTE-WAIT-FRONT"),
                          ("trisc2_reserve_back","TRISC_2","CB-COMPUTE-RESERVE-BACK")):
        # process_ops_logs.py formats source-cycle-derived ns with .0f.
        # Verify that forward conversion, without recovering ticks from ns.
        raw_sum=threads[risc]["zone_core_cycles_sum"][zone]
        if int(row[dm_report.COLS[name]]) != round(Fraction(raw_sum*1000,1350)):
            raise ValueError(f"Raw/CSV sum mismatch: {name}")
    if int(row[dm_report.DUR]) != round(Fraction(span*1000,1350)):
        raise ValueError("Raw/CSV span mismatch")
    return {"call_id":integer(row["GLOBAL CALL COUNT"]),"op":row["OP CODE"],"cores":n,
       "start_cycle":start,"end_cycle":end,"span_cycles":span,"threads":threads,
       "identity":{k:row[k] for k in ("ATTRIBUTES","MATH FIDELITY","COMPUTE KERNEL SOURCE",
         "COMPUTE KERNEL HASH","DATA MOVEMENT KERNEL SOURCE","DATA MOVEMENT KERNEL HASH","PROGRAM HASH","PROGRAM CACHE HIT")}}

def aggregate(ops,risc,zone):
    return float(sum((Fraction(o["threads"][risc]["zone_core_cycles_sum"][zone],o["cores"]) for o in ops),Fraction())/
                 sum(o["span_cycles"] for o in ops))

def fenced_intervals(rows):
    marks=[i for i,r in enumerate(rows) if dm_report.is_fence(r)]
    if len(marks)!=42: raise ValueError(f"Expected 42 fence calls, got {len(marks)}")
    groups=[marks[i:i+3] for i in range(0,len(marks),3)]
    if any(g!=list(range(g[0],g[0]+3)) for g in groups): raise ValueError("Ambiguous fence group")
    return [(a[-1],b[0]) for a,b in zip(groups[::2],groups[1::2])]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir",type=Path,default=Path(__file__).parent)
    a=ap.parse_args(); p=a.dir
    capture=json.loads((p/"out/capture.json").read_text())
    criterion=json.loads((p/"criterion.json").read_text())
    result={"verdict":"STOP","scope":criterion["scope"],"intervals":[]}
    try:
        if capture["verdict"]!="CAPTURED_ANALYSIS_REQUIRED" or not capture["counter"]["pass"]:
            raise ValueError("Capture/count control failed")
        if capture["criterion"]["sha256"]!=digest(p/"criterion.json")["sha256"]:
            raise ValueError("Criterion changed after capture")
        if capture["before"]["boot_id"]!=capture["after"]["boot_id"]: raise ValueError("Host reset")
        if not capture["clock_released"] or capture["after"]["own_nodes"] or capture["sampler_exit"]!=0:
            raise ValueError("Incomplete cleanup")
        for snap in [capture["before"],capture["opened"],*capture["holder_snapshots"],capture["after"]]:
            if snap["module_srcversion"]!="A10759A24565BC5BBE903C5" or snap["containment"]!="active" or snap["boot_id"]!=capture["before"]["boot_id"]:
                raise ValueError("Containment, driver or boot changed")
        samples=[json.loads(l) for l in (p/"out/clock.jsonl").read_text().splitlines()]
        for interval in capture["intervals"]:
            if not coverage(samples,interval)["pass"]: raise ValueError("Clock coverage")
        holders=[json.loads(l) for l in (p/"out/holders.jsonl").read_text().splitlines()]
        for r in holders:
            if r.get("error") or r["owner_nodes"]!=["/dev/tenstorrent/0"]: raise ValueError("Device observation failure")
            if any("/dev/tenstorrent/0" in h["nodes"] and h["pid"]!=capture["pid"] for h in r["holders"]):
                raise ValueError("Foreign assigned-node holder")
        result["other_chip_holder_pids"]=sorted({h["pid"] for r in holders for h in r["holders"] if h["pid"]!=capture["pid"]})
        result["holder_observations"]=len(holders)
        rawpath=p/"raw/profile_log_device.csv.gz"
        if not rawpath.exists(): rawpath=p/"tracy/.logs/profile_log_device.csv"
        csvpath=p/"raw/ops.csv"
        if not csvpath.exists(): csvpath=next((p/"tracy/reports").glob("*/ops_perf_results_*.csv"))
        with csvpath.open() as f: rows=list(csv.DictReader(f))
        if any(r["DEVICE ID"]!="0" for r in rows): raise ValueError("Mixed device")
        ids=[integer(r["GLOBAL CALL COUNT"]) for r in rows]
        if len(ids)!=len(set(ids)): raise ValueError("Duplicate call ID")
        header,raw,zones=read_raw(rawpath)
        if set(ids)!=set(raw): raise ValueError("Raw/CSV call identity mismatch")
        result["profiler_header"]=header
        result["runtime_zone_counts"]={f"{r}/{z}":n for (r,z),n in sorted(zones.items())}
        if not set(ZONES).issubset({z for r,z in zones}): raise ValueError("Missing runtime DM zone")
        graph=json.loads((p/"out/add_graph.json").read_text())
        adds=[n for n in graph if n["node_type"]=="function_start" and (n.get("params") or {}).get("name")=="BinaryNgDeviceOperation"]
        byid={n["counter"]:n for n in graph}
        if len(adds)!=40: raise ValueError("Add graph count mismatch")
        for op in adds:
            if len(op["input_tensors"])!=2 or "BinaryOpType::ADD" not in str(op["arguments"]):
                raise ValueError("Add graph identity mismatch")
            if any(byid[i]["params"]["shape"]!="Shape([8192, 8192])" for i in op["input_tensors"]):
                raise ValueError("Add graph shape mismatch")
        sys.path[:0]=[str(ROOT/"perf/roof_budget"),str(ROOT/"perf/b2x_difflayer")]
        import exec_flops
        from real_traffic import counts
        flops=exec_flops.totals(graph); traffic=counts({"nodes":graph})
        result["add_graph"]={"calls":40,"expected_flops":40*8192**2,"observed_flops":flops["eltwise_logical"],
          "expected_bytes":40*402653184,"observed_bytes":round(traffic["real_MB"]*1e6),"traffic_model_version":traffic["traffic_model_version"]}
        if result["add_graph"]["expected_flops"]!=result["add_graph"]["observed_flops"] or result["add_graph"]["expected_bytes"]!=result["add_graph"]["observed_bytes"]:
            raise ValueError("Add counter mismatch")
        result["cached_metadata_mismatches"]=[]
        reduced=[reduce_op(r,raw[i]) for r,i in zip(rows,ids)]
        if any(b["start_cycle"]<a["end_cycle"] for a,b in zip(reduced,reduced[1:])):
            raise ValueError("Mixed/reset/overlapping serial program timebase")
        pairs=fenced_intervals(rows)
        if len(capture["intervals"])!=len(pairs): raise ValueError("Host/device fence mismatch")
        for interval,(left,right) in zip(capture["intervals"],pairs):
            ops=reduced[left+1:right]
            if len(ops)!=interval["reps"]: raise ValueError("Fenced invocation count mismatch")
            expected="BinaryNgDeviceOperation" if interval["case"]=="stream_add" else "MatmulDeviceOperation"
            if any(o["op"]!=expected for o in ops): raise ValueError("Fenced operation identity mismatch")
            # Check shape identity from the operation metadata independently of order.
            n="8192" if interval["case"]!="dense_matmul" else "2048"
            for r in rows[left+1:right]:
                dims=[r[k] for k in ("INPUT_0_X_PAD[LOGICAL]","INPUT_0_Y_PAD[LOGICAL]")]
                if dims!=[f"{n}[{n}]"]*2:
                    # The imported cache keys metadata by program hash, which the add
                    # shares with the 32-square initialization. The fresh graph above
                    # plus this fenced schedule supplies the actual input identity.
                    if interval["case"]!="stream_add" or dims!=["32[32]"]*2 or r["PROGRAM HASH"]!=rows[0]["PROGRAM HASH"]:
                        raise ValueError("Control shape mismatch")
                    result["cached_metadata_mismatches"].append({"call":r["GLOBAL CALL COUNT"],"csv_xy":dims,
                       "graph_xy":[8192,8192],"program_hash":r["PROGRAM HASH"]})
            for op in ops:
                required=[("NCRISC","DM-NOC-READ-BARRIER"),("NCRISC","DM-CB-RESERVE-BACK"),
                          ("TRISC_0","CB-COMPUTE-WAIT-FRONT"),("TRISC_2","CB-COMPUTE-RESERVE-BACK")]
                if any(k not in raw[op["call_id"]]["zones"] for k in required): raise ValueError("Expected per-op runtime zones absent")
            result["intervals"].append({"case":interval["case"],"clock":interval["clock"],
                "host_start_monotonic_ns":interval["start_monotonic_ns"],
                "host_end_monotonic_ns":interval["end_monotonic_ns"],
                "start_fence_end_cycle":reduced[left]["end_cycle"],
                "end_fence_start_cycle":reduced[right]["start_cycle"],
                "ncrisc_read_fraction":aggregate(ops,"NCRISC","DM-NOC-READ-BARRIER"),
                "ncrisc_reserve_fraction":aggregate(ops,"NCRISC","DM-CB-RESERVE-BACK"),
                "ops":ops})
        result["rounds"]=[]
        for a,b in zip(result["intervals"][1::2],result["intervals"][2::2]):
            read=a["ncrisc_read_fraction"]-b["ncrisc_read_fraction"]
            reserve=b["ncrisc_reserve_fraction"]-a["ncrisc_reserve_fraction"]
            result["rounds"].append({"add_minus_matmul_read":read,"matmul_minus_add_reserve":reserve,
                "pass":read>=criterion["separation"]["add_minus_matmul_ncrisc_noc_read_min"] and
                reserve>=criterion["separation"]["matmul_minus_add_ncrisc_reserve_back_min"]})
        if not all(r["pass"] for r in result["rounds"]): raise ValueError("Prespecified separation failed")
        result["verdict"]="GO"
        result["limits"]="GO only to fresh quiet-host measurement. No model timing/roof/fold census. Sum profiling disables C++ report; raw CSV integer markers used directly. Unclassified activity is not issuing."
    except (ValueError,KeyError,AssertionError) as e:
        result["error"]=repr(e)
    write_json(p/"analysis.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("intervals","runtime_zone_counts","cached_metadata_mismatches")},indent=2))
    return 0 if result["verdict"]=="GO" else 2
if __name__=="__main__": raise SystemExit(main())
