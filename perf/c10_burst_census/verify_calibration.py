"""Accept only raw, clock-qualified controls under the committed fixed criterion."""
import json, subprocess, sys
from pathlib import Path
from reduce import archive_check, sha
ROOT=Path(__file__).resolve().parents[2]
p=ROOT/"perf/c10_dm_control"
subprocess.run([sys.executable,str(p/"analyze.py"),"--dir",str(p)],check=True,stdout=subprocess.DEVNULL)
archives=archive_check(p)
c=json.loads((p/"out/capture.json").read_text()); r=json.loads((p/"analysis.json").read_text()); cr=json.loads((p/"criterion.json").read_text())
assert r["verdict"]=="GO" and c["counter"]["pass"] and len(r["intervals"])==7 and len(r["rounds"])==3
assert cr["separation"]=={"add_minus_matmul_ncrisc_noc_read_min":0.4,"matmul_minus_add_ncrisc_reserve_back_min":0.2,"require_each_round":True}
assert all(i["clock"]["pass"] and i["clock"]["min_MHz"]==i["clock"]["max_MHz"]==1350 for i in r["intervals"])
assert all(x["add_minus_matmul_read"]>=.4 and x["matmul_minus_add_reserve"]>=.2 for x in r["rounds"])
sys.path[:0]=[str(ROOT/"perf/roof_budget"),str(ROOT/"perf/b2x_difflayer"),str(ROOT/"perf/roof_arb")]
import exec_flops
from real_traffic import counts
from corrected_traffic import counts as corrected
matmul=json.loads((p/"out/matmul_graph.json").read_text())
flops=exec_flops.totals(matmul)["matmul_padded"]
bytes1=round(counts({"nodes":matmul})["real_MB"]*1e6)
bytes2=round(corrected({"nodes":matmul})["real_MB"]*1e6)
assert flops==1099511627776 and bytes1==bytes2==402653184
result={"verdict":"GO","measured_commit":"ee11d5263ffc96cf25939967d3bdd38c1a0f96e3","criterion_commit":"ec761a98f","criterion_sha256":sha(p/"criterion.json"),"raw_archives_verified":archives,"dense_graph_recount":{"matrix_flops":flops,"modeled_bytes_both_counters":bytes1},"interval_clocks":[i["clock"] for i in r["intervals"]],"rounds":r["rounds"],"runtime_zone_counts":r["runtime_zone_counts"]}
Path(__file__).with_name("dependency_acceptance.json").write_text(json.dumps(result,indent=2)+"\n")
print("Calibration accepted from raw evidence: exact dense counts, seven 1350 MHz intervals, all three fixed separation rounds.")
