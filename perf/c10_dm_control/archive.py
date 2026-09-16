"""Keep one compressed copy of raw evidence and hash its source bytes."""
import argparse, ast, csv, gzip, hashlib, json, shutil
from pathlib import Path
from control import METAL, digest, write_json

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument("--dir",type=Path,default=Path(__file__).parent)
p=ap.parse_args().dir
dest=p/"raw"; dest.mkdir(exist_ok=True)
log=p/"tracy/.logs"
report=next((p/"tracy/reports").glob("*/ops_perf_results_*.csv"))
sources=[(log/name,name+".gz") for name in ("profile_log_device.csv","tracy_ops_times.csv","tracy_ops_data.csv",
            "new_zone_src_locations.log","zone_src_locations.log")]
sources.append((log/"tracy_profile_log_host.tracy","tracy_profile_log_host.tracy"))
sources.append((report,"ops.csv"))
manifest=[]
for source,name in sources:
    target=dest/name
    row=digest(source)
    if name.endswith(".gz"):
        with source.open("rb") as inp,target.open("wb") as outf:
            with gzip.GzipFile(filename="",mode="wb",fileobj=outf,mtime=0) as out:
                shutil.copyfileobj(inp,out)
    else: shutil.copyfile(source,target)
    row["archive"]=digest(target)
    manifest.append(row)
write_json(p/"raw_manifest.json",manifest)
kernels=set()
with report.open() as f:
    for row in csv.DictReader(f):
        for key in ("COMPUTE KERNEL SOURCE","DATA MOVEMENT KERNEL SOURCE"):
            kernels.update(ast.literal_eval(row[key].replace(";",",")))
write_json(p/"kernel_sources.json",[digest(METAL/name) for name in sorted(kernels)])
print(json.dumps({"archives":len(manifest),"archive_bytes":sum(r["archive"]["bytes"] for r in manifest),
                  "kernel_sources":len(kernels)}))
