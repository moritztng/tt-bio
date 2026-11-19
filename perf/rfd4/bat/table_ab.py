import json,sys,glob,os
# ms/design/pass from the depth-3 traced call, the metric state/bat/MECHANISM.md uses: call/3/b.
rows=[]
for f in sorted(glob.glob(os.path.expanduser("~/bat-lever/*/[a-z]*-n*.json"))):
    d=json.load(open(f)); run=os.path.basename(os.path.dirname(f)); nm=os.path.basename(f)[:-5]
    arm=nm.split("-n")[0]; b=d["batch"]
    clk=(d["aiclk"].get("0") or {})
    rows.append((run,arm,d["n"],d["real"],b,d["call_traced_ms"]["median"],
                 d["call_traced_ms"]["median"]/3/b, d["ops_per_pass"], clk.get("median"), clk.get("min")))
print(f"{'run':6s} {'arm':7s} {'n':>4s} {'real':>4s} {'b':>3s} {'call ms':>9s} {'ms/des/pass':>12s} {'ops':>5s} {'clk':>5s} {'min':>5s}")
for r in rows: print(f"{r[0]:6s} {r[1]:7s} {r[2]:4d} {r[3]:4d} {r[4]:3d} {r[5]:9.1f} {r[6]:12.2f} {r[7]:5d} {str(r[8]):>5s} {str(r[9]):>5s}")
