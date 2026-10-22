"""Per-call sub-op table from tri_prof's ops report: median device kernel time per op position.

    tri_prof_table.py <ops_perf_results.csv>

Rows after the `tri_prof` signpost are cut into calls at each `tri_out` / `tri_in` signpost. Within a
module every call runs the same op sequence, so op i of each call is the same op; its median DEVICE
KERNEL DURATION over the calls is printed with its op code, input/output shapes and memory.
"""
import csv
import statistics
import sys


def main():
    rows = list(csv.DictReader(open(sys.argv[1])))
    calls, cur, on = {"tri_out": [], "tri_in": []}, None, False
    for r in rows:
        if r.get("OP TYPE") == "signpost":
            name = r["OP CODE"]
            if name == "tri_prof":
                on = True
            elif name == "tri_end":
                on, cur = False, None
            elif on and name in calls:
                cur = []
                calls[name].append(cur)
            continue
        if on and cur is not None:
            cur.append(r)
    for name, cs in calls.items():
        lens = {len(c) for c in cs}
        print(f"== {name}: {len(cs)} calls, ops per call {sorted(lens)}")
        if len(lens) != 1:
            continue
        tot = [sum(float(r["DEVICE KERNEL DURATION [ns]"] or 0) for r in c) / 1e6 for c in cs]
        print(f"   device kernel sum per call: median {statistics.median(tot):.4f} ms")
        for i in range(len(cs[0])):
            r = cs[0][i]
            ms = statistics.median(float(c[i]["DEVICE KERNEL DURATION [ns]"] or 0) for c in cs) / 1e6
            shp = lambda k: "x".join(r.get(f"{k}_{a}_PAD[LOGICAL]", "?").split("[")[0] for a in "WZYX")  # noqa: E731
            print(f"   {i:2d} {ms:8.4f} ms  {r['OP CODE'][:40]:40s} in0 {shp('INPUT_0')} {r.get('INPUT_0_DATATYPE','')} "
                  f"{r.get('INPUT_0_MEMORY','')[:14]} -> {shp('OUTPUT_0')} {r.get('OUTPUT_0_MEMORY','')[:14]} "
                  f"cores {r.get('CORE COUNT','')}")


if __name__ == "__main__":
    main()
