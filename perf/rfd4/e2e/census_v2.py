"""census/e2e-128-v2.json from an elt_prof run: per-family device time, ops and FLOPs per trunk pass.

    census_v2.py <prof_dir> <out.json>

FLOPs are counted from the profiled shapes, not a model: 2*M*K*N per matmul (batch dims multiplied
in), 4*b*h*n*n*d per SDPA. Pass 2 (a recycled pass) is the steady pass; a design is 60 first passes
and 120 recycled ones at thinking_medium, so per design = (60 * first + 120 * recycled) / batch, a pass carrying
`batch` designs (run.json's batch, elt_prof.py --batch). Per-pass figures are for the whole batched pass.
"""
import collections
import csv
import json
import sys


def dims(r, k):
    return [int(r.get(f"{k}_{a}_PAD[LOGICAL]", "1").split("[")[0] or 1) for a in "WZYX"]


def flops(r):
    code = r["OP CODE"]
    if "Matmul" in code:
        a, b = dims(r, "INPUT_0"), dims(r, "INPUT_1")
        return 2 * a[0] * a[1] * a[2] * a[3] * b[3]
    if "SDPA" in code:
        q = dims(r, "INPUT_0")
        return 4 * q[0] * q[1] * q[2] * q[2] * q[3]
    return 0


def main():
    prof, out = sys.argv[1], sys.argv[2]
    rows = list(csv.DictReader(open(f"{prof}/ops.csv")))
    run = json.load(open(f"{prof}/run.json"))
    nb = run.get("batch", 1)
    # The recycle projections run between two passes; they are charged to the pass they feed.
    on, stack, passes, pending = False, [], [], collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        if r["OP TYPE"] == "signpost":
            c = r["OP CODE"]
            on = {"elt_prof": True, "elt_end": False}.get(c, on)
            if on and c.startswith("B|"):
                stack.append(c[2:])
                if c == "B|pass":
                    passes.append(pending)
                    pending = collections.defaultdict(lambda: collections.Counter())
            elif on and c.startswith("E|"):
                stack.pop()
            continue
        if not on:
            continue
        fam = stack[-1] if stack else "outside"
        fam = "pass_glue" if fam == "pass" else fam
        f = (passes[-1] if "pass" in stack else pending)[fam]
        ms = float(r["DEVICE KERNEL DURATION [ns]"] or 0) / 1e6
        f["ms"] += ms
        f["ops"] += 1
        f["gflop"] += flops(r) / 1e9
        if any(k in r["OP CODE"] for k in ("Binary", "Unary", "Typecast")):
            f["elt_ms"] += ms
            f["elt_ops"] += 1
    first, rec = passes[0], passes[-1]
    fams = {}
    for name in sorted(set(first) | set(rec), key=lambda k: -rec[k]["ms"]):
        a, b = first[name], rec[name]
        design_s = (60 * a["ms"] + 120 * b["ms"]) / 1e3 / nb
        fams[name] = {"ms_per_pass": round(b["ms"], 4), "ops_per_pass": b["ops"],
                      "elementwise_ms_per_pass": round(b["elt_ms"], 4), "elementwise_ops_per_pass": b["elt_ops"],
                      "gflop_per_pass": round(b["gflop"], 3),
                      "tflops": round(b["gflop"] / b["ms"], 2) if b["ms"] and b["gflop"] else 0.0,
                      "device_s_per_design": round(design_s, 3)}
    tot = lambda p, k: sum(v[k] for v in p.values())  # noqa: E731
    dev_s = (60 * tot(first, "ms") + 120 * tot(rec, "ms")) / 1e3 / nb
    for v in fams.values():
        v["share_of_device"] = round(v["device_s_per_design"] / dev_s, 4)
    doc = {
        "size": run["n"], "recipe": "thinking_medium", "trunk_passes": 180, "first_passes": 60, "recycled_passes": 120,
        "batch": nb, "dims": {"token": run["tok"], "pair": run["pair"], "cond": run["cond"]},
        "instrument": "device profiler (Tracy build tt-metal-b2z = ttnn 0.68.0), DEVICE KERNEL DURATION per op, "
                      "eager trunk call of 2 passes on random inputs of the design shapes (perf/rfd4/e2e/elt_prof.py); "
                      "families by signposts around each sublayer, innermost wins; gaps between kernels not counted",
        "aiclk": run["aiclk"], "host_load": round(run["load"], 1), "card": f"qb2 physical {run["card"]}",
        "per_pass": {"first": {"device_ms": round(tot(first, "ms"), 3), "ops": tot(first, "ops")},
                     "recycled": {"device_ms": round(tot(rec, "ms"), 3), "ops": tot(rec, "ops"),
                                  "elementwise_ms": round(tot(rec, "elt_ms"), 3), "elementwise_ops": tot(rec, "elt_ops"),
                                  "gflop": round(tot(rec, "gflop"), 2)}},
        "device_s_per_design": round(dev_s, 3),
        "tflop_per_design": round((60 * tot(first, "gflop") + 120 * tot(rec, "gflop")) / 1e3 / nb, 2),
        "families": fams,
    }
    json.dump(doc, open(out, "w"), indent=1)
    print(json.dumps({k: doc[k] for k in ("per_pass", "device_s_per_design", "tflop_per_design")}))


if __name__ == "__main__":
    main()
