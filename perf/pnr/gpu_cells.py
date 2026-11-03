"""Reduce one PNR GPU session (perf/pnr/gpu/<label>/box, pulled by pnr_session.sh) to its cells in GPU-CELLS.json.

    python3 gpu_cells.py --gpu a100|h200|b200 --label LABEL [--out /home/moritz/.coworker/state/pnr/GPU-CELLS.json]

Per model, every arm the box measured: median seconds per fold / design over its warm runs, n, spread_pct
((max - min) / median), batch, SM clock and power medians from the session's 500 ms nvidia-smi log over exactly
those runs' windows, and the raw path. The chosen arm is the fastest with n >= 5 (PROTOCOL.md). BindCraft 2 counts
only warm trajectories that ran all 140 iterations; RFD4 publishes each arm's final rerun, its ladder rungs only
pick the batch. Nothing is typed in by hand: every number here comes from a file under the label.
"""
import argparse, json, re, statistics
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEED_N, BC2_ITERS = 5, 140


def jl(p):
    out = []
    for line in p.read_text().splitlines() if p.exists() else []:
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def iso(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


class Smi:
    def __init__(self, path):
        self.rows = []
        for line in path.read_text().splitlines() if path.exists() else []:
            p = [x.strip() for x in line.split(",")]
            try:
                t = datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
                self.rows.append((t, float(p[1].split()[0]), float(p[3].split()[0])))
            except (ValueError, IndexError):
                pass

    def over(self, wins):
        sm = [r[1] for r in self.rows if any(a <= r[0] <= b for a, b in wins)]
        pw = [r[2] for r in self.rows if any(a <= r[0] <= b for a, b in wins)]
        return {"sm_clock_mhz": statistics.median(sm) if sm else None, "sm_clock_min": min(sm) if sm else None,
                "power_w": round(statistics.median(pw), 1) if pw else None, "smi_samples": len(sm)}


def cell(vals, wins, smi, **kw):
    if not vals:
        return {"n": 0, **kw}
    m = statistics.median(vals)
    return {"n": len(vals), "median": round(m, 4), "spread_pct": round(100 * (max(vals) - min(vals)) / m, 2),
            "values": [round(v, 3) for v in vals], **smi.over(wins), **kw}


def mol1(box, smi):
    arms = {}
    for d in sorted((box / "mol1/arms").glob("512-*-k*")):
        if not d.is_dir():
            continue
        rs = jl(d / "runs.jsonl")
        w = [r for r in rs if r["kind"] == "warm" and r.get("status") != "FAILED" and r["predictions"] == 1]
        arm = rs[0]["arm"] if rs else d.name
        c = cell([r["wall_s"] for r in w], [(iso(r["t"]) - r["wall_s"], iso(r["t"])) for r in w], smi,
                 unit="s_per_fold", batch=1, knobs=(rs[0].get("knobs") if rs else None),
                 cold_s=[r["wall_s"] for r in rs if r["kind"] == "cold"], raw=str(d / "runs.jsonl"))
        if any(r.get("status") == "FAILED" for r in rs):
            c["error"] = next(r.get("error") for r in rs if r.get("status") == "FAILED")[-400:]
        arms[arm] = c
    return arms


def bc2(box, smi):
    arms = {}
    dirs = sorted((box / "bc2/arms").glob("*")) + [box / "bc2/winner"]
    res = json.loads((box / "bc2/ARMS.json").read_text()) if (box / "bc2/ARMS.json").exists() else {}
    for d in dirs:
        rs = jl(d / "runs.jsonl")
        reps = [r for r in rs if r.get("ev") == "rep" and r.get("kind") == "warm" and not r.get("err")]
        full = [r for r in reps if r.get("iterations") == BC2_ITERS and (r.get("design_s") or 0) > 0]
        name = d.name if d.name != "winner" else f"winner:{(res.get('_winner') or {}).get('arm')}"
        wins = [(r["t"] - r["traj_s"], r["t"] - r["traj_s"] + r["design_s"]) for r in full]
        arms[name] = cell([r["design_s"] for r in full], wins, smi, unit="s_per_design", batch=1,
                          s_per_it=round(statistics.median(r["design_s"] / r["iterations"] for r in full), 4) if full else None,
                          validate_s=round(statistics.median(r["validate_s"] for r in full if r.get("validate_s")), 2)
                          if any(r.get("validate_s") for r in full) else None,
                          warm_trajectories=len(reps), iterations_seen=sorted({r.get("iterations") for r in reps}),
                          settings=(res.get(d.name) or res.get("_winner") or {}).get("settings"), raw=str(d / "runs.jsonl"))
    return arms


def rfd4(box, smi):
    arms = {}
    for d in sorted((box / "rfd4").glob("s5*-*/224-b*")):
        rs = jl(d / "runs.jsonl")
        w = [r for r in rs if r["kind"] == "warm" and r.get("status") != "FAILED" and r["designs"] > 0]
        stage = "final" if d.parent.name.startswith("s5final-") else "rung"
        arm = d.parent.name.split("-", 1)[1]
        b = int(d.name.split("-b")[1])
        c = cell([r["wall_s"] / r["designs"] for r in w], [(iso(r["t"]) - r["wall_s"], iso(r["t"])) for r in w], smi,
                 unit="s_per_design", batch=b, stage=stage, raw=str(d / "runs.jsonl"),
                 **({"error": next(r.get("error") for r in rs if r.get("status") == "FAILED")[-300:]}
                    if any(r.get("status") == "FAILED" for r in rs) else {}))
        arms[f"{arm}@b{b}" + (":final" if stage == "final" else "")] = c
    return arms


def versions(box):
    q = (box / "nvidia-smi-q.txt").read_text() if (box / "nvidia-smi-q.txt").exists() else ""
    g = lambda k: (re.search(rf"{k}\s*:\s*(.+)", q) or [None, None])[1]

    def pin(f, *pk):
        t = (box / f).read_text() if (box / f).exists() else ""
        return {k: (re.search(rf"^{re.escape(k)}(?:==| @ )(.+)$", t, re.M) or [None, None])[1] for k in pk}
    inst = (box / "instance").read_text().split() if (box / "instance").exists() else []
    return {"gpu": g("Product Name"), "driver": g("Driver Version"), "cuda": g("CUDA Version"),
            "image": inst[2] if len(inst) > 2 else None, "box": inst[0] if inst else None,
            "mol1": {"commit": "e6e64866ee89640a91cef7ee777a85f1d8be03bb", **pin("mol1/pip-freeze.txt", "torch", "cuequivariance-ops-torch-cu13", "cuequivariance-torch")},
            "bindcraft2": {"commit": ((box / "bc2/commit").read_text().strip() if (box / "bc2/commit").exists() else None),
                           **pin("bc2/pip-freeze.txt", "jax", "jaxlib", "jax-cuda13-plugin")},
            "rfd4": {"commit": "2db07bb444d91695f8c10e6acb7648e9297f9221", "ckpt_rev": "a06e2509c7033b010fc43614e9fd840303588502",
                     **pin("rfd4/pip-freeze.txt", "torch", "cuequivariance-torch")}}


def chosen(arms, final_only=False):
    ok = {k: v for k, v in arms.items() if v["n"] >= NEED_N and (not final_only or k.endswith(":final"))}
    return min(ok, key=lambda k: ok[k]["median"]) if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, default=Path("/home/moritz/.coworker/state/pnr/GPU-CELLS.json"))
    a = ap.parse_args()
    box = HERE / "gpu" / a.label / "box"
    smi = Smi(box / "smi.csv")
    when = datetime.fromtimestamp(smi.rows[0][0], timezone.utc).strftime("%FT%TZ") if smi.rows else None
    v = versions(box)
    models = {"mol1": mol1(box, smi), "bindcraft2": bc2(box, smi), "rfd4": rfd4(box, smi)}
    g = {"label": a.label, "when": when, "box": v["box"], "versions": v, "raw": str(box), "models": {}}
    for m, arms in models.items():
        c = chosen(arms, final_only=(m == "rfd4"))
        g["models"][m] = {"arms": arms, "chosen": c, "cell": arms.get(c) if c else None}
    doc = json.loads(a.out.read_text()) if a.out.exists() else {"protocol": "state/pnr/PROTOCOL.md", "gpus": {}}
    doc["gpus"][a.gpu] = g
    a.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({m: (x["chosen"], (x["cell"] or {}).get("median"), (x["cell"] or {}).get("n"))
                      for m, x in g["models"].items()}))


if __name__ == "__main__":
    main()
