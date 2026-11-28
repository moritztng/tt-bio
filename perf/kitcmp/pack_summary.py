"""Summarise a box_bc4_pack.sh run: per arm and phase, what N concurrent BindCraft 2 trajectories on one GPU retire.
    python pack_summary.py RESULTS/bc4 [--json OUT]
Per phase: s/it per trajectory (design_s / iterations, as bc2_gpu.py times it), design iterations per second summed
over the slots that ran (each slot's iterations / its design_s), the same over the phase's wall clock (validation
included, so it is what a campaign sees), trajectories per hour, and the GPU's busy fraction, SM clock and memory over
the phase from the 1 Hz nvidia-smi log. gain = summed it/s against the same arm's n1 phase."""
import argparse, json, statistics as st
from datetime import datetime
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("root", type=Path)
ap.add_argument("--json", type=Path)
a = ap.parse_args()

smi = []
for ln in open(a.root / "smi.csv", errors="replace"):
    p = [s.strip() for s in ln.split(",")]
    try:
        smi.append((datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").timestamp(), float(p[1].split()[0]),
                    float(p[3].split()[0]), float(p[6].split()[0]), float(p[9].split()[0])))
    except (ValueError, IndexError):
        pass


def window(t0, t1):
    w = [r for r in smi if t0 <= r[0] <= t1]
    if not w:
        return {}
    return dict(sm_mhz=st.median(r[1] for r in w), sm_mhz_min=min(r[1] for r in w), power_w=round(st.mean(r[2] for r in w)),
                gpu_busy=round(st.mean(r[3] for r in w) / 100, 3), mem_mib=max(r[4] for r in w))


out = {}
for arm in sorted(p.parent.name for p in a.root.glob("pack/*/phases.tsv")):
    G = a.root / "pack" / arm
    reps, start = [], None
    for f in sorted(G.glob("s*/runs.jsonl")):
        for ln in open(f):
            r = json.loads(ln)
            if r["ev"] == "start":
                start = r
            if r["ev"] == "rep" and r.get("s_per_iter"):
                reps.append(dict(r, slot=int(f.parent.name[1:])))
    out[arm] = dict(gpu=start["gpu"], bc2_sha=start["bc2_sha"], jax=start["jax"], cpu=start["cpu"], ncpu=start["ncpu"],
                    phases={})
    for ln in open(G / "phases.tsv"):
        name, n, w, t0, t1 = ln.split()
        rs = [r for r in reps if r.get("phase") == name]
        if not rs:
            continue
        slots = sorted({r["slot"] for r in rs})
        its = sum(r["iterations"] for r in rs)
        wall = float(t1) - float(t0)
        spi = sorted(r["s_per_iter"] for r in rs)
        d0, d1 = min(r["t0w"] for r in rs), max(r["t0w"] + r["design_s"] for r in rs)
        out[arm]["phases"][name] = dict(
            n=int(n), per_slot=int(w), trajectories=len(rs), iterations=sorted(r["iterations"] for r in rs),
            errors=sum(bool(r.get("err")) for r in rs),
            s_per_iter=dict(median=round(st.median(spi), 4), min=round(spi[0], 4), max=round(spi[-1], 4)),
            it_per_s_design=round(sum(sum(r["iterations"] for r in rs if r["slot"] == j) /
                                      sum(r["design_s"] for r in rs if r["slot"] == j) for j in slots), 4),
            it_per_s_wall=round(its / wall, 4), traj_per_h=round(len(rs) * 3600 / wall, 2), wall_s=round(wall, 1),
            loadavg_max=max((max(r["loadavg"]) for r in rs if r.get("loadavg")), default=None), **window(d0, d1))
    ph = out[arm]["phases"]
    if "n1" in ph:
        for v in ph.values():
            v["gain_design"] = round(v["it_per_s_design"] / ph["n1"]["it_per_s_design"], 3)
            v["gain_wall"] = round(v["it_per_s_wall"] / ph["n1"]["it_per_s_wall"], 3)

if a.json:
    a.json.write_text(json.dumps(out, indent=1))
for arm, d in out.items():
    print(f"{arm}: {d['gpu']} | {d['cpu']} x{d['ncpu']} | BC2 {d['bc2_sha'][:9]} JAX {d['jax']}")
    for name, v in d["phases"].items():
        s = v["s_per_iter"]
        print(f"  {name:5} N={v['n']} traj={v['trajectories']} it={v['iterations']} s/it {s['median']} [{s['min']}, {s['max']}]"
              f" it/s design {v['it_per_s_design']} (x{v.get('gain_design')}) wall {v['it_per_s_wall']} (x{v.get('gain_wall')})"
              f" traj/h {v['traj_per_h']} busy {v.get('gpu_busy')} SM {v.get('sm_mhz')} MHz {v.get('power_w')} W"
              f" mem {v.get('mem_mib')} MiB load {v['loadavg_max']} err {v['errors']}")
