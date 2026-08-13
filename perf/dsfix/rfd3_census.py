"""RFD3 TT dispatch census: programs per diffusion step, K, per rung.

Same differential trick the timing uses, applied to the program count. Capture the graph of a
whole `sampler.sample` at N=1 and again at N=2 timesteps; model load, featurisation, per-design
device setup and the final untilize are identical in both, so

    K = count(N=2) - count(N=1)

is the per-step program count with every one-off cancelled exactly. That avoids having to reach
into the sampler loop and guess where a step begins.

Two views are counted because ttnn reports both: the python wrapper level (`ttnn.<op>`) and the
device primitive level (`ttnn::prim::...`). Allocations and views are not programs and are
excluded from both. D = K * t_d / step_wall is computed against the measured t_d and the measured
step wall from the ladder.
"""
import json, os, pathlib, sys
from collections import Counter

import torch
import ttnn

sys.path.insert(0, os.getcwd())
from tt_bio.rfd3.design import build_token_initializer, build_diffusion_module
from tt_bio.rfd3.sampler import RFD3Sampler
from tt_bio.rfd3.input import InputSpecification
from tt_bio.rfd3.featurize import featurize

CKPT = pathlib.Path("/home/ttuser/.boltz/rfd3/weights")
OUT = pathlib.Path("perf/dsfix/results/rfd3_census.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)
T_D = 19.179e-6  # measured, perf/dsfix/results/td.json
RUNGS = sys.argv[1:] or ["R0", "R1", "R2", "R3", "R4"]

# Not programs: allocations, deallocations and metadata-only views.
SKIP = ("create_device_tensor", "Tensor::deallocate", "Tensor::reshape", "tt::tt_metal::create_device_tensor")

# Per-step wall at batch 1, MEASURED by perf/dsfix/rfd3_ladder.py on this card/stack.
WALL = {}
for line in pathlib.Path("perf/dsfix/results/rfd3_tt.jsonl").read_text().splitlines():
    if line.strip():
        r = json.loads(line)
        if r["batch"] == 1:
            WALL[r["rung"]] = r["per_step_s_median"]

done = set()
if OUT.exists():
    done = {json.loads(l)["rung"] for l in OUT.read_text().splitlines() if l.strip()}

dm = build_diffusion_module(torch.load(CKPT / "diffusion_module.real_weights.pt",
                                       map_location="cpu", weights_only=True))
ti = build_token_initializer(torch.load(CKPT / "token_initializer.real_weights.pt",
                                        map_location="cpu", weights_only=True))
print("[census] weights loaded", flush=True)


def counts(captured):
    py, dev = Counter(), Counter()
    for n in captured:
        if n.get("node_type") != "function_start":
            continue
        nm = (n.get("params") or {}).get("name", "?")
        if any(s in nm for s in SKIP):
            continue
        if nm.startswith("ttnn.prim.") or "::prim::" in nm:
            dev[nm] += 1
        elif nm.startswith("ttnn."):
            py[nm] += 1
    return py, dev


for rung in RUNGS:
    if rung in done:
        print("[census] %s already done" % rung, flush=True)
        continue
    specs = json.loads(pathlib.Path("perf/dsfix/fixtures/rfd3_%s.json" % rung).read_text())
    sid, sdict = next(iter(specs.items()))
    spec = InputSpecification.from_dict(sdict)
    f = featurize(spec.input, spec)
    with torch.no_grad():
        init = ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
    L = init["Q_L_init"].shape[0]
    is_motif = f["is_motif_atom_with_fixed_coord"]
    coord0 = f["motif_pos"].float().unsqueeze(0) if "motif_pos" in f else torch.zeros(1, L, 3)
    print("[census] %s featurised L=%d" % (rung, L), flush=True)

    def run(nsteps):
        smp = RFD3Sampler(num_timesteps=nsteps)
        g = [torch.Generator().manual_seed(11)]
        with torch.no_grad():
            return smp.sample(dm, 1, L, coord0, f, init, is_motif, generator=g)

    run(2)  # warm: compile every program shape before anything is counted
    cap = {}
    for n in (1, 2):
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        run(n)
        cap[n] = counts(ttnn.graph.end_graph_capture())

    py1, dev1 = cap[1]
    py2, dev2 = cap[2]
    k_py = sum(py2.values()) - sum(py1.values())
    k_dev = sum(dev2.values()) - sum(dev1.values())
    K = max(k_py, k_dev)
    wall = WALL.get(rung)
    D = (K * T_D / wall) if wall else None
    top = sorted(((py2[o] - py1[o], o) for o in set(py2) | set(py1)), reverse=True)[:8]
    rec = {"rung": rung, "atoms": L, "K_py": k_py, "K_dev": k_dev, "K": K,
           "t_d_s": T_D, "step_wall_s_b1": wall,
           "D": round(D, 4) if D else None,
           "us_per_program": round(wall / K * 1e6, 2) if wall and K else None,
           "top_ops": [{"op": o, "n": c} for c, o in top if c],
           "total_py_n1": sum(py1.values()), "total_py_n2": sum(py2.values()),
           "host": "qb1", "card": 0, "ttnn": "0.67.4"}
    with OUT.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print("[census] %s K=%d (py %d / dev %d) D=%s" % (rung, K, k_py, k_dev, rec["D"]), flush=True)
print("[census] ALL DONE", flush=True)
