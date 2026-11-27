"""Host ceiling of a docking rollout, card-free (row dfd-scaling).

How many chips can one host keep fed? Each chip needs a host process doing the score step's
host half: edge features, harmonics, masks, crops, and the per-call graph prep the card layers
ask for. This runs that host half for real with the card replaced by a stub, at 1/2/4/8
concurrent pipelines, and reports CPU-seconds per step and the knee.

The stub is a stand-in ``ttnn`` whose ``from_torch``/``to_torch`` are the identity and whose
conv layer returns zeros of the right shape, so every line of ``static_forward`` and of
``DeviceConvModule.forward`` runs, including the dense scatter matrix it builds each call. What
it cannot see is ttnn's own host cost (tilize, op dispatch), which depends on the kernel: that
part is the card run's ``host_cpu_s_per_step`` minus this one.

    PYTHONPATH=<worktree> python3 scripts/diffdock_port/host_ceiling.py \
        --fixture fx_8G43.pt --pipelines 1,2,4,8 --steps 19 --out host.json
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
import types
from pathlib import Path

CKPT = Path(os.environ.get("DFD_CKPT", str(Path.home() / "scratch/dfd/ckpt/extracted")))


def fake_ttnn():
    """Identity transfers, no device. Installed before anything imports ttnn."""
    m = types.ModuleType("ttnn")
    m.TILE_LAYOUT, m.ROW_MAJOR_LAYOUT = "TILE", "RM"
    m.float32, m.bfloat16, m.uint32 = "f32", "bf16", "u32"
    m.from_torch = lambda t, **kw: t
    m.to_torch = lambda t, **kw: t
    sys.modules["ttnn"] = m


class StubLayer:
    """DeviceConvLayer's host half (graph upload prep) with the arithmetic replaced by zeros."""

    def __init__(self, out_pad, dev_s):
        import ttnn
        from tt_bio.diffdock.device import DeviceConvLayer
        self.ttnn, self.device, self.out_pad, self.dev_s = ttnn, None, out_pad, dev_s
        self.graph = DeviceConvLayer.graph.__get__(self)

    def __call__(self, node_attr, g, edge_attr_groups, edge_sh, edge_mask):
        import torch
        if self.dev_s:
            time.sleep(self.dev_s)
        return torch.zeros(g["n_out"], self.out_pad)


def worker(args):
    import tt_bio  # noqa: F401  (its import-time overlay check looks for the real ttnn spec)
    fake_ttnn()
    import torch
    if args.threads:
        torch.set_num_threads(args.threads)
    from tt_bio.diffdock import config as C
    from tt_bio.diffdock.conv import TensorProductConvLayer, pad_nodes
    from tt_bio.diffdock.device import DeviceConvModule, pad32
    from tt_bio.diffdock.score_model import HostScoreNetwork, load_released
    from tt_bio.diffdock.static_net import StaticComplex, static_forward

    fx = torch.load(args.fixture, weights_only=False)
    cfg = C.load(str(CKPT / "score_model"), old_model=False)
    sd = torch.load(CKPT / "score_model/best_ema_inference_epoch_model.pt", map_location="cpu",
                    weights_only=False)
    card = load_released(sd, cfg)
    t_g = time.perf_counter()
    graph = fx["graph"]
    rec_x = HostScoreNetwork(load_released(sd, cfg), lm_embeddings=fx["emb"])._receptor_x(
        graph.receptor)
    n_convs = 0
    for _, mod in list(card.named_modules()):
        for name, child in list(mod._modules.items()):
            if isinstance(child, TensorProductConvLayer):
                stub = DeviceConvModule.__new__(DeviceConvModule)
                torch.nn.Module.__init__(stub)
                stub.name, stub.log, stub.secs, stub.host, stub.layer = name, [], 0.0, None, child
                stub.dev = StubLayer(pad32(child.out_irreps.dim), args.dev_s)
                mod._modules[name] = stub
                n_convs += 1

    keys, P = sorted(fx["steps"]), args.poses
    n_lig = fx["steps"][keys[0]].lig_pos.shape[1]
    c = StaticComplex.build(graph.ligand, rec_x, graph.receptor["pos"].float(),
                            graph.receptor["edge_index"], P)
    build_s = time.perf_counter() - t_g
    prepared = []
    for k in keys:
        st = fx["steps"][k]
        poses = st.lig_pos[torch.arange(P) % st.lig_pos.shape[0]]
        prepared.append((pad_nodes(poses.reshape(P * n_lig, 3).float(), c.sg.n_lig_padded),
                         (st.t_tr, st.tr_sigma, st.rot_sigma, st.tor_sigma), st.crop_cutoff))

    rundir = Path(args.rundir)
    with torch.no_grad():
        static_forward(card, c, *prepared[0][:1], *prepared[0][1], crop_cutoff=prepared[0][2])
        (rundir / f"ready.{args.idx}").write_text(str(os.getpid()))
        while not (rundir / "go").exists():
            time.sleep(0.01)
        ru0 = resource.getrusage(resource.RUSAGE_SELF)
        rows = []
        for i in range(args.steps):
            lig_pos, sig, crop = prepared[i % len(prepared)]
            t0 = time.time()
            static_forward(card, c, lig_pos, *sig, crop_cutoff=crop)
            rows.append({"t0": t0, "t1": time.time()})
        ru1 = resource.getrusage(resource.RUSAGE_SELF)
    (rundir / f"worker.{args.idx}.json").write_text(json.dumps({
        "pid": os.getpid(), "torch_threads": torch.get_num_threads(), "convs_stubbed": n_convs,
        "complex_build_s": build_s, "maxrss_mb": ru1.ru_maxrss / 1024,
        "cpu_s": (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime), "steps": rows}))


def box_cpu():
    v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    return sum(v), v[3] + v[4]


def run_n(args, n):
    rundir = Path(args.rundir) / f"n{n}"
    rundir.mkdir(parents=True, exist_ok=True)
    for f in rundir.glob("*"):
        f.unlink()
    procs = [subprocess.Popen(
        [sys.executable, __file__, "--worker", "--idx", str(i), "--fixture", args.fixture,
         "--steps", str(args.steps), "--poses", str(args.poses), "--rundir", str(rundir),
         "--threads", str(args.threads), "--dev-s", str(args.dev_s)],
        stdout=open(rundir / f"worker.{i}.log", "w"), stderr=subprocess.STDOUT) for i in range(n)]
    while not all((rundir / f"ready.{i}").exists() for i in range(n)):
        if any(p.poll() not in (None, 0) for p in procs):
            raise SystemExit(f"a worker died; see {rundir}")
        time.sleep(0.2)
    c0, t0 = box_cpu(), time.time()
    (rundir / "go").write_text("")
    if any(p.wait() for p in procs):
        raise SystemExit(f"a worker failed; see {rundir}")
    c1, wall = box_cpu(), time.time() - t0
    ws = [json.loads((rundir / f"worker.{i}.json").read_text()) for i in range(n)]
    lo, hi = max(w["steps"][0]["t0"] for w in ws), min(w["steps"][-1]["t1"] for w in ws)
    step_s = sorted(s["t1"] - s["t0"] for w in ws for s in w["steps"])
    own = sum(w["cpu_s"] for w in ws)
    nproc = os.cpu_count()
    box_cores = (1 - (c1[1] - c0[1]) / (c1[0] - c0[0])) * nproc
    return {"pipelines": n, "median_step_wall_s": step_s[len(step_s) // 2],
            "cpu_s_per_step": own / sum(len(w["steps"]) for w in ws),
            "own_cores": own / wall, "box_busy_cores": box_cores,
            "external_cores": box_cores - own / wall, "overlap_s": hi - lo, "wall_s": wall,
            "torch_threads": ws[0]["torch_threads"], "complex_build_s": ws[0]["complex_build_s"],
            "maxrss_mb": max(w["maxrss_mb"] for w in ws), "convs_stubbed": ws[0]["convs_stubbed"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--pipelines", default="1,2,4,8")
    ap.add_argument("--steps", type=int, default=19)
    ap.add_argument("--poses", type=int, default=10, help="PROTOCOL.md samples_per_complex")
    ap.add_argument("--threads", type=int, default=1, help="torch threads per pipeline, 0 = torch default")
    ap.add_argument("--dev-s", type=float, default=0.0, help="stub seconds per conv call")
    ap.add_argument("--rundir", default="/tmp/dfd_host_ceiling")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.worker:
        return worker(args)
    res = []
    for n in [int(x) for x in args.pipelines.split(",")]:
        res.append(run_n(args, n))
        print(json.dumps(res[-1]), flush=True)
    base = res[0]["median_step_wall_s"] / res[0]["pipelines"] if res else None
    out = {"host": os.uname().nodename, "nproc": os.cpu_count(), "args": vars(args), "runs": res}
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
