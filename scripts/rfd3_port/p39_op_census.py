"""p39 — the per-callsite op census, priced at the per-op host cost p38 measured.

p35's ledger says 3418 ttnn ops/step and 83.0 ms of enqueue, and §10 asserts "36 DiT block
executions at ~22 ops each are most of the census". 36 x 22 = 792, which is 23% of 3418, so the
census was never actually taken. p38 then measured that the host cost of an enqueue is fixed per
op (shape-independent: 21.6 us at 32x32 and at 45 M elements), is NOT command-queue backpressure
(flat 21.8 us behind a 1.19 ms device op), and varies 8x by op type -- reshape 3.9 us, rms_norm
10.4, linear 13.4, multiply 21.6, typecast 26.3, softmax 26.7, sigmoid 27.0, permute 30.2.

So the op-count lever is not "delete ops", it is "delete the expensive ops". This takes the census
per CALLSITE with its measured host time, which turns fusion candidates into a priced, ordered list
instead of a guess.

Run: TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:rfd3-host-half PYTHONPATH=$PWD \
     python3 scripts/rfd3_port/p39_op_census.py --num_timesteps 8 --out perf/p39/census.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import ttnn  # noqa: E402

import tt_bio.rfd3.model as R  # noqa: E402
from tt_bio.rfd3.design import build_diffusion_module, build_token_initializer  # noqa: E402
from tt_bio.rfd3.featurize import featurize  # noqa: E402
from tt_bio.rfd3.input import InputSpecification  # noqa: E402
from tt_bio.rfd3.sampler import RFD3Sampler  # noqa: E402

TT_OPS = ("linear", "matmul", "add", "add_", "multiply", "multiply_", "subtract", "softmax",
          "typecast", "scatter", "embedding", "reshape", "permute", "pad", "to_layout", "concat",
          "rms_norm", "layer_norm", "sigmoid", "silu", "clone", "full", "from_torch", "to_torch",
          "transpose", "repeat", "sum", "mean", "exp", "where", "zeros", "arange", "slice",
          "unsqueeze", "squeeze", "gelu", "relu", "div", "tanh", "experimental")

ACC: dict[tuple, float] = defaultdict(float)
CNT: dict[tuple, int] = defaultdict(int)
SNAP: list = []


def _wrap(name):
    fn = getattr(ttnn, name, None)
    if fn is None or not callable(fn):
        return
    def w(*a, **k):
        fr = sys._getframe(1)
        key = (name, fr.f_code.co_filename.rsplit("/", 1)[-1], fr.f_code.co_name, fr.f_lineno)
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            ACC[key] += time.perf_counter() - t0
            CNT[key] += 1
    setattr(ttnn, name, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", default="scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb")
    ap.add_argument("--contig", default="A1-10,230,A31-40")
    ap.add_argument("--ckpt", default="/home/ttuser/.boltz/rfd3/weights")
    ap.add_argument("--num_timesteps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    for n in TT_OPS:
        _wrap(n)

    spec = InputSpecification.from_dict({"input": a.pdb, "contig": a.contig})
    spec.validate()
    f = featurize(a.pdb, spec)
    cap = Path(a.ckpt)
    ti_w = torch.load(cap / "token_initializer.real_weights.pt", map_location="cpu", weights_only=True)
    dm_w = torch.load(cap / "diffusion_module.real_weights.pt", map_location="cpu", weights_only=True)
    dev_ti = build_token_initializer(ti_w)
    dev_dm = build_diffusion_module(dm_w)
    with torch.no_grad():
        init = dev_ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
    L = init["Q_L_init"].shape[0]
    coord0 = f["motif_pos"].float().unsqueeze(0) if "motif_pos" in f else torch.zeros(1, L, 3)
    ACC.clear(); CNT.clear()

    cls = type(dev_dm)
    dm_call = cls.__call__
    def stepped(self, *ar, **kw):
        t0 = time.perf_counter()
        try:
            return dm_call(self, *ar, **kw)
        finally:
            SNAP.append((time.perf_counter() - t0, dict(ACC), dict(CNT)))
    cls.__call__ = stepped

    sampler = RFD3Sampler(num_timesteps=a.num_timesteps)
    t0 = time.perf_counter()
    with torch.no_grad():
        sampler.sample(dev_dm, 1, L, coord0, f, init,
                       f["is_motif_atom_with_fixed_coord"],
                       generator=torch.Generator().manual_seed(a.seed))
    wall = time.perf_counter() - t0

    keys = set().union(*[s[1].keys() for s in SNAP])
    med_ms, med_cnt = {}, {}
    for k in keys:
        med_ms[k] = statistics.median(
            [(SNAP[i][1].get(k, 0.0) - SNAP[i - 1][1].get(k, 0.0)) * 1e3
             for i in range(2, len(SNAP))])
        med_cnt[k] = statistics.median(
            [SNAP[i][2].get(k, 0) - SNAP[i - 1][2].get(k, 0) for i in range(2, len(SNAP))])
    step_walls = [s[0] * 1e3 for s in SNAP]
    med_step = statistics.median(step_walls[2:])

    rows = sorted(med_ms.items(), key=lambda kv: -kv[1])
    tot_ms = sum(med_ms.values())
    tot_n = sum(med_cnt.values())
    print(f"\nL={L} atoms  wall={wall:.1f}s  median warm step={med_step:.1f} ms  "
          f"ops/step={tot_n:.0f}  wrapped host time={tot_ms:.1f} ms/step\n")
    print(f"{'op':12s} {'caller':38s} {'line':>6s} {'n/step':>7s} {'ms/step':>8s} {'us/call':>8s}")
    for (op, fname, fn, ln), ms in rows[:55]:
        n = med_cnt[(op, fname, fn, ln)]
        if ms < 0.10:
            continue
        print(f"{op:12s} {fname + ':' + fn:38.38s} {ln:6d} {n:7.0f} {ms:8.3f} "
              f"{1000 * ms / max(n, 1):8.1f}")

    # rollup by op name
    by_op = defaultdict(lambda: [0.0, 0])
    for (op, _f, _fn, _ln), ms in med_ms.items():
        by_op[op][0] += ms
        by_op[op][1] += med_cnt[(op, _f, _fn, _ln)]
    print(f"\n{'op':14s} {'n/step':>7s} {'ms/step':>8s} {'us/call':>8s}")
    for op, (ms, n) in sorted(by_op.items(), key=lambda kv: -kv[1][0]):
        if ms < 0.05:
            continue
        print(f"{op:14s} {n:7.0f} {ms:8.3f} {1000 * ms / max(n, 1):8.1f}")

    # rollup by caller function
    by_fn = defaultdict(lambda: [0.0, 0])
    for (op, _f, fn, _ln), ms in med_ms.items():
        by_fn[fn][0] += ms
        by_fn[fn][1] += med_cnt[(op, _f, fn, _ln)]
    print(f"\n{'caller fn':44s} {'n/step':>7s} {'ms/step':>8s}")
    for fn, (ms, n) in sorted(by_fn.items(), key=lambda kv: -kv[1][0]):
        if ms < 0.05:
            continue
        print(f"{fn:44.44s} {n:7.0f} {ms:8.3f}")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({
            "atoms": L, "median_warm_step_ms": med_step, "wall_s": wall,
            "ops_per_step": tot_n, "wrapped_ms_per_step": tot_ms,
            "callsites": [{"op": k[0], "file": k[1], "fn": k[2], "line": k[3],
                           "n": med_cnt[k], "ms": med_ms[k]} for k in med_ms],
        }, indent=2))
        print(f"\n[done] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
